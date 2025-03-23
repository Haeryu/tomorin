const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const Context = @import("context.zig").Context;
const util = @import("util.zig");

pub fn getSpiralAlloc(comptime F: type, allocator: std.mem.Allocator, train: bool) !struct { x: []F, t: []usize } {
    const seed: u64 = if (train) 1984 else 2020;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    const num_data: usize = 100;
    const num_class: usize = 3;
    const input_dim: usize = 2;
    const data_size: usize = num_class * num_data;

    var x = try allocator.alloc(f32, data_size * input_dim);
    defer allocator.free(x);
    var t = try allocator.alloc(usize, data_size);
    defer allocator.free(t);

    for (0..num_class) |j| {
        for (0..num_data) |i| {
            const rate: F = @as(F, @floatFromInt(i)) / @as(F, @floatFromInt(num_data));
            const radius: F = 1.0 * rate;
            const theta: F = @as(F, @floatFromInt(j)) * 4.0 + 4.0 * rate + random.floatNorm(F) * 0.2;
            const ix: usize = num_data * j + i;
            x[ix * input_dim] = radius * std.math.sin(theta);
            x[ix * input_dim + 1] = radius * std.math.cos(theta);
            t[ix] = @intCast(j);
        }
    }

    var indices = try allocator.alloc(usize, data_size);
    defer allocator.free(indices);

    for (0..data_size) |i| {
        indices[i] = i;
    }
    random.shuffle(usize, indices);

    var shuffled_x = try allocator.alloc(F, data_size * input_dim);
    errdefer allocator.free(shuffled_x);
    var shuffled_t = try allocator.alloc(usize, data_size);
    errdefer allocator.free(shuffled_t);

    for (0..data_size) |i| {
        const src_ix = indices[i];
        shuffled_x[i * input_dim] = x[src_ix * input_dim];
        shuffled_x[i * input_dim + 1] = x[src_ix * input_dim + 1];
        shuffled_t[i] = t[src_ix];
    }

    return .{ .x = shuffled_x, .t = shuffled_t };
}

pub fn SpiralDataset(comptime T: type) type {
    return struct {
        x: []T,
        t: []usize,
        allocator: std.mem.Allocator,

        pub const num_data: usize = 100;
        pub const num_class: usize = 3;
        pub const input_dim: usize = 2;
        pub const data_size: usize = num_class * num_data;

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) !Self {
            const spiral = try getSpiralAlloc(T, allocator, true);
            errdefer allocator.free(spiral.x);
            errdefer allocator.free(spiral.t);

            return .{
                .x = spiral.x,
                .t = spiral.t,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.x);
            self.allocator.free(self.t);
        }

        pub fn write(
            self: *Self,
            i: usize,
            batch_i: usize,
            batch: [2]*GPUTensor(T),
            context: *Context,
        ) !void {
            const data, const tag = batch;

            const host_slice = self.x[batch_i * Self.input_dim .. (batch_i + 1) * Self.input_dim];
            var host_tag: [3]T = .{0.0} ** Self.num_class;
            host_tag[self.t[batch_i]] = 1.0; // one_hot

            try data.writeFromHostAsync(host_slice, i * Self.input_dim, context.stream);
            try tag.writeFromHostAsync(&host_tag, i * Self.num_class, context.stream);
        }

        pub fn len(_: *const Self) usize {
            return Self.num_data;
        }
    };
}

pub fn MNISTDataset(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        train: bool,
        label_data: LabelData,
        image_data: ImageData,

        const Self = @This();

        const LabelData = struct {
            labels: []u8,
            num_labels: u32,

            pub fn deinit(self: *LabelData, allocator: std.mem.Allocator) void {
                allocator.free(self.labels);
            }
        };

        const ImageData = struct {
            data: []u8,
            num_images: u32,
            rows: u32,
            columns: u32,

            pub fn deinit(self: *ImageData, allocator: std.mem.Allocator) void {
                allocator.free(self.data);
            }
        };

        pub fn init(
            allocator: std.mem.Allocator,
            train: bool,
        ) !Self {
            std.fs.cwd().makeDir("datasets/mnist") catch |err| {
                switch (err) {
                    error.PathAlreadyExists => {},
                    else => return err,
                }
            };

            var label_data = try loadLabels(allocator, if (train) "./datasets/mnist/train-labels-idx1-ubyte" else "./datasets/mnist/t10k-labels-idx1-ubyte");
            errdefer label_data.deinit(allocator);

            var image_data = try loadImages(allocator, if (train) "./datasets/mnist/train-images-idx3-ubyte" else "./datasets/mnist/t10k-images-idx3-ubyte");
            errdefer image_data.deinit(allocator);

            return .{
                .allocator = allocator,
                .train = train,
                .label_data = label_data,
                .image_data = image_data,
            };
        }

        pub fn deinit(self: *Self) void {
            self.image_data.deinit(self.allocator);
            self.label_data.deinit(self.allocator);
        }

        fn loadLabels(allocator: std.mem.Allocator, filepath: []const u8) !LabelData {
            var unzipped_file = try std.fs.cwd().openFile(filepath, .{});
            defer unzipped_file.close();

            const data = try unzipped_file.readToEndAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(data);

            var fbs = std.io.fixedBufferStream(data);
            const reader = fbs.reader();

            const magic = try reader.readInt(u32, .big);
            if (magic != 2049) return error.InvalidMagicNumber;

            const num_labels = try reader.readInt(u32, .big);

            const labels = try allocator.alloc(u8, num_labels);
            errdefer allocator.free(labels);

            const read_bytes = try reader.readAll(labels);
            if (read_bytes != num_labels) {
                return error.InvalidFile;
            }

            return LabelData{
                .labels = labels,
                .num_labels = num_labels,
            };
        }

        fn loadImages(allocator: std.mem.Allocator, filepath: []const u8) !ImageData {
            var unzipped_file = try std.fs.cwd().openFile(filepath, .{});
            defer unzipped_file.close();

            const data = try unzipped_file.readToEndAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(data);

            var fbs = std.io.fixedBufferStream(data);
            const reader = fbs.reader();

            const magic = try reader.readInt(u32, .big);
            if (magic != 2051) return error.InvalidMagicNumber;

            const num_images = try reader.readInt(u32, .big);
            const rows = try reader.readInt(u32, .big);
            const columns = try reader.readInt(u32, .big);

            if (rows != 28 or columns != 28) return error.InvalidDimensions;

            const data_size = num_images * rows * columns;

            const image_data = try allocator.alloc(u8, data_size);
            const read_bytes = try reader.readAll(image_data);
            errdefer allocator.free(image_data);
            if (read_bytes != data_size) {
                return error.InvalidFile;
            }

            return ImageData{
                .data = image_data,
                .num_images = num_images,
                .rows = rows,
                .columns = columns,
            };
        }

        pub fn write(
            self: *Self,
            i: usize,
            batch_i: usize,
            batch: [2]*GPUTensor(T),
            context: *Context,
        ) !void {
            const data, const tag = batch;

            const host_slice = self.image_data.data[batch_i * self.image_data.rows * self.image_data.columns .. (batch_i + 1) * self.image_data.rows * self.image_data.columns];
            var host_tag: [10]T = .{0.0} ** 10;
            host_tag[self.label_data.labels[batch_i]] = 1.0; // one_hot

            const host_slice_f = try self.allocator.alloc(T, host_slice.len);
            defer self.allocator.free(host_slice_f);

            for (host_slice, host_slice_f) |u, *f| {
                f.* = @floatFromInt(u);
            }

            try data.writeFromHostAsync(host_slice_f, i * self.image_data.rows * self.image_data.columns, context.stream);
            try tag.writeFromHostAsync(&host_tag, i * 10, context.stream);
        }

        pub fn len(self: *const Self) usize {
            return self.image_data.num_images;
        }
    };
}

pub fn CIFAR10Dataset(comptime T: type) type {
    return struct {
        allocator: std.mem.Allocator,
        train: bool,
        files: []CIFARFile, // Array of batch files
        total_records: usize, // Sum of records across all .bin files

        const Self = @This();

        // Original CIFAR-10 format
        pub const record_size = 3073; // 1 byte label + 3072 bytes image
        pub const input_height = 32;
        pub const input_width = 32;
        pub const input_channels = 3;

        // We want to shrink to 16×16×3:
        pub const out_height = 16;
        pub const out_width = 16;
        pub const out_channels = 3;
        pub const out_image_size = out_height * out_width * out_channels; // 16*16*3 = 768

        pub const num_classes = 10; // 10 classes in CIFAR-10

        /// Holds a path to a .bin file and how many 3073-byte records it contains.
        const CIFARFile = struct {
            path: []const u8,
            record_count: usize,
        };

        /// Initialize the dataset in "streaming" fashion, so we do not load all data at once.
        /// We'll just store file metadata. Then, in `write()`, we open the file and read only
        /// the relevant record.
        pub fn init(allocator: std.mem.Allocator, train: bool) !Self {
            // Make sure our parent directory exists.
            std.fs.cwd().makeDir("datasets/cifar-10/cifar-10-binary") catch |err| {
                switch (err) {
                    error.PathAlreadyExists => {},
                    else => return err,
                }
            };

            // Decide which .bin files to read from
            const file_list: []const []const u8 = if (train) &[_][]const u8{
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin",
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_2.bin",
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_3.bin",
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_4.bin",
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_5.bin",
            } else &[_][]const u8{
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin",
            };

            // Allocate array of file metadata
            var files_buffer = try allocator.alloc(CIFARFile, file_list.len);
            errdefer allocator.free(files_buffer);

            var total_records: usize = 0;

            for (file_list, 0..) |filepath, i| {
                const count = try computeRecordCount(filepath);
                // Save metadata
                files_buffer[i] = CIFARFile{
                    .path = filepath,
                    .record_count = count,
                };
                total_records += count;
            }

            return .{
                .allocator = allocator,
                .train = train,
                .files = files_buffer,
                .total_records = total_records,
            };
        }

        /// Frees the array of file structs.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.files);
        }

        /// Return the total number of samples across all files.
        pub fn len(self: *Self) usize {
            return self.total_records;
        }

        /// Writes the `batch_i`-th sample from the dataset to GPU buffers.
        ///  - `i` is the position in your GPU batch dimension
        ///  - `batch_i` is which sample you want from the entire dataset
        ///  - `batch` is [data_tensor, label_tensor]
        ///  - `context` is your GPU context
        pub fn write(
            self: *Self,
            i: usize,
            batch_i: usize,
            batch: [2]*GPUTensor(T),
            context: *Context,
        ) !void {
            const data_tensor, const label_tensor = batch;

            // Figure out which file holds the `batch_i`-th record
            const file_index, const record_index_in_file = self.findFileIndex(batch_i);
            const file_metadata = self.files[file_index];

            // We'll open that file, read exactly one record (3073 bytes),
            // do the conversions, then write it onto the GPU.
            var file_handle = try std.fs.cwd().openFile(file_metadata.path, .{});
            defer file_handle.close();

            // Seek to the correct offset for this record
            const offset_in_bytes = record_index_in_file * record_size;
            try file_handle.seekTo(offset_in_bytes);

            // We'll read the single record into a small local buffer
            // (stack or a small allocator).
            var record_buffer: [record_size]u8 = undefined;
            const read_bytes = try file_handle.readAll(record_buffer[0..]);
            if (read_bytes != record_size) {
                return error.InvalidFile;
            }

            // The first byte is the label
            const label_u8 = record_buffer[0];

            // The next 3072 are the image data
            const image_slice = record_buffer[1..];

            // Convert the label to a one-hot vector
            var host_label: [num_classes]T = .{0.0} ** num_classes;
            host_label[label_u8] = 1.0;

            // ----------------------------------------------------------
            // STEP 1: Convert the 32×32×3 image to a 2D/3D representation
            // (Here it's just a slice, so let's treat it as [H×W×C].)
            // We'll do nearest-neighbor downsampling to 16×16×3.
            // ----------------------------------------------------------

            // Destination buffer for the smaller image (16×16×3 = 768).
            const host_small_image = try self.allocator.alloc(T, out_image_size);
            defer self.allocator.free(host_small_image);

            // We'll iterate over 16×16 and pick source pixels from 32×32.
            // nearest-neighbor scaling factors:
            const scale_h = @as(T, @floatCast(input_height)) / @as(T, @floatCast(out_height)); // 32/16 = 2.0
            const scale_w = @as(T, @floatCast(input_width)) / @as(T, @floatCast(out_width)); // 32/16 = 2.0

            for (0..out_height) |row_out| {
                for (0..out_width) |col_out| {
                    // nearest neighbor source coords:
                    const row_in: usize = @intFromFloat(@as(T, @floatFromInt(row_out)) * scale_h);
                    const col_in: usize = @intFromFloat(@as(T, @floatFromInt(col_out)) * scale_w);

                    // For each color channel:
                    for (0..out_channels) |c| {
                        const src_index = row_in * (input_width * input_channels) + col_in * input_channels + c;
                        const dst_index = row_out * (out_width * out_channels) + col_out * out_channels + c;
                        // Convert from u8 to T
                        host_small_image[dst_index] = @floatFromInt(image_slice[src_index]);
                    }
                }
            }

            // ----------------------------------------------------------
            // STEP 2: Write the smaller image (16×16×3) to data_tensor
            // offset by i * out_image_size.
            // Make sure data_tensor is sized for [batch, 768].
            // ----------------------------------------------------------
            try data_tensor.writeFromHostAsync(
                host_small_image,
                i * out_image_size,
                context.stream,
            );

            // ----------------------------------------------------------
            // STEP 3: Write the label (one-hot vector) to label_tensor
            // offset by i * num_classes
            // ----------------------------------------------------------
            try label_tensor.writeFromHostAsync(
                &host_label,
                i * num_classes,
                context.stream,
            );
        }

        /// Given an absolute record index `record_i` in the entire dataset,
        /// figure out which file it belongs to and the 0-based index within that file.
        fn findFileIndex(self: *Self, record_i: usize) [2]usize {
            var remaining = record_i;
            for (self.files, 0..) |file_meta, f_idx| {
                if (remaining < file_meta.record_count) {
                    return .{ f_idx, remaining };
                } else {
                    remaining -= file_meta.record_count;
                }
            }
            // If out of bounds, clamp to the last record.
            const last = self.files.len - 1;
            return .{ last, self.files[last].record_count - 1 };
        }

        /// Opens a file, computes how many 3073-byte records are in it.
        fn computeRecordCount(filepath: []const u8) !usize {
            var file_handle = try std.fs.cwd().openFile(filepath, .{});
            defer file_handle.close();

            const info = try file_handle.stat();
            if (info.size % record_size != 0) {
                return error.InvalidFile;
            }
            return @intCast(info.size / record_size);
        }
    };
}
