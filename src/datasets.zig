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

        // Output dimensions (no shrinking)
        pub const out_height = input_height; // 32
        pub const out_width = input_width; // 32
        pub const out_channels = input_channels; // 3
        pub const out_image_size = out_height * out_width * out_channels; // 32*32*3 = 3072

        pub const num_classes = 10; // 10 classes in CIFAR-10

        // Holds a path to a .bin file and how many 3073-byte records it contains.
        const CIFARFile = struct {
            path: []const u8,
            record_count: usize,
        };

        // Initialize the dataset in "streaming" fashion, so we do not load all data at once.
        pub fn init(allocator: std.mem.Allocator, train: bool) !Self {
            std.fs.cwd().makeDir("datasets/cifar-10/cifar-10-binary") catch |err| {
                switch (err) {
                    error.PathAlreadyExists => {},
                    else => return err,
                }
            };

            const file_list: []const []const u8 = if (train) &[_][]const u8{
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin",
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_2.bin",
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_3.bin",
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_4.bin",
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/data_batch_5.bin",
                // Add other training batches as needed
            } else &[_][]const u8{
                "datasets/cifar-10/cifar-10-binary/cifar-10-batches-bin/test_batch.bin",
            };

            var files_buffer = try allocator.alloc(CIFARFile, file_list.len);
            errdefer allocator.free(files_buffer);

            var total_records: usize = 0;

            for (file_list, 0..) |filepath, i| {
                const count = try computeRecordCount(filepath);
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

        // Frees the array of file structs.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.files);
        }

        // Return the total number of samples across all files.
        pub fn len(self: *Self) usize {
            return self.total_records;
        }

        // Writes the `batch_i`-th sample from the dataset to GPU buffers.
        pub fn write(
            self: *Self,
            i: usize,
            batch_i: usize,
            batch: std.meta.Tuple(&.{ *GPUTensor(T), *GPUTensor(usize) }),
            context: *Context,
        ) !void {
            const data_tensor, const label_tensor = batch;

            // Locate the file and record
            const file_index, const record_index_in_file = self.findFileIndex(batch_i);
            const file_metadata = self.files[file_index];

            var file_handle = try std.fs.cwd().openFile(file_metadata.path, .{});
            defer file_handle.close();

            const offset_in_bytes = record_index_in_file * record_size;
            try file_handle.seekTo(offset_in_bytes);

            var record_buffer: [record_size]u8 = undefined;
            const read_bytes = try file_handle.readAll(record_buffer[0..]);
            if (read_bytes != record_size) {
                return error.InvalidFile;
            }

            // Extract label and image
            const label: [1]usize = .{@intCast(record_buffer[0])};
            const image_slice = record_buffer[1..];

            // Convert label to one-hot vector

            // Allocate buffer for the full 32×32×3 image
            const host_image = try self.allocator.alloc(T, out_image_size);
            defer self.allocator.free(host_image);

            // Copy image data in [C, H, W] order, convert u8 to T, and normalize
            for (0..out_channels) |c| {
                for (0..out_height) |h| {
                    for (0..out_width) |w| {
                        const src_index = c * (input_height * input_width) + h * input_width + w;
                        const dst_index = c * (out_height * out_width) + h * out_width + w;
                        host_image[dst_index] = @as(T, @floatFromInt(image_slice[src_index])) / @as(T, 255.0);
                    }
                }
            }

            // Write image to GPU tensor
            try data_tensor.writeFromHostAsync(
                host_image,
                i * out_image_size,
                context.stream,
            );

            // Write label to GPU tensor
            try label_tensor.writeFromHostAsync(
                &label,
                i,
                context.stream,
            );
        }

        // Given an absolute record index, find the file and index within that file.
        fn findFileIndex(self: *Self, record_i: usize) [2]usize {
            var remaining = record_i;
            for (self.files, 0..) |file_meta, f_idx| {
                if (remaining < file_meta.record_count) {
                    return .{ f_idx, remaining };
                } else {
                    remaining -= file_meta.record_count;
                }
            }
            const last = self.files.len - 1;
            return .{ last, self.files[last].record_count - 1 };
        }

        // Compute the number of records in a file.
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
