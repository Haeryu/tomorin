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
            batch: std.meta.Tuple(&.{ *GPUTensor(T), *GPUTensor(T) }),
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

            // const url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/";
            // if (train) {
            //     try util.downloadFile(allocator, url_base ++ "train-images-idx3-ubyte.gz", "./datasets/mnist/train-images-idx3-ubyte.gz");
            //     try util.downloadFile(allocator, url_base ++ "train-labels-idx1-ubyte.gz", "./datasets/mnist/train-labels-idx1-ubyte.gz");
            // } else {
            //     try util.downloadFile(allocator, url_base ++ "t10k-images-idx3-ubyte.gz", "./datasets/mnist/t10k-images-idx3-ubyte.gz");
            //     try util.downloadFile(allocator, url_base ++ "t10k-labels-idx1-ubyte.gz", "./datasets/mnist/t10k-labels-idx1-ubyte.gz");
            // }

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
            // Open the file
            // var zip_file = try std.fs.cwd().openFile(filepath, .{});
            // defer zip_file.close();

            // var unzipped_file = try std.fs.cwd().createFile(filepath, .{});
            // defer unzipped_file.close();

            var unzipped_file = try std.fs.cwd().openFile(filepath, .{});
            defer unzipped_file.close();

            // Create a gzip decompressor
            // try std.compress.gzip.decompress(zip_file.reader(), unzipped_file.writer());

            // Read all decompressed data into a buffer
            const data = try unzipped_file.readToEndAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(data);

            // Create a reader for the buffer
            var fbs = std.io.fixedBufferStream(data);
            const reader = fbs.reader();

            // Read and verify magic number (2049 for labels)
            const magic = try reader.readInt(u32, .big);
            if (magic != 2049) return error.InvalidMagicNumber;

            // Read number of labels
            const num_labels = try reader.readInt(u32, .big);

            // Allocate and read the labels
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
            // Open the file
            // const zip_file = try std.fs.cwd().openFile(filepath, .{});
            // defer zip_file.close();

            // const unzipped_file = try std.fs.cwd().createFile(filepath, .{});
            // defer unzipped_file.close();

            var unzipped_file = try std.fs.cwd().openFile(filepath, .{});
            defer unzipped_file.close();

            // Create a gzip decompressor
            // try std.compress.gzip.decompress(allocator, unzipped_file.reader());

            // Read all decompressed data into a buffer
            const data = try unzipped_file.readToEndAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(data);

            // Create a reader for the buffer
            var fbs = std.io.fixedBufferStream(data);
            const reader = fbs.reader();

            // Read and verify magic number (2051 for images)
            const magic = try reader.readInt(u32, .big);
            if (magic != 2051) return error.InvalidMagicNumber;

            // Read dimensions
            const num_images = try reader.readInt(u32, .big);
            const rows = try reader.readInt(u32, .big);
            const columns = try reader.readInt(u32, .big);

            // Verify MNIST dimensions (28x28)
            if (rows != 28 or columns != 28) return error.InvalidDimensions;

            // Calculate total data size
            const data_size = num_images * rows * columns;

            // Allocate and read the image data
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
            batch: std.meta.Tuple(&.{ *GPUTensor(T), *GPUTensor(T) }),
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
