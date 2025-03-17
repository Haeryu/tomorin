const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const Context = @import("context.zig").Context;

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
