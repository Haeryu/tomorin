const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const CudaContext = tomo.cuda_context.CudaContext;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const Function = @import("function.zig").Function;

pub fn Variable(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        data: GPUTensor(T, rank),
        grad: ?*Self = null,
        creator: ?*Function = null,
        generation: usize = 0,

        const Self = @This();
        pub const Elemtype = T;

        pub fn create(allocator: std.mem.Allocator, shape: [rank]usize, stream: *const Stream) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            self.* = .{
                .data = try GPUTensor(T, rank).initAsync(shape, stream),
                .grad = null,
                .creator = null,
                .generation = 0,
            };
            return self;
        }

        pub fn fromTensor(allocator: std.mem.Allocator, data: GPUTensor(T, rank)) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            self.* = .{
                .data = data,
                .grad = null,
                .creator = null,
                .generation = 0,
            };
            return self;
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator, stream: *const Stream) void {
            self.data.deinitAsync(stream);

            if (self.grad) |grad| {
                grad.destroy(allocator, stream);
                self.grad = null;
            }

            allocator.destroy(self);
        }

        pub fn clone(self: *const Self, allocator: std.mem.Allocator, stream: *const Stream) !*Self {
            var cloned = try Self.create(allocator, self.data.base.shape, stream);
            errdefer cloned.destroy(allocator, stream);

            try cloned.data.writeAsync(self.data.ptr.?, self.data.calcLen(), 0, stream);

            if (self.grad) |grad| {
                cloned.grad = try grad.clone(allocator, stream);
            }

            return cloned;
        }

        pub fn setCreator(self: *Self, creator: *Function) void {
            self.creator = creator;
            self.generation = creator.getGeneration() + 1;
        }

        pub fn backward(
            self: *Self,
            allocator: std.mem.Allocator,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            if (self.grad == null) {
                self.grad = try Self.create(allocator, self.data.base.shape, stream);
                errdefer {
                    self.grad.?.destroy(allocator, stream);
                    self.grad = null;
                }
                try self.grad.?.data.fill(if (T == BF16) BF16.fromF32(1.0) else 1.0, stream);
            }

            var funcs = Function.Queue.init(allocator, {});
            defer funcs.deinit();

            var seen_set = std.AutoHashMap(*Function, void).init(allocator);
            defer seen_set.deinit();

            if (self.creator) |creator| {
                try funcs.add(creator);
                try seen_set.put(creator, {});
            }

            while (funcs.removeOrNull()) |func| {
                _ = try func.backwardErased(allocator, cuda_context, stream);
                try func.pushInputsCreator(&funcs, &seen_set);
            }
        }
    };
}
