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

        const Self = @This();
        pub const Elemtype = T;

        pub fn create(allocator: std.mem.Allocator, shape: [rank]usize, stream: *const Stream) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);
            self.* = .{
                .data = try GPUTensor(T, rank).initAsync(shape, stream),
                .grad = null,
                .creator = null,
            };
            return self;
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator, stream: *const Stream) void {
            self.data.deinitAsync(stream);

            if (self.grad) |grad| {
                grad.destroy(allocator, stream);
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

        // pub fn backward(
        //     self: *Self,
        //     allocator: std.mem.Allocator,
        //     cuda_context: *const CudaContext,
        //     stream: *const Stream,
        // ) !void {
        //     if (self.creator) |creator| {
        //         self.grad = self.grad orelse try blk: {
        //             var ones = try Self.create(allocator, self.data.base.shape, stream);
        //             errdefer ones.destroy(allocator, stream);
        //             try ones.data.fill(if (T == BF16) BF16.fromF32(1.0) else 1.0, stream);
        //             break :blk ones;
        //         };

        //         _ = try creator.backwardErased(
        //             allocator,
        //             self.grad.?,
        //             cuda_context,
        //             stream,
        //         );
        //     }
        // }

        pub fn backward(
            self: *Self,
            allocator: std.mem.Allocator,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            var funcs = std.ArrayList(*Function).init(allocator);
            defer funcs.deinit();

            if (self.creator) |creator| {
                try funcs.append(creator);
            }

            while (funcs.popOrNull()) |func| {
                const gy = func.getOutputGradErased() orelse blk: {
                    try func.setOutputGradOne(allocator, stream);
                    break :blk func.getOutputGradErased();
                };
                _ = try func.backwardErased(allocator, gy.?, cuda_context, stream);
                const x_creator = func.getInputCreator();

                if (x_creator) |c| {
                    try funcs.append(c);
                }
            }
        }
    };
}
