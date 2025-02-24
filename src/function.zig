const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const Stream = tomo.stream.Stream;
const Variable = @import("variable.zig").Variable;

pub fn Function(
    comptime InT: type,
    comptime in_rank: comptime_int,
    comptime OutT: type,
    comptime out_rank: comptime_int,
) type {
    return struct {
        ptr: *anyopaque,
        vtable: *const Vtable,

        const Vtable = struct {
            forward: *const fn (
                ctx: *anyopaque,
                in: *Variable(InT, in_rank),
                stream: *const Stream,
                out: *Variable(OutT, out_rank),
            ) anyerror!void,
        };

        const Self = @This();

        pub fn forward(
            self: *Self,
            in: *Variable(InT, in_rank),
            stream: *const Stream,
            out: *Variable(OutT, out_rank),
        ) !void {
            try self.vtable.forward(self.ptr, in, stream, out);
        }
    };
}

pub fn Square(
    comptime InT: type,
    comptime in_rank: comptime_int,
    comptime OutT: type,
    comptime out_rank: comptime_int,
) type {
    return struct {
        in: *Variable(InT, in_rank),

        const Self = @This();

        pub fn function(self: *Self) Function(InT, in_rank, OutT, out_rank) {
            return .{
                .ptr = @ptrCast(self),
                .vtable = &.{
                    .forward = forward,
                },
            };
        }

        fn forward(
            ctx: *anyopaque,
            in: *Variable(InT, in_rank),
            stream: *const Stream,
            out: *Variable(OutT, out_rank),
        ) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in = in;
            try out.data.writeAsync(self.in.data.ptr.?, self.in.data.calcLen(), 0, stream);
            try out.data.square(stream);
        }
    };
}
