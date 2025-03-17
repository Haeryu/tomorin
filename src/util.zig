const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const Context = @import("context.zig").Context;

pub fn sliceCast(comptime T: type, slice: anytype) []T {
    const childsize = @sizeOf(@typeInfo(@TypeOf(slice)).pointer.child);
    comptime std.debug.assert(@divExact(slice.len * childsize, @sizeOf(T)));
    const casted_slice: []T = @as([*]T, @ptrCast(@alignCast(slice.ptr)))[0 .. (slice.len * childsize) / @sizeOf(T)];
    return casted_slice;
}

pub fn debugPrintGpuTensor(comptime T: type, gpu_tensor: *const GPUTensor(T), context: *const Context) !void {
    var host = try gpu_tensor.toHost(context.allocator, context.stream);
    defer host.deinit(context.allocator);

    std.debug.print("{d}\n", .{host});
}

pub fn arangeAlloc(allocator: std.mem.Allocator, comptime T: type, start: T, stop: T, step: T) ![]T {
    if (step == 0) return error.InvalidStep;

    const len = (stop - start) / step;

    const buf = try allocator.alloc(T, len);
    errdefer allocator.free(buf);

    for (buf, 0..) |*e, i| {
        const val = start + step * i;
        if (val >= stop) break;

        e.* = val;
    }

    return buf;
}
