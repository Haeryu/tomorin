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

    std.debug.print("{d}", .{host});
}
