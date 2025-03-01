const std = @import("std");

pub fn sliceCast(comptime T: type, slice: anytype) []T {
    const childsize = @sizeOf(@typeInfo(@TypeOf(slice)).pointer.child);
    comptime std.debug.assert(@divExact(slice.len * childsize, @sizeOf(T)));
    const casted_slice: []T = @as([*]T, @ptrCast(@alignCast(slice.ptr)))[0 .. (slice.len * childsize) / @sizeOf(T)];
    return casted_slice;
}
