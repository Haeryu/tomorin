pub fn sliceCast(comptime T: type, slice: anytype) []T {
    const casted_slice: []T = @as([*]T, @ptrCast(@alignCast(slice.ptr)))[0..slice.len];
    return casted_slice;
}
