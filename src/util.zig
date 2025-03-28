const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const Context = @import("context.zig").Context;
const TaggedVar = @import("variable.zig").TaggedVar;

pub fn sliceCast(comptime T: type, slice: anytype) []T {
    const childsize = @sizeOf(@typeInfo(@TypeOf(slice)).pointer.child);
    std.debug.assert((slice.len * childsize) % @sizeOf(T) == 0);
    const casted_slice: []T = @as([*]T, @ptrCast(@alignCast(slice.ptr)))[0 .. (slice.len * childsize) / @sizeOf(T)];
    return casted_slice;
}

pub fn constSliceCast(comptime T: type, slice: anytype) []const T {
    const childsize = @sizeOf(@typeInfo(@TypeOf(slice)).pointer.child);
    std.debug.assert((slice.len * childsize) % @sizeOf(T) == 0);
    const casted_slice: []const T = @as([*]const T, @ptrCast(@alignCast(slice.ptr)))[0 .. (slice.len * childsize) / @sizeOf(T)];
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

pub fn accuracy(comptime T: type, y: *TaggedVar, t: *TaggedVar, axis: usize) !T {
    const context = y.getContext();
    var pred = try y.asUntagged(T).data.argmax(context.allocator, &.{@intCast(axis)}, true, context.stream);
    defer pred.deinitAsync(context.stream);

    try pred.reshape(t.getShape());

    try pred.equal(&t.asUntagged(usize).data, context.stream);

    var acc = try pred.meanInt(T, context.allocator, null, true, context.stream);
    defer acc.deinitAsync(context.stream);

    var acc_host = try acc.toHost(context.allocator, context.stream);
    defer acc_host.deinit(context.allocator);

    return acc_host.data[0];
}

pub fn getDeconvOutsize(size: usize, k: usize, s: usize, p: usize) usize {
    return s * (size - 1) + k - 2 * p;
}

// pub fn accuracy(comptime T: type, y: *TaggedVar, t: *TaggedVar, num_classes: usize) !T {
//     const context = y.getContext();
//     var pred = try y.asUntagged(T).data.argmax(context.allocator, &.{1}, false, context.stream);
//     defer pred.deinitAsync(context.stream);

//     var pred_one_hot = try pred.toOneHot(T, num_classes, context.stream);
//     defer pred_one_hot.deinitAsync(context.stream);

//     var pred_one_hot_reshaped = try pred_one_hot.reshape(t.getShape(), context.stream);
//     defer pred_one_hot_reshaped.deinitAsync(context.stream);

//     try pred_one_hot_reshaped.equalApprox(&t.asUntagged(T).data, 1e-4, context.stream);

//     var acc = try pred_one_hot_reshaped.mean(context.allocator, null, false, context.stream);
//     defer acc.deinitAsync(context.stream);

//     var acc_host = try acc.toHost(context.allocator, context.stream);
//     defer acc_host.deinit(context.allocator);

//     // var result = pred_reshaped
//     return acc_host.at(&.{0}).*;
// }

pub fn downloadFile(allocator: std.mem.Allocator, url: []const u8, file_path: []const u8) !void {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    var server_header_buf: [4096]u8 = undefined;

    const uri = try std.Uri.parse(url);
    var req = try client.open(.GET, uri, .{
        .server_header_buffer = &server_header_buf,
    });
    defer req.deinit();

    try req.send();
    try req.wait();

    if (req.response.status != .ok) {
        return error.HttpError;
    }

    const content_length = blk: {
        var iter = req.response.iterateHeaders();
        while (iter.next()) |header| {
            if (std.mem.eql(u8, header.name, "Content-Length")) {
                break :blk header.value;
            }
        } else {
            break :blk null;
        }
    };
    const total_size = if (content_length) |cl| try std.fmt.parseInt(u64, cl, 10) else null;

    var file = try std.fs.cwd().createFile(file_path, .{});
    defer file.close();

    var downloaded: u64 = 0;
    var buf: [4096]u8 = undefined;

    while (true) {
        const read = try req.read(&buf);
        if (read == 0) break;
        try file.writeAll(buf[0..read]);
        downloaded += read;

        if (total_size) |ts| {
            const percent = @as(f32, @floatFromInt(downloaded)) / @as(f32, @floatFromInt(ts)) * 100.0;
            std.debug.print("\rProgress: {d:.2}%", .{percent});
        } else {
            std.debug.print("\rDownloaded: {d} bytes", .{downloaded});
        }
    }
    std.debug.print("\nDone\n", .{});
}

pub fn pow(comptime T: type, x: T, y: T) T {
    if (T == f32 or T == f64) {
        return std.math.pow(T, x, y);
    } else if (T == f16) {
        return @floatCast(std.math.pow(f32, @floatCast(x), @floatCast(y)));
    }
}
