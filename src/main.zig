const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream = try tomo.stream.Stream.create();
    defer stream.destroy();

    var cuda_context = try tomo.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    var v1 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v1.deinitAsync(&stream);
    try v1.fill(2.0, &stream);

    var v2 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v2.deinitAsync(&stream);
    try v2.fill(3.0, &stream);

    var v3 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v3.deinitAsync(&stream);
    try v3.fill(4.0, &stream);

    var x1 = try tomorin.variable.Variable(f32).create(allocator, v1.move(), &stream);
    defer x1.release(allocator);
    var x2 = try tomorin.variable.Variable(f32).create(allocator, v2.move(), &stream);
    defer x2.release(allocator);
    // var x3 = try tomorin.variable.Variable(f32).create(allocator, v3.move(), &stream);
    // defer x3.release(allocator);

    var y1 = try tomorin.function.neg(
        f32,
        allocator,
        try tomorin.function.mul(
            f32,
            allocator,
            try tomorin.function.sub(
                f32,
                allocator,
                x1.clone(),
                x2.clone(),
                &cuda_context,
                &stream,
            ),
            x1.clone(),
            &cuda_context,
            &stream,
        ),
        &cuda_context,
        &stream,
    );
    defer y1.release(allocator);

    try y1.get().?.backward(allocator, &cuda_context, &stream);

    try stream.sync();

    var res = try y1.get().?.data.toHost(allocator, &stream);
    defer res.deinit(allocator);

    var gx1 = try x1.get().?.grad.?.get().?.data.toHost(allocator, &stream);
    defer gx1.deinit(allocator);

    var gx2 = try x2.get().?.grad.?.get().?.data.toHost(allocator, &stream);
    defer gx2.deinit(allocator);

    var gx3 = try x1.get().?.grad.?.get().?.data.toHost(allocator, &stream);
    defer gx3.deinit(allocator);

    std.debug.print("{d}\n", .{res});
    std.debug.print("{d}\n", .{gx1});
    std.debug.print("{d}\n", .{gx2});
    std.debug.print("{d}\n", .{gx3});
}
