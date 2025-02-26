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

    const context: tomorin.context.Context = .{
        .cuda_context = &cuda_context,
        .stream = &stream,
        .enable_backprop_graph = true,
        .function_allocator = allocator,
        .multi_purpose_allocator = allocator,
        .variable_allocator = allocator,
    };

    var v1 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v1.deinitAsync(&stream);
    try v1.fill(2.0, &stream);

    var v2 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v2.deinitAsync(&stream);
    try v2.fill(3.0, &stream);

    var v3 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v3.deinitAsync(&stream);
    try v3.fill(4.0, &stream);

    var x1 = try tomorin.variable.Variable(f32).create(v1.move(), null, &context);
    defer x1.release(context.variable_allocator);
    var x2 = try tomorin.variable.Variable(f32).create(v2.move(), null, &context);
    defer x2.release(context.variable_allocator);
    var x3 = try tomorin.variable.Variable(f32).create(v3.move(), null, &context);
    defer x3.release(context.variable_allocator);

    var y1 = try tomorin.function.neg(
        f32,
        try tomorin.function.mul(
            f32,
            try tomorin.function.sub(
                f32,
                x1.clone(),
                x2.clone(),
                &context,
            ),
            x3.clone(),
            &context,
        ),
        &context,
    );
    defer y1.release(context.variable_allocator);

    try y1.get().?.backward();

    try context.stream.sync();

    var res = try y1.get().?.data.toHost(allocator, &stream);
    defer res.deinit(allocator);

    var gx1 = try x1.get().?.grad.?.get().?.data.toHost(allocator, &stream);
    defer gx1.deinit(allocator);

    // var gx2 = try x2.get().?.grad.?.get().?.data.toHost(allocator, &stream);
    // defer gx2.deinit(allocator);

    // var gx3 = try x1.get().?.grad.?.get().?.data.toHost(allocator, &stream);
    // defer gx3.deinit(allocator);

    var gy = try y1.get().?.grad.?.get().?.data.toHost(allocator, &stream);
    defer gy.deinit(allocator);

    std.debug.print("{d}\n", .{res});
    std.debug.print("{d}\n", .{gx1});
    // std.debug.print("{d}\n", .{gx2});
    // std.debug.print("{d}\n", .{gx3});
    std.debug.print("{d}\n", .{gy});
}
