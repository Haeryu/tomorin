const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");
const add = tomorin.function.add;
const sub = tomorin.function.sub;
const neg = tomorin.function.neg;
const mul = tomorin.function.mul;
const div = tomorin.function.div;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream = try tomo.stream.Stream.create();
    defer stream.destroy();

    var cuda_context = try tomo.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    var context = try tomorin.context.Context.init(allocator, &cuda_context, &stream, true);
    defer context.deinit();

    var v1 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v1.deinitAsync(&stream);
    try v1.fill(2.0, &stream);

    var v2 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v2.deinitAsync(&stream);
    try v2.fill(4.0, &stream);

    var v3 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v3.deinitAsync(&stream);
    try v3.fill(7.0, &stream);

    const x1 = try context.createVariable(f32, v1, "x1");

    const x2 = try context.createVariable(f32, v2, "x2");

    const x3 = try context.createVariable(f32, v3, "x3");

    // const y = try tomorin.function.neg(f32, try tomorin.function.neg(f32, try tomorin.function.neg(f32, x1, &context), &context), &context);
    //const y = try tomorin.function.neg(f32, x1, &context);
    const y = try div(f32, try add(f32, try div(f32, x1, try neg(f32, x1, &context), &context), x2, &context), x3, &context);

    try context.backward(f32, y);

    try stream.sync();

    var v_host = try context.getVariable(y).asUntagged(f32).data.toHost(allocator, &stream);
    defer v_host.deinit(allocator);

    const px1 = context.getVariable(x1).asUntagged(f32);
    var gx1 = context.getVariable(px1.grad.?).asUntagged(f32);
    var gx1_host = try gx1.data.toHost(allocator, &stream);
    defer gx1_host.deinit(allocator);

    const px2 = context.getVariable(x2).asUntagged(f32);
    var gx2 = context.getVariable(px2.grad.?).asUntagged(f32);
    var gx2_host = try gx2.data.toHost(allocator, &stream);
    defer gx2_host.deinit(allocator);

    const px3 = context.getVariable(x3).asUntagged(f32);
    var gx3 = context.getVariable(px3.grad.?).asUntagged(f32);
    var gx3_host = try gx3.data.toHost(allocator, &stream);
    defer gx3_host.deinit(allocator);

    std.debug.print("{d}", .{v_host});
    std.debug.print("{d}", .{gx1_host});
    std.debug.print("{d}", .{gx2_host});
    std.debug.print("{d}", .{gx3_host});

    std.debug.print("{any}\n", .{x1});
    std.debug.print("{any}\n", .{px1.grad.?});
    std.debug.print("{any}\n", .{px2.grad.?});
    std.debug.print("{any}\n", .{px3.grad.?});
}
