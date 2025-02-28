const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");

// TODO: graphviz first, bugfix next

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

    const x1 = try context.createVariable(f32, v1, "x1");
    const px1 = context.getVariable(x1).asUntagged(f32);

    // const y = try tomorin.function.neg(f32, try tomorin.function.neg(f32, try tomorin.function.neg(f32, x1, &context), &context), &context);
    //const y = try tomorin.function.neg(f32, x1, &context);
    const y = try tomorin.function.sub(f32, x1, x1, &context);

    try stream.sync();
    try context.backward(f32, y);

    try stream.sync();

    var v_host = try context.getVariable(y).asUntagged(f32).data.toHost(allocator, &stream);
    defer v_host.deinit(allocator);

    var gx = context.getVariable(px1.grad.?).asUntagged(f32);
    var gx_host = try gx.data.toHost(allocator, &stream);
    defer gx_host.deinit(allocator);

    std.debug.print("{d}", .{v_host});
    std.debug.print("{d}", .{gx_host});
}
