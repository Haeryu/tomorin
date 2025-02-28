const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");
const add = tomorin.function.add;
const sub = tomorin.function.sub;
const neg = tomorin.function.neg;
const mul = tomorin.function.mul;
const div = tomorin.function.div;
const scale = tomorin.function.scale;
const pow = tomorin.function.pow;
const shift = tomorin.function.shift;
const scaleShift = tomorin.function.scaleShift;
const square = tomorin.function.square;

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
    try v1.fill(1.0, &stream);

    const x1 = try context.createVariable(f32, v1, "x1");
    defer context.releaseVariable(x1);

    const y = try shift(f32, x1, 3.0);
    const y_sq = try square(f32, y);

    try context.backward(f32, y_sq, &.{x1});

    try stream.sync();

    var host_y_sq = try context.refVariable(y_sq).asUntagged(f32).data.toHost(allocator, context.stream);
    defer host_y_sq.deinit(allocator);

    var gx = try context.refVariable(x1).refGrad().?.asUntaggedConst(f32).data.toHost(allocator, context.stream);
    defer gx.deinit(allocator);

    std.debug.print("{d}", .{host_y_sq});
    std.debug.print("{d}", .{gx});
}
