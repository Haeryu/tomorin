const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");
const function = tomorin.function;
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

    var v2 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v2.deinitAsync(&stream);
    try v2.fill(1.0, &stream);

    try stream.sync();

    const x1 = try context.createVariable(f32, v1, "x1");
    defer context.releaseVariable(x1);

    const x2 = try context.createVariable(f32, v2, "x2");
    defer context.releaseVariable(x2);

    const z = try function.matyas(f32, x1, x2);

    try context.backward(f32, z, &.{ x1, x2 });

    try stream.sync();

    var host_y = try context.refVariable(z).asUntagged(f32).data.toHost(allocator, context.stream);
    defer host_y.deinit(allocator);

    var gx1 = try context.refVariable(x1).refGrad().?.asUntaggedConst(f32).data.toHost(allocator, context.stream);
    defer gx1.deinit(allocator);

    var gx2 = try context.refVariable(x2).refGrad().?.asUntaggedConst(f32).data.toHost(allocator, context.stream);
    defer gx2.deinit(allocator);

    std.debug.print("{d}", .{host_y});
    std.debug.print("{d}", .{gx1});
    std.debug.print("{d}", .{gx2});
    std.debug.print("{}\n", .{context.countAliveVariableAtLevel(0)});
    std.debug.print("{}\n", .{context.countAliveVariableAtLevel(1)});
}
