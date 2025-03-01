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

    var context = try tomorin.context.Context.init(allocator, &cuda_context, &stream, false, true);
    defer context.deinit();

    var v1 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v1.deinitAsync(&stream);
    try v1.fill(2.0, &stream);

    var v2 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v2.deinitAsync(&stream);

    try v2.fill(3.0, &stream);

    const x1 = try context.createVariable(f32, v1.move(), "x1");
    defer x1.release();

    const x2 = try context.createVariable(f32, v2.move(), "x2");
    defer x2.release();

    const z = try function.matyas(f32, try function.matyas(f32, try function.matyas(f32, x1, x2), try function.matyas(f32, x1, x2)), try function.matyas(f32, x1, x2));

    //const z = try add(f32, x1, try div(f32, x1, x2));

    try context.backward(f32, z, &.{ x1, x2 });

    try context.saveDot(0, "graph/graph.dot");
    try context.saveDot(1, "graph/graph_grad.dot");

    try stream.sync();

    var host_z = try z.dataToHost(f32);
    defer host_z.deinit(allocator);

    var gx1 = try x1.gradToHost(f32);
    defer gx1.deinit(allocator);

    var gx2 = try x2.gradToHost(f32);
    defer gx2.deinit(allocator);

    try stream.sync();

    std.debug.print("{d}", .{host_z});
    std.debug.print("{d}", .{gx1});
    std.debug.print("{d}", .{gx2});
    std.debug.print("{}\n", .{context.countAliveVariableAtLevel(0)});
    std.debug.print("{}\n", .{context.countAliveVariableAtLevel(1)});
}
