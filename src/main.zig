const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");
const function = tomorin.function;
const VarKey = tomorin.context.VarKey;
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
const sin = tomorin.function.sin;

fn factorial(x: f64) f64 {
    if (x == 0) {
        return 1.0;
    }
    return x * factorial(x - 1.0);
}

fn taylorSin(comptime T: type, x: VarKey, threshold: f32) !VarKey {
    var y_ten = try tomo.tensor.GPUTensor(T).initAsync(x.refConst().asUntaggedConst(T).data.base.getShapeConst(), x.context.stream);
    errdefer y_ten.deinitAsync(x.context.stream);
    try y_ten.fill(0.0, x.context.stream);

    var y = try x.context.createVariable(T, y_ten.move(), null);

    var cpu_t = try tomo.tensor.CPUTensor(T).init(x.context.allocator, x.refConst().asUntaggedConst(T).data.base.getShapeConst());
    defer cpu_t.deinit(x.context.allocator);

    for (0..100000) |i| {
        const sign = try std.math.powi(i32, -1, @intCast(i));
        const c = @as(T, @floatFromInt(sign)) / factorial(@as(T, @floatFromInt(2 * i + 1)));
        var t = try scale(T, try pow(T, x, 2 * @as(i32, @intCast(i)) + 1), c);

        try cpu_t.writeFromDevice(t.ref().asUntagged(T).data.ptr.?, t.ref().asUntagged(T).data.calcLen(), 0, x.context.stream);
        // y.release();
        y = try add(T, y, t);

        try x.context.stream.sync();
        if (@abs(cpu_t.at(&.{0}).*) < threshold) {
            break;
        }
    }

    return y;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream = try tomo.stream.Stream.create();
    defer stream.destroy();

    var cuda_context = try tomo.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    var context = try tomorin.context.Context.init(allocator, &cuda_context, &stream, true, true, false);
    defer context.deinit();

    const F = f64;

    var v1 = try tomo.tensor.GPUTensor(F).initAsync(&.{1}, &stream);
    errdefer v1.deinitAsync(&stream);
    try v1.fill(std.math.pi / 4.0, &stream);

    // var v2 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    // errdefer v2.deinitAsync(&stream);

    // try v2.fill(3.0, &stream);

    var x1 = try context.createVariable(F, v1.move(), "x");
    defer x1.release();

    // const x2 = try context.createVariable(f32, v2.move(), "x2");
    // defer x2.release();

    //const z = try function.matyas(f32, try function.matyas(f32, try function.matyas(f32, x1, x2), try function.matyas(f32, x1, x2)), try function.matyas(f32, x1, x2));

    //const z = try add(f32, x1, try div(f32, x1, x2));

    var z = try taylorSin(F, x1, 1e-40);
    z.ref().setName("z");
    defer z.release();

    //try context.backward(f32, z, &.{ x1, x2 });
    std.debug.print("back\n", .{});
    try context.backward(F, z, &.{x1});
    x1.ref().refGrad().?.setName("x_grad");
    z.ref().refGrad().?.setName("z_grad");

    // try context.saveDot(0, "graph/graph.dot");
    // try context.saveDot(1, "graph/graph_grad.dot");

    try stream.sync();

    var host_z = try z.dataToHost(F);
    defer host_z.deinit(allocator);

    var gx1 = try x1.gradToHost(F);
    defer gx1.deinit(allocator);

    // var gx2 = try x2.gradToHost(f32);
    // defer gx2.deinit(allocator);

    try stream.sync();

    std.debug.print("{d}", .{host_z});
    std.debug.print("{d}", .{gx1});
    //std.debug.print("{d}", .{gx2});
    std.debug.print("{}\n", .{context.countAliveVariableAtLevel(0)});
    std.debug.print("{}\n", .{context.countAliveVariableAtLevel(1)});
}
