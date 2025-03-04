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
const sin = tomorin.function.sin;
const TaggedVar = tomorin.variable.TaggedVar;

fn factorial(x: f64) f64 {
    if (x == 0) {
        return 1.0;
    }
    return x * factorial(x - 1.0);
}

fn taylorSin(comptime T: type, x: *TaggedVar, threshold: f32) !*TaggedVar {
    var y_ten = try tomo.tensor.GPUTensor(T).initAsync(x.asUntaggedConst(T).data.base.getShapeConst(), x.getContextConst().stream);
    errdefer y_ten.deinitAsync(x.getContextConst().stream);
    try y_ten.fill(0.0, x.getContextConst().stream);

    var y = try x.getContext().createVariable(T, y_ten.move(), null);

    var cpu_t = try tomo.tensor.CPUTensor(T).init(x.getContextConst().allocator, x.asUntaggedConst(T).data.base.getShapeConst());
    defer cpu_t.deinit(x.getContextConst().allocator);

    for (0..100000) |i| {
        const sign = try std.math.powi(i32, -1, @intCast(i));
        const c = @as(T, @floatFromInt(sign)) / factorial(@as(T, @floatFromInt(2 * i + 1)));
        var t = try scale(T, try pow(T, x, 2 * @as(i32, @intCast(i)) + 1), c);

        try cpu_t.writeFromDevice(t.asUntagged(T).data.ptr.?, t.asUntagged(T).data.calcLen(), 0, x.getContextConst().stream);
        // y.release();
        y = try add(T, y, t);

        try x.getContextConst().stream.sync();
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

    var context = try tomorin.context.Context.init(allocator, &cuda_context, &stream, .{
        .aggressive_release = false,
        .init_func_capacity = 0,
        .init_var_capacity = 0,
        .verbose_dot = true,
        .front_only = false,
    });
    defer context.deinit();

    const F = f64;

    var v1 = try tomo.tensor.GPUTensor(F).initAsync(&.{1}, &stream);
    errdefer v1.deinitAsync(&stream);
    try v1.fill(std.math.pi / 4.0, &stream);

    var x = try context.createVariable(F, v1.move(), "x");

    var y = try taylorSin(
        F,
        x,
        1e-40,
    );

    try y.saveDot("graph/graph.dot");

    try y.backward();

    x.refGrad().?.setName("x_grad");
    y.refGrad().?.setName("y_grad");

    try x.refGrad().?.saveDot("graph/graph_grad.dot");

    const gx = x.refGrad().?;
    x.setGrad(null);
    try gx.backward();
    try x.refGrad().?.saveDot("graph/graph_grad_grad.dot");

    try stream.sync();

    var host_y = try y.asUntaggedConst(F).data.toHost(allocator, &stream);
    defer host_y.deinit(allocator);

    var host_gx = try gx.asUntaggedConst(F).data.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    var host_ggx = try x.refGrad().?.asUntaggedConst(F).data.toHost(allocator, &stream);
    defer host_ggx.deinit(allocator);

    try stream.sync();

    std.debug.print("{d}", .{host_y});
    std.debug.print("{d}", .{host_gx});
    std.debug.print("{d}", .{host_ggx});
    std.debug.print("{}\n", .{context.countVariable()});
    std.debug.print("{}\n", .{context.countFunction()});
    // std.debug.print("{}\n", .{y.calcLen()});
    // std.debug.print("{}\n", .{x.refGradConst().?.calcLen()});
    // std.debug.print("{}\n", .{y.refGrad().?.refGrad().?.calcLen()});
}
