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
const tanh = tomorin.function.tanh;
const reshape = tomorin.function.reshape;
const transpose = tomorin.function.transpose;
const sumTo = tomorin.function.sumTo;
const broadcastTo = tomorin.function.broadcastTo;
const matmul = tomorin.function.matmul;
const meanSquaredError = tomorin.function.meanSquaredError;
const TaggedVar = tomorin.variable.TaggedVar;

const F = f32;

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

fn example() !void {
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
    });
    defer context.deinit();

    var v1 = try tomo.tensor.GPUTensor(F).initAsync(&.{1}, &stream);
    errdefer v1.deinitAsync(&stream);
    try v1.fill(std.math.pi / 4.0, &stream);

    var x = try context.createVariable(F, v1.move(), "x");

    //var y = try tanh(F, x);
    var y = try taylorSin(F, x, 1e-4);

    x.setName("x");
    y.setName("y");

    try y.backward();

    x.refGrad().?.setName("gx");
    try x.refGrad().?.saveDot("./graph/grad1.dot");

    const iters = 8;

    inline for (0..iters) |i| {
        var gx = x.detatchGrad();
        try gx.backward();

        const fname = std.fmt.comptimePrint("./graph/grad{}.dot", .{i + 2});

        const gxname = std.fmt.comptimePrint("gx{}", .{i + 2});
        x.refGrad().?.setName(gxname);
        try x.refGrad().?.saveDot(fname);
    }
}

fn predict(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar) !*TaggedVar {
    return try add(T, try matmul(T, x, w), try broadcastTo(T, b, &.{ 100, 1 }));
}

fn example2() !void {
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
    });
    defer context.deinit();

    var xv = try tomo.tensor.GPUTensor(F).initAsync(&.{ 100, 1 }, &stream);
    errdefer xv.deinitAsync(&stream);
    try xv.fillUniform(&cuda_context, &stream);

    var noisev = try tomo.tensor.GPUTensor(F).initAsync(&.{ 100, 1 }, &stream);
    errdefer noisev.deinitAsync(&stream);
    try noisev.fillUniform(&cuda_context, &stream);

    var wv = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, 1 }, &stream);
    errdefer wv.deinitAsync(&stream);
    try wv.fill(0.0, &stream);

    var bv = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, 1 }, &stream);
    errdefer bv.deinitAsync(&stream);
    try bv.fill(0.0, &stream);

    const x = try context.createVariable(F, xv.move(), "x");
    const noise = try context.createVariable(F, noisev.move(), "noise");
    const y = try add(F, try scaleShift(F, x, 2.0, 5.0), noise);

    const w = try context.createVariable(F, wv.move(), "w");
    const b = try context.createVariable(F, bv.move(), "w");

    const lr = 0.1;
    const iters = 100;

    for (0..iters) |_| {
        const y_pred = try predict(F, x, w, b);
        const loss = try meanSquaredError(F, y, y_pred);

        try loss.backward();
        const gw = w.detatchGrad();
        const gb = b.detatchGrad();

        var gw_data = try gw.asUntagged(F).data.cloneAsync(&stream);
        defer gw_data.deinitAsync(&stream);

        var dgw = try gw_data.scale(lr, &cuda_context, &stream);
        defer dgw.deinitAsync(&stream);

        var gb_data = try gb.asUntagged(F).data.cloneAsync(&stream);
        defer gb_data.deinitAsync(&stream);

        var dgb = try gb_data.scale(lr, &cuda_context, &stream);
        defer dgb.deinitAsync(&stream);

        var w_new = try w.asUntagged(F).data.sub(
            &dgw,
            &cuda_context,
            &stream,
        );
        defer w_new.deinitAsync(&stream);

        var b_new = try b.asUntagged(F).data.sub(
            &dgb,
            &cuda_context,
            &stream,
        );
        defer b_new.deinitAsync(&stream);

        std.mem.swap(tomo.tensor.GPUTensor(F), &w.asUntagged(F).data, &w_new);
        std.mem.swap(tomo.tensor.GPUTensor(F), &b.asUntagged(F).data, &b_new);

        try stream.sync();

        var w_host = try w.asUntagged(F).data.toHost(allocator, &stream);
        defer w_host.deinit(allocator);

        var b_host = try b.asUntagged(F).data.toHost(allocator, &stream);
        defer b_host.deinit(allocator);

        std.debug.print("w: {d}", .{w_host});
        std.debug.print("b: {d}\n\n", .{b_host});
    }
}

pub fn main() !void {
    try example2();
}
