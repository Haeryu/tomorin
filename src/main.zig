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
const sum = tomorin.function.sum;
const broadcastTo = tomorin.function.broadcastTo;
const matmul = tomorin.function.matmul;
const meanSquaredError = tomorin.function.meanSquaredError;
const sigmoid = tomorin.function.sigmoid;
const linear = tomorin.function.linear;
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
    const b = try context.createVariable(F, bv.move(), "b");

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

        try stream.sync();

        std.debug.print("w: {d}", .{w_host});
        std.debug.print("b: {d}\n\n", .{b_host});
    }
}

fn predict2(
    comptime T: type,
    x: *TaggedVar,
    w1: *TaggedVar,
    b1: *TaggedVar,
    w2: *TaggedVar,
    b2: *TaggedVar,
) !*TaggedVar {
    const y1 = try linear(T, x, w1, b1);
    const y1_sig = try sigmoid(T, y1);
    const y2 = try linear(T, y1_sig, w2, b2);

    return y2;
}

fn example3() !void {
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
    defer xv.deinitAsync(&stream);
    try xv.fillUniformRange(-std.math.pi, std.math.pi, &cuda_context, &stream);

    var noisev = try tomo.tensor.GPUTensor(F).initAsync(&.{ 100, 1 }, &stream);
    defer noisev.deinitAsync(&stream);
    try noisev.fillUniformRange(-0.1, 0.1, &cuda_context, &stream);

    var yv = try xv.cloneAsync(&stream);
    defer yv.deinitAsync(&stream);
    try yv.sin(&stream);

    var y_scale = try yv.scale(2.0 * std.math.pi, &cuda_context, &stream);
    defer y_scale.deinitAsync(&stream);

    var yv_noise = try y_scale.add(&noisev, &cuda_context, &stream);
    defer yv_noise.deinitAsync(&stream);

    const I = 1;
    const H = 20;
    const O = 1;

    var w1v = try tomo.tensor.GPUTensor(F).initAsync(&.{ I, H }, &stream);
    errdefer w1v.deinitAsync(&stream);
    try w1v.fillUniformRange(-0.001, 0.001, &cuda_context, &stream);

    var b1v = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, H }, &stream);
    errdefer b1v.deinitAsync(&stream);
    try b1v.fill(0.0, &stream);

    var w2v = try tomo.tensor.GPUTensor(F).initAsync(&.{ H, O }, &stream);
    errdefer w2v.deinitAsync(&stream);
    try w2v.fillUniformRange(-0.1, 0.1, &cuda_context, &stream);

    var b2v = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, O }, &stream);
    errdefer b2v.deinitAsync(&stream);
    try b2v.fill(0.0, &stream);

    const x = try context.createVariable(F, xv.move(), "x");
    const y = try context.createVariable(F, yv_noise.move(), "y");
    const w1 = try context.createVariable(F, w1v.move(), "w1");
    const b1 = try context.createVariable(F, b1v.move(), "b1");

    const w2 = try context.createVariable(F, w2v.move(), "w2");
    const b2 = try context.createVariable(F, b2v.move(), "b2");

    // const gw1 = w1.detatchGrad();
    // const gb1 = b1.detatchGrad();
    // const gw2 = w2.detatchGrad();
    // const gb2 = b2.detatchGrad();

    var gw1v = try tomo.tensor.GPUTensor(F).initAsync(&.{ I, H }, &stream);
    errdefer gw1v.deinitAsync(&stream);

    var gb1v = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, H }, &stream);
    errdefer gb1v.deinitAsync(&stream);

    var gw2v = try tomo.tensor.GPUTensor(F).initAsync(&.{ H, O }, &stream);
    errdefer gw2v.deinitAsync(&stream);

    var gb2v = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, O }, &stream);
    errdefer gb2v.deinitAsync(&stream);

    const lr = 0.0001;
    const iters = 1;

    // TODO : protect -> not bool, out from context. protext from context

    const base_state = context.popState();
    defer context.restoreState(base_state);

    for (0..iters) |i| {
        defer {
            stream.sync() catch {};
            // context.destroyFunctions();
            // context.destroyVariables();
            // snapshot.destroyFunctions();
        }

        const y_pred = try predict2(F, x, w1, b1, w2, b2);
        const loss = try meanSquaredError(F, y, y_pred);

        try loss.backward();
        // TODO move grad out of for loop -> prevent destroy
        const gw1 = w1.detatchGrad();
        const gb1 = b1.detatchGrad();
        const gw2 = w2.detatchGrad();
        const gb2 = b2.detatchGrad();

        var dgw1 = try gw1.asUntagged(F).data.scale(lr, &cuda_context, &stream);
        defer dgw1.deinitAsync(&stream);

        var dgw2 = try gw2.asUntagged(F).data.scale(lr, &cuda_context, &stream);
        defer dgw2.deinitAsync(&stream);

        var dgb1 = try gb1.asUntagged(F).data.scale(lr, &cuda_context, &stream);
        defer dgb1.deinitAsync(&stream);

        var dgb2 = try gb2.asUntagged(F).data.scale(lr, &cuda_context, &stream);
        defer dgb2.deinitAsync(&stream);

        var w1_new = try w1.asUntagged(F).data.sub(
            &dgw1,
            &cuda_context,
            &stream,
        );
        defer w1_new.deinitAsync(&stream);

        var w2_new = try w2.asUntagged(F).data.sub(
            &dgw2,
            &cuda_context,
            &stream,
        );
        defer w2_new.deinitAsync(&stream);

        var b1_new = try b1.asUntagged(F).data.sub(
            &dgb1,
            &cuda_context,
            &stream,
        );
        defer b1_new.deinitAsync(&stream);

        var b2_new = try b2.asUntagged(F).data.sub(
            &dgb2,
            &cuda_context,
            &stream,
        );
        defer b2_new.deinitAsync(&stream);

        w1.asUntagged(F).data.deinitAsync(&stream);
        w1.asUntagged(F).data = w1_new.move();
        w2.asUntagged(F).data.deinitAsync(&stream);
        w2.asUntagged(F).data = w2_new.move();
        b1.asUntagged(F).data.deinitAsync(&stream);
        b1.asUntagged(F).data = b1_new.move();
        b2.asUntagged(F).data.deinitAsync(&stream);
        b2.asUntagged(F).data = b2_new.move();

        if (i % 1000 == 0) {
            var pi_4v = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, 1 }, &stream);
            errdefer pi_4v.deinitAsync(&stream);
            try pi_4v.fill(std.math.pi, &stream);

            const pi_4 = try context.createVariable(F, pi_4v.move(), "pi/4");

            const pi_4_y = try predict2(F, pi_4, w1, b1, w2, b2);
            try stream.sync();

            var loss_host = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer loss_host.deinit(allocator);

            var pi_4_y_host = try pi_4_y.asUntagged(F).data.toHost(allocator, &stream);
            defer pi_4_y_host.deinit(allocator);

            std.debug.print("loss: {d}", .{loss_host});
            std.debug.print("pi_4: {d}\n", .{pi_4_y_host});
        }
    }
}

pub fn main() !void {
    try example3();
}
