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
const meanSquaredErrorEx = tomorin.function.meanSquaredErrorEx;
const sigmoid = tomorin.function.sigmoid;
const sigmoidEx = tomorin.function.sigmoidEx;
const reluEx = tomorin.function.reluEx;
const geluEx = tomorin.function.geluEx;
const linear = tomorin.function.linear;
const softmaxCrossEntropyEx = tomorin.function.softmaxCrossEntropyEx;
const getItemEx = tomorin.function.getItemEx;
const TaggedVar = tomorin.variable.TaggedVar;
const GPUTensor = tomo.tensor.GPUTensor;

const dbg = tomorin.util.debugPrintGpuTensor;

const F = f32;

fn factorial(x: f64) f64 {
    if (x == 0) {
        return 1.0;
    }
    return x * factorial(x - 1.0);
}

fn taylorSin(comptime T: type, x: *TaggedVar, threshold: f32) !*TaggedVar {
    var y_ten = try tomo.tensor.GPUTensor(T).initAsync(x.asUntaggedConst(T).data.base.getShapeConst(), x.getContextConst().stream);
    defer y_ten.deinitAsync(x.getContextConst().stream);
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
    defer v1.deinitAsync(&stream);
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
    defer xv.deinitAsync(&stream);
    try xv.fillUniform(&cuda_context, &stream);

    var noisev = try tomo.tensor.GPUTensor(F).initAsync(&.{ 100, 1 }, &stream);
    defer noisev.deinitAsync(&stream);
    try noisev.fillUniform(&cuda_context, &stream);

    var wv = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, 1 }, &stream);
    defer wv.deinitAsync(&stream);
    try wv.fill(0.0, &stream);

    var bv = try tomo.tensor.GPUTensor(F).initAsync(&.{ 1, 1 }, &stream);
    defer bv.deinitAsync(&stream);
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
    const y1 = try linear(T, x, w1, try broadcastTo(F, b1, &.{ x.getRow(), w1.getCol() }));
    const y1_sig = try sigmoid(T, y1);
    const y2 = try linear(T, y1_sig, w2, try broadcastTo(F, b2, &.{ y1_sig.getRow(), w2.getCol() }));

    return y2;
}

fn example3() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 100, 1 }, &stream);
    defer xv.deinitAsync(&stream);
    try xv.fillUniformRange(0.0, std.math.pi, &cuda_context, &stream);

    var noisev: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 100, 1 }, &stream);
    defer noisev.deinitAsync(&stream);
    try noisev.fillUniformRange(-0.001, 0.001, &cuda_context, &stream);

    var yv = try xv.cloneAsync(&stream);
    defer yv.deinitAsync(&stream);
    try yv.sin(&stream);

    // var y_scale = try yv.scale(2.0 * std.math.pi, &cuda_context, &stream);
    // defer y_scale.deinitAsync(&stream);

    try yv.add(&noisev, &stream);

    const I = 1;
    const H = 20;
    const O = 1;

    var w1v: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ I, H }, &stream);
    errdefer w1v.deinitAsync(&stream);
    try w1v.fillUniformRange(-0.1, 0.1, &cuda_context, &stream);

    var b1v: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 1, H }, &stream);
    errdefer b1v.deinitAsync(&stream);
    try b1v.fill(0.0, &stream);

    var w2v: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ H, O }, &stream);
    errdefer w2v.deinitAsync(&stream);
    try w2v.fillUniformRange(-0.1, 0.1, &cuda_context, &stream);

    var b2v: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 1, O }, &stream);
    errdefer b2v.deinitAsync(&stream);
    try b2v.fill(0.0, &stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const y = try base_chain.createVariable(F, yv.move(), "y");

    const w1 = try base_chain.createVariable(F, w1v.move(), "w1");
    const b1 = try base_chain.createVariable(F, b1v.move(), "b1");

    const w2 = try base_chain.createVariable(F, w2v.move(), "w2");
    const b2 = try base_chain.createVariable(F, b2v.move(), "b2");

    // const gw1 = w1.detatchGrad();
    // const gb1 = b1.detatchGrad();
    // const gw2 = w2.detatchGrad();
    // const gb2 = b2.detatchGrad();

    const lr = 0.002;
    const iters = 10000;

    const iter_chain = try context.createChain();
    context.current_chain = iter_chain;

    for (0..iters + 1) |i| {
        const y_pred = try predict2(F, x, w1, b1, w2, b2);

        const loss = try meanSquaredError(F, y, y_pred);

        try loss.backward();

        x.setGrad(null);
        y.setGrad(null);
        const gw1 = w1.detatchGrad();
        const gb1 = b1.detatchGrad();
        const gw2 = w2.detatchGrad();
        const gb2 = b2.detatchGrad();

        try gw1.asUntagged(F).data.scale(lr, &stream);
        try gw2.asUntagged(F).data.scale(lr, &stream);
        try gb1.asUntagged(F).data.scale(lr, &stream);
        try gb2.asUntagged(F).data.scale(lr, &stream);

        try w1.asUntagged(F).data.sub(&gw1.asUntagged(F).data, &stream);
        try w2.asUntagged(F).data.sub(&gw2.asUntagged(F).data, &stream);
        try b1.asUntagged(F).data.sub(&gb1.asUntagged(F).data, &stream);
        try b2.asUntagged(F).data.sub(&gb2.asUntagged(F).data, &stream);

        // try loss.saveDot(std.fmt.comptimePrint("graph{}.dot", .{i}));
        // try gw1.saveDot(std.fmt.comptimePrint("graphgw1{}.dot", .{i}));
        // try gb1.saveDot(std.fmt.comptimePrint("graphgb1{}.dot", .{i}));
        // try gw2.saveDot(std.fmt.comptimePrint("graphgw2{}.dot", .{i}));
        // try gb2.saveDot(std.fmt.comptimePrint("graphgb2{}.dot", .{i}));

        if (i % 1000 == 0) {
            var pi_4v: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 1, 1 }, &stream);
            errdefer pi_4v.deinitAsync(&stream);
            try pi_4v.fill(std.math.pi / 4.0, &stream);

            const pi_4 = try iter_chain.createVariable(F, pi_4v.move(), "pi/4");

            const pi_4_y = try predict2(F, pi_4, w1, b1, w2, b2);
            try stream.sync();

            var loss_host = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer loss_host.deinit(allocator);

            var pi_4_y_host = try pi_4_y.asUntagged(F).data.toHost(allocator, &stream);
            defer pi_4_y_host.deinit(allocator);
            try stream.sync();

            std.debug.print("loss: {d}", .{loss_host});
            std.debug.print("pi_4: {d}\n", .{pi_4_y_host});
        }

        // try stream.sync();
        iter_chain.clear();
    }
}

fn predict3(
    x: *TaggedVar,
    l1: *tomorin.layer.Linear(F),
    l2: *tomorin.layer.Linear(F),
    chain: *tomorin.chain.Chain,
) !*TaggedVar {
    const y1 = try l1.forward(x, chain);
    const y1_sig = try sigmoid(F, y1);
    const y2 = try l2.forward(y1_sig, chain);

    return y2;
}

fn example4() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 100, 1 }, &stream);
    defer xv.deinitAsync(&stream);
    try xv.fillUniformRange(0.0, std.math.pi, &cuda_context, &stream);

    var noisev: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 100, 1 }, &stream);
    defer noisev.deinitAsync(&stream);
    try noisev.fillUniformRange(-0.001, 0.001, &cuda_context, &stream);

    var yv = try xv.cloneAsync(&stream);
    defer yv.deinitAsync(&stream);
    try yv.sin(&stream);

    try yv.add(&noisev, &stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const y = try base_chain.createVariable(F, yv.move(), "y");

    var l1: tomorin.layer.Linear(F) = try .init(null, 20, false, &context, base_chain);
    defer l1.destroy();

    var l2: tomorin.layer.Linear(F) = try .init(null, 1, false, &context, base_chain);
    defer l2.destroy();

    const iter_chain = try context.createChain();
    context.current_chain = iter_chain;

    const lr = 0.1;
    const iters = 10000;
    for (0..iters + 1) |i| {
        const y_pred = try predict3(x, &l1, &l2, iter_chain);

        const loss = try meanSquaredErrorEx(F, y, y_pred, iter_chain);

        l1.clearGrads();
        l2.clearGrads();
        y.setGrad(null);
        x.setGrad(null);

        try loss.backward();

        const l1_ps = l1.getParams();
        const l2_ps = l2.getParams();

        for (&l1_ps) |l1_p| {
            if (l1_p) |p| {
                var gp = p.detatchGrad();
                //  defer gp.destroy();

                try gp.asUntagged(F).data.scale(lr, &stream);
                try p.asUntagged(F).data.sub(&gp.asUntagged(F).data, &stream);
            }
        }

        for (&l2_ps) |l2_p| {
            if (l2_p) |p| {
                var gp = p.detatchGrad();
                // defer gp.destroy();

                try gp.asUntagged(F).data.scale(lr, &stream);
                try p.asUntagged(F).data.sub(&gp.asUntagged(F).data, &stream);
            }
        }
        if (i % 1000 == 0) {
            var pi_2v: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 1, 1 }, &stream);
            errdefer pi_2v.deinitAsync(&stream);
            try pi_2v.fill(std.math.pi / 2.0, &stream);

            const pi_2 = try iter_chain.createVariable(F, pi_2v.move(), "pi/2");

            const pi_2_y = try predict3(pi_2, &l1, &l2, iter_chain);
            try stream.sync();

            var loss_host = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer loss_host.deinit(allocator);

            var pi_2_y_host = try pi_2_y.asUntagged(F).data.toHost(allocator, &stream);
            defer pi_2_y_host.deinit(allocator);
            try stream.sync();

            std.debug.print("loss: {d}", .{loss_host});
            std.debug.print("pi_2: {d}\n", .{pi_2_y_host});
        }

        try stream.sync();
        iter_chain.clear();

        // try stream.sync();

    }
}

fn example5() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 100, 1 }, &stream);
    defer xv.deinitAsync(&stream);
    try xv.fillUniformRange(0.0, std.math.pi, &cuda_context, &stream);

    var noisev: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 100, 1 }, &stream);
    defer noisev.deinitAsync(&stream);
    try noisev.fillUniformRange(-0.001, 0.001, &cuda_context, &stream);

    var yv = try xv.cloneAsync(&stream);
    defer yv.deinitAsync(&stream);
    try yv.sin(&stream);

    try yv.add(&noisev, &stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const y = try base_chain.createVariable(F, yv.move(), "y");

    var model: tomorin.layer.MLP(F, 2) = try .init(&.{ 10, 1 }, &context, base_chain);
    defer model.destroy();

    // var optimizer :tomorin.optimizer.MomentumSGD(F)= try .init(.default, &context);
    // var optimizer:tomorin.optimizer.AdaGrad(F) = try .init(.default,  &context);
    // var optimizer:tomorin.optimizer.AdaDelta(F) = try .init(.default,  &context);
    // optimizer: tomorin.optimizer.Adam(F) = try .init(.default, &context);
    var optimizer: tomorin.optimizer.AdamW(F) = try .init(.default, &context);
    defer optimizer.deinit();

    const iter_chain = try context.createChain();
    context.current_chain = iter_chain;

    const iters = 10000;
    for (0..iters + 1) |i| {
        const y_pred = try model.forward(x, sigmoidEx, iter_chain);

        const loss = try meanSquaredErrorEx(F, y, y_pred, iter_chain);

        model.clearGrads();
        y.setGrad(null);
        x.setGrad(null);

        try loss.backward();

        try optimizer.update(&model.getParams());

        if (i % 1000 == 0) {
            std.debug.print("{}\n", .{i / 1000});
            try stream.sync();
            var pi_2v: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 1, 1 }, &stream);
            errdefer pi_2v.deinitAsync(&stream);
            try pi_2v.fill(std.math.pi / 2.0, &stream);

            const pi_2 = try iter_chain.createVariable(F, pi_2v.move(), "pi/2");

            const pi_2_y = try model.forward(pi_2, sigmoidEx, iter_chain);
            try stream.sync();

            var loss_host = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer loss_host.deinit(allocator);

            var pi_2_y_host = try pi_2_y.asUntagged(F).data.toHost(allocator, &stream);
            defer pi_2_y_host.deinit(allocator);
            try stream.sync();

            std.debug.print("loss: {d}", .{loss_host});
            std.debug.print("pi_2: {d}\n", .{pi_2_y_host});
        }

        // try stream.sync();
        iter_chain.clear();

        // try stream.sync();

    }
}

fn example6() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 4, 2 }, &stream);
    defer xv.deinitAsync(&stream);
    try xv.writeFromHostAsync(&.{ 0.2, -0.4, 0.3, 0.5, 1.3, -3.2, 2.0, 0.3 }, 0, &stream);

    var tv: tomo.tensor.GPUTensor(usize) = try .initAsync(&.{4}, &stream);
    defer tv.deinitAsync(&stream);
    try tv.writeFromHostAsync(&.{ 2, 0, 1, 0 }, 0, &stream);

    var t_one_hot = try tv.toOneHot(F, 3, &stream);
    defer t_one_hot.deinitAsync(&stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const t = try base_chain.createVariable(F, t_one_hot.move(), "t");

    var model: tomorin.layer.MLP(F, 2) = try .init(&.{ 10, 3 }, &context, base_chain);
    defer model.destroy();

    const y = try model.forward(x, sigmoidEx, base_chain);
    const loss = try softmaxCrossEntropyEx(F, y, t, &.{1}, base_chain);

    try loss.backward();

    try dbg(F, &loss.asUntagged(F).data, &context);

    try stream.sync();
}

// TODO: make dataset, dataloader first
// contains gputensor Batch -> return ?*GPutensor at .next() like iterator
// cpu data -> fetch -> write at gpu data -> return gputensor
fn example7() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;

    const spiral = try tomorin.datasets.getSpiralAlloc(F, allocator, true);
    defer allocator.free(spiral.x);
    defer allocator.free(spiral.t);

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ 300, 2 }, &stream);
    defer xv.deinitAsync(&stream);

    var tv: tomo.tensor.GPUTensor(usize) = try .initAsync(&.{300}, &stream);
    defer tv.deinitAsync(&stream);

    try xv.writeFromHostAsync(spiral.x, 0, &stream);
    try tv.writeFromHostAsync(spiral.t, 0, &stream);

    var t_one_hot = try tv.toOneHot(F, 3, &stream);
    defer t_one_hot.deinitAsync(&stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const t = try base_chain.createVariable(F, t_one_hot.move(), "t");

    const max_epoch = 100000;
    const hidden_size: comptime_int = 20;
    // const lr = 1.0;

    var model: tomorin.layer.MLP(F, 3) = try .init(&.{ hidden_size, hidden_size, 3 }, &context, base_chain);
    defer model.destroy();

    //var optimizer: tomorin.optimizer.SGD(F) = try .init(.{ .lr = 0.02 }, &context);
    var optimizer: tomorin.optimizer.AdamW(F) = try .init(.default, &context);
    defer optimizer.deinit();

    const iter_chain = try context.createChain();
    context.current_chain = iter_chain;

    for (0..max_epoch) |epoch| {
        const y = try model.forward(x, sigmoidEx, iter_chain);
        const loss = try softmaxCrossEntropyEx(F, y, t, &.{1}, iter_chain);

        x.clearGrad();
        t.clearGrad();
        model.clearGrads();
        try loss.backwardEx(iter_chain);

        try optimizer.update(&model.getParams());

        if (epoch % 1000 == 0) {
            std.debug.print("{}\n", .{epoch / 1000});

            var host_loss = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            var host_y = try y.asUntagged(F).data.toHost(allocator, &stream);
            defer host_y.deinit(allocator);

            try stream.sync();
            std.debug.print("epoch {} loss {d}\n", .{ epoch + 1, host_loss.at(&.{ 0, 0 }).* });
            // std.debug.print("loss {d}\n", .{host_loss});
            // std.debug.print("y {d}\n", .{host_y});
        }

        iter_chain.clear();
    }
}

fn example8() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;

    var spiral: tomorin.datasets.SpiralDataset(F) = try .init(allocator);
    defer spiral.deinit();

    var data_loader: tomorin.dataloader.DataLoader(@TypeOf(spiral)) = try .init(
        allocator,
        &spiral,
        50,
        true,
        &context,
    );
    defer data_loader.deinit();

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ data_loader.batch_size, tomorin.datasets.SpiralDataset(F).input_dim }, &stream);
    defer xv.deinitAsync(&stream);

    var tv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ data_loader.batch_size, tomorin.datasets.SpiralDataset(F).num_class }, &stream);
    defer tv.deinitAsync(&stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const t = try base_chain.createVariable(F, tv.move(), "t");

    const max_epoch = 100001;
    const hidden_size: comptime_int = 20;
    // const lr = 1.0;

    var model: tomorin.layer.MLP(F, 3) = try .init(&.{ hidden_size, hidden_size, 3 }, &context, base_chain);
    defer model.destroy();

    //var optimizer: tomorin.optimizer.SGD(F) = try .init(.{ .lr = 0.02 }, &context);
    var optimizer: tomorin.optimizer.AdamW(F) = try .init(.default, &context);
    defer optimizer.deinit();

    const iter_chain = try context.createChain();
    context.current_chain = iter_chain;

    const start = std.time.timestamp();

    for (0..max_epoch) |epoch| {
        var sum_loss: F = 0.0;
        var sum_acc: F = 0.0;

        while (try data_loader.writeNextBatch(.{ &x.asUntagged(F).data, &t.asUntagged(F).data })) |_| {
            const y = try model.forward(x, sigmoidEx, iter_chain);
            const loss = try softmaxCrossEntropyEx(F, y, t, &.{1}, iter_chain);

            const acc = try tomorin.util.accuracy(F, y, t, tomorin.datasets.SpiralDataset(F).num_class);

            x.clearGrad();
            t.clearGrad();
            model.clearGrads();
            try loss.backwardEx(iter_chain);

            try optimizer.update(&model.getParams());

            var host_loss = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            try stream.sync();
            sum_loss += host_loss.at(&.{ 0, 0 }).*;
            sum_acc += acc;
        }

        if (epoch % 1000 == 0) {
            const len = @as(F, @floatFromInt(tomorin.datasets.SpiralDataset(F).num_data)) / @as(F, @floatFromInt(data_loader.batch_size));
            std.debug.print("epoch {} avg loss {d} acc {d}\n", .{ epoch + 1, sum_loss / len, sum_acc / len });
        }

        iter_chain.clear();
    }

    const end = std.time.timestamp();

    std.debug.print("done - elapsed {}\n", .{end - start});
}

fn example9() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 100,
        .init_var_capacity = 100,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    defer base_chain.destroy();
    context.current_chain = base_chain;

    var mnist: tomorin.datasets.MNISTDataset(F) = try .init(allocator, true);
    defer mnist.deinit();

    var data_loader: tomorin.dataloader.DataLoader(@TypeOf(mnist)) = try .init(
        allocator,
        &mnist,
        100,
        true,
        &context,
    );
    defer data_loader.deinit();

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ data_loader.batch_size, mnist.image_data.rows * mnist.image_data.columns }, &stream);
    defer xv.deinitAsync(&stream);

    var tv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ data_loader.batch_size, 10 }, &stream);
    defer tv.deinitAsync(&stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const t = try base_chain.createVariable(F, tv.move(), "t");

    const max_epoch = 10;
    const hidden_size: comptime_int = 100;
    // const lr = 1.0;

    var model: tomorin.layer.MLP(F, 7) = try .init(&(.{hidden_size} ** 6 ++ .{10}), .he_normal, &context, base_chain);
    defer model.destroy();

    //var optimizer: tomorin.optimizer.SGD(F) = try .init(.{ .lr = 0.02 }, &context);
    var optimizer: tomorin.optimizer.AdamW(F) = try .init(.default, &context);
    defer optimizer.deinit();

    const iter_chain = try context.createChain();
    defer iter_chain.destroy();
    context.current_chain = iter_chain;

    var timer = try std.time.Timer.start();
    for (0..max_epoch) |epoch| {
        var sum_loss: F = 0.0;
        var sum_acc: F = 0.0;
        timer.reset();

        while (try data_loader.writeNextBatch(.{ &x.asUntagged(F).data, &t.asUntagged(F).data })) |_| {
            const y = try model.forward(x, geluEx, iter_chain);
            const loss = try softmaxCrossEntropyEx(F, y, t, &.{1}, iter_chain);

            const acc = try tomorin.util.accuracy(F, y, t);

            x.clearGrad();
            t.clearGrad();
            model.clearGrads();
            try loss.backwardEx(iter_chain);

            try optimizer.update(&model.getParams());

            var host_loss = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            try stream.sync();
            sum_loss += host_loss.at(&.{ 0, 0 }).*;
            sum_acc += acc;
            iter_chain.clear();

            // std.debug.print("epoch {} loss {d} acc {d}\n", .{
            //     epoch + 1,
            //     host_loss.at(&.{ 0, 0 }).*,
            //     acc,
            // });
        }

        const len: F = @floatFromInt(data_loader.max_iter);
        const elapsed = timer.lap();

        std.debug.print("epoch {} avg loss {d} acc {d} elapsed {d}\n", .{
            epoch + 1,
            sum_loss / len,
            sum_acc / len,
            @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(std.time.ns_per_s)),
        });
    }
}

fn example10() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    defer base_chain.destroy();
    context.current_chain = base_chain;

    var mnist: tomorin.datasets.MNISTDataset(F) = try .init(allocator, true);
    defer mnist.deinit();

    var data_loader: tomorin.dataloader.DataLoader(@TypeOf(mnist)) = try .init(
        allocator,
        &mnist,
        100,
        true,
        &context,
    );
    defer data_loader.deinit();

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ data_loader.batch_size, mnist.image_data.rows * mnist.image_data.columns }, &stream);
    defer xv.deinitAsync(&stream);

    var tv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{ data_loader.batch_size, 10 }, &stream);
    defer tv.deinitAsync(&stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const t = try base_chain.createVariable(F, tv.move(), "t");

    const max_epoch = 10;
    const hidden_size: comptime_int = 10;
    // const lr = 1.0;

    var model: tomorin.layer.MLP(F, 3) = try .init(&.{ hidden_size, hidden_size, 10 }, .he_normal, &context, base_chain);
    defer model.destroy();

    //var optimizer: tomorin.optimizer.SGD(F) = try .init(.{ .lr = 0.02 }, &context);
    var optimizer: tomorin.optimizer.AdamW(F) = try .init(.default, &context);
    defer optimizer.deinit();

    const iter_chain = try context.createChain();
    defer iter_chain.destroy();
    context.current_chain = iter_chain;

    var timer = try std.time.Timer.start();
    for (0..max_epoch) |epoch| {
        var sum_loss: F = 0.0;
        var sum_acc: F = 0.0;
        timer.reset();

        while (try data_loader.writeNextBatch(.{ &x.asUntagged(F).data, &t.asUntagged(F).data })) |_| {
            const y = try model.forward(x, geluEx, iter_chain);
            const loss = try softmaxCrossEntropyEx(F, y, t, &.{1}, iter_chain);

            const acc = try tomorin.util.accuracy(F, y, t);

            x.clearGrad();
            t.clearGrad();
            model.clearGrads();
            try loss.backwardEx(iter_chain);

            try optimizer.update(&model.getParams());

            var host_loss = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            try stream.sync();
            sum_loss += host_loss.at(&.{ 0, 0 }).*;
            sum_acc += acc;
            iter_chain.clear();
        }

        const len: F = @floatFromInt(data_loader.max_iter);
        const elapsed = timer.lap();

        std.debug.print("epoch {} avg loss {d} acc {d} elapsed {d}\n", .{
            epoch + 1,
            sum_loss / len,
            sum_acc / len,
            @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(std.time.ns_per_s)),
        });
    }

    try model.saveJsonStringField(allocator, "out.json");

    try stream.sync();
    try model.loadJsonStringField(allocator, "out.json");

    try stream.sync();
    try model.saveJsonStringField(allocator, "out2.json");

    var out = try std.fs.cwd().openFile("out.json", .{ .mode = .read_only });
    defer out.close();

    const src1 = try out.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(src1);

    var out2 = try std.fs.cwd().openFile("out2.json", .{ .mode = .read_only });
    defer out2.close();

    const src2 = try out2.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(src2);

    std.debug.assert(std.mem.eql(u8, src1, src2));

    for (0..max_epoch) |epoch| {
        var sum_loss: F = 0.0;
        var sum_acc: F = 0.0;
        timer.reset();

        while (try data_loader.writeNextBatch(.{ &x.asUntagged(F).data, &t.asUntagged(F).data })) |_| {
            const y = try model.forward(x, &geluEx, iter_chain);
            const loss = try softmaxCrossEntropyEx(F, y, t, &.{1}, iter_chain);

            const acc = try tomorin.util.accuracy(F, y, t);

            x.clearGrad();
            t.clearGrad();
            model.clearGrads();
            try loss.backwardEx(iter_chain);

            try optimizer.update(&model.getParams());

            var host_loss = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            try stream.sync();
            sum_loss += host_loss.at(&.{ 0, 0 }).*;
            sum_acc += acc;
            iter_chain.clear();
        }

        const len: F = @floatFromInt(data_loader.max_iter);
        const elapsed = timer.lap();

        std.debug.print("epoch {} avg loss {d} acc {d} elapsed {d}\n", .{
            epoch + 1,
            sum_loss / len,
            sum_acc / len,
            @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(std.time.ns_per_s)),
        });
    }
}

fn example11() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 100,
        .init_var_capacity = 100,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    defer base_chain.destroy();
    context.current_chain = base_chain;

    var cifar10: tomorin.datasets.CIFAR10Dataset(F) = try .init(allocator, true);
    defer cifar10.deinit();

    //std.debug.print("{}", .{cifar10.total_records});

    var data_loader: tomorin.dataloader.DataLoader(@TypeOf(cifar10)) = try .init(
        allocator,
        &cifar10,
        128,
        true,
        &context,
    );
    defer data_loader.deinit();

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{
        data_loader.batch_size,
        3,
        tomorin.datasets.CIFAR10Dataset(F).out_width,
        tomorin.datasets.CIFAR10Dataset(F).out_height,
    }, &stream);
    defer xv.deinitAsync(&stream);

    var tv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{
        data_loader.batch_size,
        tomorin.datasets.CIFAR10Dataset(F).num_classes,
    }, &stream);
    defer tv.deinitAsync(&stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const t = try base_chain.createVariable(F, tv.move(), "t");

    const max_epoch = 100;
    // const lr = 1.0;

    var model: tomorin.layer.VGG16(F) = try .init(10, &context, base_chain);
    defer model.destroy();

    //var optimizer: tomorin.optimizer.SGD(F) = try .init(.{ .lr = 0.02 }, &context);
    var optimizer: tomorin.optimizer.AdamW(F) = try .init(.default, &context);
    defer optimizer.deinit();

    const iter_chain = try context.createChain();
    defer iter_chain.destroy();
    context.current_chain = iter_chain;

    var timer = try std.time.Timer.start();
    for (0..max_epoch) |epoch| {
        var sum_loss: F = 0.0;
        var sum_acc: F = 0.0;
        timer.reset();

        var i: usize = 0;
        while (try data_loader.writeNextBatch(.{ &x.asUntagged(F).data, &t.asUntagged(F).data })) |_| : (i += 1) {
            const y = try model.forward(x, true, iter_chain);

            //  std.debug.print("{any} {any}\n", .{ y.getShape(), t.getShape() });

            // if (i == 3) {
            //     // try dbg(F, &model.fields.conv4_1.fields.w.?.asUntaggedConst(F).data, &context);
            // }

            // try dbg(F, &t.asUntaggedConst(F).data, t.getContext());
            // try dbg(F, &y.asUntaggedConst(F).data, y.getContext());

            const loss = try softmaxCrossEntropyEx(F, y, t, &.{1}, iter_chain);

            const acc = try tomorin.util.accuracy(F, y, t);

            x.clearGrad();
            t.clearGrad();
            model.clearGrads();

            try loss.backwardEx(iter_chain);

            try optimizer.update(&model.getParams());

            var host_loss = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            try stream.sync();
            sum_loss += host_loss.at(&.{ 0, 0 }).*;
            sum_acc += acc;

            iter_chain.clear();
            try stream.sync();

            std.debug.print("epoch {} loss {d} acc {d}\n", .{
                epoch + 1,
                host_loss.at(&.{ 0, 0 }).*,
                acc,
            });

            // if (i == 0) {
            //     try dbg(F, &model.fields.out.fields.w.?.asUntaggedConst(F).data, &context);
            // }

            // std.debug.print("epoch {} loss {d} acc {d}\n", .{
            //     epoch + 1,
            //     host_loss.at(&.{ 0, 0 }).*,
            //     acc,
            // });
        }

        const len: F = @floatFromInt(data_loader.max_iter);
        const elapsed = timer.lap();

        std.debug.print("epoch {} avg loss {d} acc {d} elapsed {d}\n", .{
            epoch + 1,
            sum_loss / len,
            sum_acc / len,
            @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(std.time.ns_per_s)),
        });
    }
    try stream.sync();

    try model.saveJsonStringField(allocator, "vgg16_out.json");
}

fn example12() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream: tomo.stream.Stream = try .create();
    defer stream.destroy();

    var cuda_context: tomo.cuda_context.CudaContext = try .init();
    defer cuda_context.deinit();

    var context: tomorin.context.Context = try .init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 100,
        .init_var_capacity = 100,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    defer base_chain.destroy();
    context.current_chain = base_chain;

    var cifar10: tomorin.datasets.CIFAR10Dataset(F) = try .init(allocator, true);
    defer cifar10.deinit();

    //std.debug.print("{}", .{cifar10.total_records});

    var data_loader: tomorin.dataloader.DataLoader(@TypeOf(cifar10)) = try .init(
        allocator,
        &cifar10,
        100,
        true,
        &context,
    );
    defer data_loader.deinit();

    var xv: tomo.tensor.GPUTensor(F) = try .initAsync(&.{
        data_loader.batch_size,
        3,
        tomorin.datasets.CIFAR10Dataset(F).out_width,
        tomorin.datasets.CIFAR10Dataset(F).out_height,
    }, &stream);
    defer xv.deinitAsync(&stream);

    var tv: tomo.tensor.GPUTensor(usize) = try .initAsync(&.{
        data_loader.batch_size,
    }, &stream);
    defer tv.deinitAsync(&stream);

    const x = try base_chain.createVariable(F, xv.move(), "x");
    const t = try base_chain.createVariable(usize, tv.move(), "t");

    const max_epoch = 1;
    // const lr = 1.0;

    var model: tomorin.layer.ResNet50(F) = try .init(
        tomorin.datasets.CIFAR10Dataset(F).num_classes,
        &context,
        base_chain,
    );
    defer model.destroy();

    //var optimizer: tomorin.optimizer.SGD(F) = try .init(.{ .lr = 0.02 }, &context);
    var optimizer: tomorin.optimizer.AdamW(F) = try .init(.default, &context);
    defer optimizer.deinit();

    const iter_chain = try context.createChain();
    defer iter_chain.destroy();
    context.current_chain = iter_chain;

    // try model.loadJsonStringField(allocator, "resnet50_out.json");
    _ = try model.forward(x, true, iter_chain);
    x.clearGrad();
    t.clearGrad();
    model.clearGrads();
    //  try model.loadJsonStringField(allocator, "resnet50_out.json");
    //    try model.loadBinary(allocator, "resnet50_out.bin");

    var timer = try std.time.Timer.start();

    for (0..max_epoch) |epoch| {
        var sum_loss: F = 0.0;
        var sum_acc: F = 0.0;
        timer.reset();

        var i: usize = 0;
        while (try data_loader.writeNextBatch(.{ &x.asUntagged(F).data, &t.asUntagged(usize).data })) |_| : (i += 1) {
            const y = try model.forward(x, true, iter_chain);

            const loss = try softmaxCrossEntropyEx(F, y, .{ .t = t }, iter_chain);

            const acc = try tomorin.util.accuracy(F, y, t, 1);

            x.clearGrad();
            t.clearGrad();
            model.clearGrads();

            try loss.backwardEx(iter_chain);

            try optimizer.update(&model.getParams());

            var host_loss = try loss.asUntagged(F).data.toHost(allocator, &stream);
            defer host_loss.deinit(allocator);

            try stream.sync();
            sum_loss += host_loss.at(&.{ 0, 0 }).*;
            sum_acc += acc;

            iter_chain.clear();

            try stream.sync();

            std.debug.print("epoch {} loss {d} acc {d}\n", .{
                epoch + 1,
                host_loss.at(&.{ 0, 0 }).*,
                acc,
            });
        }

        const len: F = @floatFromInt(data_loader.max_iter);
        const elapsed = timer.lap();

        std.debug.print("epoch {} avg loss {d} acc {d} elapsed {d}\n", .{
            epoch + 1,
            sum_loss / len,
            sum_acc / len,
            @as(f32, @floatFromInt(elapsed)) / @as(f32, @floatFromInt(std.time.ns_per_s)),
        });
    }
    try stream.sync();

    //try model.saveJsonStringField(allocator, "resnet50_out.json");
    // try model.saveBinary(allocator, "resnet50_out.bin");
}

// TODO: make metaprogramming tools that makes program easier
pub fn main() !void {
    // try example4();
    // try example9();
    try example12();
    // try function.testFunctions();
}
