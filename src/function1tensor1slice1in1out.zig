const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;

const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;

const Function = @import("function.zig").Function;
const FunctionBase = @import("function.zig").FunctionBase;
const FuncDecorator1in1outBase = @import("function.zig").FuncDecorator1in1outBase;
const Chain = @import("chain.zig").Chain;
const makefunc1in1outBase = @import("function.zig").makefunc1in1outBase;

const addEx = @import("function2in1out.zig").addEx;
const expEx = @import("function1in1out.zig").expEx;
const scaleEx = @import("function1scalar1in1out.zig").scaleEx;
const shiftEx = @import("function1scalar1in1out.zig").shiftEx;
const mulEx = @import("function2in1out.zig").mulEx;
const divEx = @import("function2in1out.zig").divEx;
const subEx = @import("function2in1out.zig").subEx;
const sumEx = @import("function1in1out.zig").sumEx;
const broadcastToEx = @import("function1slice1in1out.zig").broadcastToEx;
const logSoftmaxEx = @import("function1slice1in1out.zig").logSoftmaxEx;
const softmaxEx = @import("function1slice1in1out.zig").softmaxEx;

const dbg = @import("util.zig").debugPrintGpuTensor;

// TODO: 1in1outBase -> 1in1scalar, 1in2scalar ...

pub fn FuncDecorator1tensor1slice1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, t: *TaggedVar, slice: anytype, chain: *Chain) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try chain.registerFunction(
                .{
                    .ptr = self,
                    .vtable = &.{
                        .forward = &Base.forwardDecorated,
                        .backward = &Base.backwardDecorated,
                        .destroy = &Base.destroy,
                        .get_generation = &Base.getGeneration,
                        .enqueue = &Base.enqueue,
                        .get_dot_alloc = &getDotAlloc,
                    },
                    .chain = chain,
                },
            );

            self.* = .{
                .in = null,
                .out = null,
                .t = t,
                .slice = slice,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
                    .chain = chain,
                },
            };

            return func_ptr;
        }

        pub fn getDotAlloc(ctx: *anyopaque, var_seen_set: *TaggedVar.SeenSet) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.context.allocator;

            const in_contains = var_seen_set.contains(self.in.?);
            const in = if (!in_contains) try self.in.?.getDotAlloc() else "";
            defer if (!in_contains) allocator.free(in);

            try var_seen_set.put(self.in.?, {});

            const out_contains = var_seen_set.contains(self.out.?);
            const out = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out);

            const t_contains = var_seen_set.contains(self.t);
            const t = if (!t_contains) try self.t.getDotAlloc() else "";
            defer if (!t_contains) allocator.free(t);

            try var_seen_set.put(self.out.?, {});

            return try std.fmt.allocPrint(allocator,
                \\{} [label="{s}", color=lightblue, style=filled, shape=box]
                \\{s}
                \\{s}
                \\{s}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                in,
                out,
                t,
                @intFromPtr(self.in.?),
                @intFromPtr(ctx),
                @intFromPtr(self.t),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(comptime F: type, x: *TaggedVar, t: *TaggedVar, slice: anytype, chain: *Chain) !*TaggedVar {
    const funckey = try F.create(x.getContext(), t, slice, chain);

    return try makefunc1in1outBase(funckey, x);
}

pub fn SoftmaxCrossEntropy(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        t: *TaggedVar,
        slice: ?[]const isize, // softmax axis
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1tensor1slice1in1out(Self);

        const Self = SoftmaxCrossEntropy(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            const batch_size: T = @floatFromInt(x.base.getShapeConst()[0]);

            var x_clone = try x.cloneAsync(context.stream);
            defer x_clone.deinitAsync(context.stream);

            const x_var = try self.base.chain.createVariable(T, x_clone.move(), null);
            defer x_var.destroy();

            var logsm = try logSoftmaxEx(T, x_var, self.slice.?, self.base.chain);
            defer logsm.destroy();

            var prod = try mulEx(T, logsm, self.t, self.base.chain);
            defer prod.destroy();

            //try dbg(T, &logsm.asUntagged(T).data, context);
            //try dbg(T, &prod.asUntagged(T).data, context);

            var loss = try sumEx(T, prod, null, self.base.chain);
            defer loss.destroy();

            var scale = try scaleEx(T, loss, -1.0 / batch_size, self.base.chain);
            // var scale = try scaleEx(T, loss, 1.0 / batch_size, self.base.chain);
            defer scale.destroy();

            return scale.asUntagged(T).data.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const batch_size = self.in.?.getShape()[0];

            const y = try softmaxEx(T, self.in.?, self.slice.?, self.base.chain);

            const y_min_onehot = try subEx(T, y, self.t, self.base.chain);

            const gy_scaled = try scaleEx(
                T,
                gy,
                1.0 / @as(T, @floatFromInt(batch_size)),
                self.base.chain,
            );

            return try mulEx(
                T,
                try broadcastToEx(T, gy_scaled, y_min_onehot.getShape(), self.base.chain),
                y_min_onehot,
                self.base.chain,
            );
        }
    };
}

pub fn softmaxCrossEntropy(comptime T: type, logits: *TaggedVar, target: *TaggedVar, axis: []const isize) !*TaggedVar {
    return try softmaxCrossEntropyEx(T, logits, target, axis, logits.getContext().current_chain.?);
}

pub fn softmaxCrossEntropyEx(comptime T: type, logits: *TaggedVar, target: *TaggedVar, axis: []const isize, chain: *Chain) !*TaggedVar {
    return try makefunc(SoftmaxCrossEntropy(T), logits, target, axis, chain);
}

// tests
fn testsoftmaxCrossEntropyforward(allocator: std.mem.Allocator) !void {
    // 1. Environment Setup
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // 2. Input Tensor Creation
    // Logits: 2x3 matrix [[1, 2, 3], [4, 5, 6]]
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var logits_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_logits = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_logits.deinitAsync(&stream);
    try gpu_logits.writeFromHostAsync(&logits_data, 0, &stream);

    var var_logits = try base_chain.createVariable(T, gpu_logits.move(), "logits");
    defer var_logits.destroy();

    // Target: one-hot encoded for classes 2 and 0 [[0, 0, 1], [1, 0, 0]]
    var target_data = [_]T{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 };
    var gpu_target = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_target.deinitAsync(&stream);
    try gpu_target.writeFromHostAsync(&target_data, 0, &stream);

    var var_target = try base_chain.createVariable(T, gpu_target.move(), "target");
    defer var_target.destroy();

    // 3. Forward Pass
    const axis = &[_]isize{1}; // Apply softmax along axis 1 (columns)
    var var_loss = try softmaxCrossEntropyEx(T, var_logits, var_target, axis, base_chain);
    defer var_loss.destroy();

    // Transfer loss to host
    var gpu_loss = var_loss.asUntagged(T).data;
    var host_loss = try gpu_loss.toHost(allocator, &stream);
    defer host_loss.deinit(allocator);

    try stream.sync();

    // 4. Manual Calculation of Expected Loss
    const batch_size = shape[0];
    var expected_loss: T = 0.0;

    for (0..batch_size) |b| {
        // Compute exp(logits) sum
        var exp_sum: T = 0.0;
        for (0..shape[1]) |c| {
            exp_sum += std.math.exp(logits_data[b * shape[1] + c]);
        }
        //const log_sum_exp = std.math.log(exp_sum);

        // Find target class
        var target_class: usize = 0;
        for (0..shape[1]) |c| {
            if (target_data[b * shape[1] + c] == 1.0) {
                target_class = c;
                break;
            }
        }

        // Compute -log(softmax[target_class])
        const softmax_target = std.math.exp(logits_data[b * shape[1] + target_class]) / exp_sum;
        expected_loss += -std.math.log(T, std.math.e, softmax_target);
    }
    expected_loss /= @as(T, @floatFromInt(batch_size));

    // 5. Comparison
    const computed_loss = host_loss.data[0];
    const tolerance = 1e-5;
    if (@abs(computed_loss - expected_loss) > tolerance) {
        std.debug.print("Computed loss: {}, Expected loss: {}\n", .{ computed_loss, expected_loss });
        return error.TestFailed;
    }
    std.debug.print("SoftmaxCrossEntropy forward test passed.\n", .{});
}

fn testsoftmaxCrossEntropyBackward(allocator: std.mem.Allocator) !void {
    // 1. Environment Setup
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // 2. Input Tensor Creation
    // Logits: 2x3 matrix [[1, 2, 3], [4, 5, 6]]
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var logits_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_logits = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_logits.deinitAsync(&stream);
    try gpu_logits.writeFromHostAsync(&logits_data, 0, &stream);

    var var_logits = try base_chain.createVariable(T, gpu_logits.move(), "logits");
    defer var_logits.destroy();

    // Target: one-hot encoded for classes 2 and 0 [[0, 0, 1], [1, 0, 0]]
    var target_data = [_]T{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 };
    var gpu_target = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_target.deinitAsync(&stream);
    try gpu_target.writeFromHostAsync(&target_data, 0, &stream);

    var var_target = try base_chain.createVariable(T, gpu_target.move(), "target");
    defer var_target.destroy();

    // 3. Forward Pass
    const axis = &[_]isize{1}; // Apply softmax along axis 1 (columns)
    var var_loss = try softmaxCrossEntropyEx(T, var_logits, var_target, axis, base_chain);
    defer var_loss.destroy();

    // 4. Backward Pass
    // Set output gradient (gy) to 1.0 (scalar loss)
    var gy_data = [_]T{1.0};
    var gpu_gy = try GPUTensor(T).initAsync(&[_]usize{1}, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();

    try var_loss.backwardEx(base_chain);

    // Retrieve gradients with respect to logits
    var gpu_gx = var_logits.refGradConst().?.asUntaggedConst(T).data;
    var host_gx = try gpu_gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    try stream.sync();

    // 5. Numerical Gradient Computation
    const epsilon = 1e-4;
    var numerical_gx = try allocator.alloc(T, logits_data.len);
    defer allocator.free(numerical_gx);

    for (0..logits_data.len) |i| {
        // Perturb logits[i] by +epsilon
        logits_data[i] += epsilon;
        var gpu_logits_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_logits_plus.writeFromHostAsync(&logits_data, 0, &stream);
        const var_logits_plus = try base_chain.createVariable(T, gpu_logits_plus.move(), "logits_plus");
        var var_loss_plus = try softmaxCrossEntropyEx(T, var_logits_plus, var_target, axis, base_chain);
        var host_loss_plus = try var_loss_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_loss_plus.deinit(allocator);

        // Perturb logits[i] by -epsilon
        logits_data[i] -= 2 * epsilon;
        var gpu_logits_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_logits_minus.writeFromHostAsync(&logits_data, 0, &stream);
        const var_logits_minus = try base_chain.createVariable(T, gpu_logits_minus.move(), "logits_minus");
        var var_loss_minus = try softmaxCrossEntropyEx(T, var_logits_minus, var_target, axis, base_chain);
        var host_loss_minus = try var_loss_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_loss_minus.deinit(allocator);

        // Central difference formula
        numerical_gx[i] = (host_loss_plus.data[0] - host_loss_minus.data[0]) / (2 * epsilon);

        // Reset logits[i] to original value
        logits_data[i] += epsilon;
    }

    // 6. Comparison
    //std.debug.print("host_gx.data {any} numerical_gx {any}", .{ host_gx.data, numerical_gx });
    for (host_gx.data, numerical_gx) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-3) return error.TestFailed;
    }
    std.debug.print("SoftmaxCrossEntropy test passed.\n", .{});
}

pub fn test1tensor1slice1i1o() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    try testsoftmaxCrossEntropyforward(allocator);
    try testsoftmaxCrossEntropyBackward(allocator);

    std.debug.print("All test1tensor1slice1i1o tests passed.\n", .{});
}
