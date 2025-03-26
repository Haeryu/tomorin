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
const FuncDecorator2in1outBase = @import("function.zig").FuncDecorator2in1outBase;
const Chain = @import("chain.zig").Chain;
const makefunc2in1outBase = @import("function.zig").makefunc2in1outBase;

const negEx = @import("function1in1out.zig").negEx;
const expEx = @import("function1in1out.zig").expEx;
const powEx = @import("function1scalar1in1out.zig").powEx;
const scaleEx = @import("function1scalar1in1out.zig").scaleEx;
const squareEx = @import("function1in1out.zig").squareEx;
const transposeEx = @import("function1in1out.zig").transposeEx;
const broadcastToEx = @import("function1slice1in1out.zig").broadcastToEx;
const transposeExEx = @import("function1slice1in1out.zig").transposeExEx;

pub fn FuncDecorator2in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator2in1outBase(Self);

        pub fn create(context: *Context, chain: *Chain) !*Function {
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
                .in1 = null,
                .in2 = null,
                .out = null,
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

            const in1_contains = var_seen_set.contains(self.in1.?);
            const in1 = if (!in1_contains) try self.in1.?.getDotAlloc() else "";
            defer if (!in1_contains) allocator.free(in1);

            try var_seen_set.put(self.in1.?, {});

            const in2_contains = var_seen_set.contains(self.in2.?);
            const in2 = if (!in2_contains) try self.in2.?.getDotAlloc() else "";
            defer if (!in2_contains) allocator.free(in2);

            try var_seen_set.put(self.in2.?, {});

            const out_contains = var_seen_set.contains(self.out.?);
            const out = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out);

            try var_seen_set.put(self.out.?, {});

            return try std.fmt.allocPrint(self.base.context.allocator,
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
                in1,
                in2,
                out,
                @intFromPtr(self.in1.?),
                @intFromPtr(ctx),
                @intFromPtr(self.in2.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(
    comptime F: type,
    x1: *TaggedVar,
    x2: *TaggedVar,
    chain: *Chain,
) !*TaggedVar {
    const funckey = try F.create(x1.getContext(), chain);

    return try makefunc2in1outBase(funckey, x1, x2);
}

pub fn Add(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Add(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x1.cloneAsync(context.stream);
            defer y.deinitAsync(context.stream);

            try y.add(x2, context.stream);

            return y.move();
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            return .{ gy, gy };
        }
    };
}

pub fn add(
    comptime T: type,
    x1: *TaggedVar,
    x2: *TaggedVar,
) !*TaggedVar {
    return try addEx(T, x1, x2, x1.getContext().current_chain.?);
}

pub fn addEx(
    comptime T: type,
    x1: *TaggedVar,
    x2: *TaggedVar,
    chain: *Chain,
) !*TaggedVar {
    return try makefunc(Add(T), x1, x2, chain);
}

pub fn Sub(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Sub(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x1.cloneAsync(context.stream);
            defer y.deinitAsync(context.stream);

            try y.sub(x2, context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            return .{ gy, try negEx(
                Self.In2,
                gy,
                self.base.chain,
            ) };
        }
    };
}

pub fn sub(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try subEx(T, x1, x2, x1.getContext().current_chain.?);
}

pub fn subEx(comptime T: type, x1: *TaggedVar, x2: *TaggedVar, chain: *Chain) !*TaggedVar {
    return try makefunc(Sub(T), x1, x2, chain);
}

pub fn Mul(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Mul(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x1.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.product(x2, context.stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            return .{ try mulEx(In1, gy, self.in2.?, self.base.chain), try mulEx(In2, gy, self.in1.?, self.base.chain) };
        }
    };
}

pub fn mul(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try mulEx(T, x1, x2, x1.getContext().current_chain.?);
}

pub fn mulEx(comptime T: type, x1: *TaggedVar, x2: *TaggedVar, chain: *Chain) !*TaggedVar {
    return try makefunc(Mul(T), x1, x2, chain);
}

pub fn Div(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Div(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x1.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.divide(x2, context.stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const gx0 = try div(T, gy, self.in2.?);

            const x2_sq = try squareEx(T, self.in2.?, self.base.chain);
            const minus_x1 = try negEx(T, self.in1.?, self.base.chain);
            const denom = try divEx(T, minus_x1, x2_sq, self.base.chain);
            const gx1 = try mulEx(T, gy, denom, self.base.chain);

            return .{ gx0, gx1 };
        }
    };
}

pub fn div(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try divEx(T, x1, x2, x1.getContext().current_chain.?);
}

pub fn divEx(comptime T: type, x1: *TaggedVar, x2: *TaggedVar, chain: *Chain) !*TaggedVar {
    return try makefunc(Div(T), x1, x2, chain);
}

pub fn MatMul(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = MatMul(T);

        // pub fn matmul(
        //             self: *const Self,
        //             self_transpose: bool,
        //             other_tensor: anytype,
        //             other_transpose: bool,
        //             add_tensor: anytype,
        //             add_transpose: bool,
        //             comptime cublas_compute_type: c.cublasComputeType_t,
        //             alpha: if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) f16 else f32,
        //             beta: if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) f16 else f32,
        //             comptime EpilogueT: type,
        //             epilogue_config: EpilogueT.Config,
        //             //cublas_compute_type: c.cublasComputeType_t,
        //             stream: *const Stream,
        //             cuda_context: *const CudaContext,
        //             comptime OutType: type,
        //         )

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x1.linearImp(x2, null, context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            var perm: [GPUTensor(T).max_rank]usize = undefined;
            for (perm[0..gy.getShape().len], 0..) |*p, i| {
                p.* = i;
            }
            std.mem.swap(usize, &perm[gy.getShape().len - 1], &perm[gy.getShape().len - 2]);
            const gx = try matmulEx(T, gy, try transposeExEx(T, self.in2.?, perm[0..gy.getShape().len], self.base.chain), self.base.chain);
            const gw = try matmulEx(T, try transposeExEx(T, self.in1.?, perm[0..gy.getShape().len], self.base.chain), gy, self.base.chain);

            return .{ gx, gw };
        }
    };
}

pub fn matmul(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try matmulEx(T, x1, x2, x1.getContext().current_chain.?);
}

pub fn matmulEx(comptime T: type, x1: *TaggedVar, x2: *TaggedVar, chain: *Chain) !*TaggedVar {
    return try makefunc(MatMul(T), x1, x2, chain);
}

pub fn MeanSquaredError(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = MeanSquaredError(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var diff = try x1.cloneAsync(context.stream);
            defer diff.deinitAsync(context.stream);

            try diff.sub(x2, context.stream);

            try diff.product(&diff, context.stream);

            var sum = try diff.sum(context.allocator, null, true, context.stream);
            defer sum.deinitAsync(context.stream);

            try sum.scale(1.0 / @as(T, @floatFromInt(diff.base.countElem())), context.stream);

            return sum.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const diff = try sub(T, self.in1.?, self.in2.?);
            const gy_broad = try broadcastToEx(T, gy, diff.asUntaggedConst(T).data.base.getShapeConst(), self.base.chain);
            const gy_diff = try mulEx(T, gy_broad, diff, self.base.chain);
            const gx0 = try scaleEx(T, gy_diff, 2.0 / @as(T, @floatFromInt(diff.len())), self.base.chain);
            const gx1 = try negEx(T, gx0, self.base.chain);
            return .{ gx0, gx1 };
        }
    };
}

pub fn meanSquaredError(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try meanSquaredErrorEx(T, x1, x2, x1.getContext().current_chain.?);
}

pub fn meanSquaredErrorEx(comptime T: type, x1: *TaggedVar, x2: *TaggedVar, chain: *Chain) !*TaggedVar {
    return try makefunc(MeanSquaredError(T), x1, x2, chain);
}

// tests
fn testAddExForward(allocator: std.mem.Allocator) !void {
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

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x1_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x1 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x1.deinitAsync(&stream);
    try gpu_x1.writeFromHostAsync(&x1_data, 0, &stream);
    var var_x1 = try base_chain.createVariable(T, gpu_x1.move(), "x1");
    defer var_x1.destroy();

    var x2_data = [_]T{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var gpu_x2 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x2.deinitAsync(&stream);
    try gpu_x2.writeFromHostAsync(&x2_data, 0, &stream);
    var var_x2 = try base_chain.createVariable(T, gpu_x2.move(), "x2");
    defer var_x2.destroy();

    var var_y = try addEx(T, var_x1, var_x2, base_chain);
    defer var_y.destroy();

    var gpu_y = var_y.asUntagged(T).data;
    var host_y = try gpu_y.toHost(allocator, &stream);
    defer host_y.deinit(allocator);

    const expected_y = [_]T{ 8.0, 10.0, 12.0, 14.0, 16.0, 18.0 };
    for (host_y.data, expected_y) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testAddExForward passed successfully.\n", .{});
}

fn testAddExBackward(allocator: std.mem.Allocator) !void {
    // Environment Setup
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

    // Input Tensor Creation
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x1_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x1 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x1.deinitAsync(&stream);
    try gpu_x1.writeFromHostAsync(&x1_data, 0, &stream);
    var var_x1 = try base_chain.createVariable(T, gpu_x1.move(), "x1");
    defer var_x1.destroy();

    var x2_data = [_]T{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var gpu_x2 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x2.deinitAsync(&stream);
    try gpu_x2.writeFromHostAsync(&x2_data, 0, &stream);
    var var_x2 = try base_chain.createVariable(T, gpu_x2.move(), "x2");
    defer var_x2.destroy();

    // Forward Pass
    var var_y = try addEx(T, var_x1, var_x2, base_chain);
    defer var_y.destroy();

    // Backward Pass
    var gy_data = try allocator.alloc(T, shape[0] * shape[1]);
    defer allocator.free(gy_data);
    for (0..gy_data.len) |i| gy_data[i] = 1.0; // Output gradient set to all ones
    var gpu_gy = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_y.setGrad(var_gy);

    try var_y.backwardEx(base_chain);

    // Retrieve Gradients
    var gpu_gx1 = var_x1.refGradConst().?.asUntaggedConst(T).data;
    var host_gx1 = try gpu_gx1.toHost(allocator, &stream);
    defer host_gx1.deinit(allocator);

    var gpu_gx2 = var_x2.refGradConst().?.asUntaggedConst(T).data;
    var host_gx2 = try gpu_gx2.toHost(allocator, &stream);
    defer host_gx2.deinit(allocator);

    // Numerical Gradient Computation
    const epsilon = 1e-5;
    var numerical_gx1 = try allocator.alloc(T, x1_data.len);
    defer allocator.free(numerical_gx1);
    var numerical_gx2 = try allocator.alloc(T, x2_data.len);
    defer allocator.free(numerical_gx2);

    // Numerical Gradient for x1
    for (0..x1_data.len) |i| {
        x1_data[i] += epsilon;
        var gpu_x1_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x1_plus.writeFromHostAsync(&x1_data, 0, &stream);
        const var_x1_plus = try base_chain.createVariable(T, gpu_x1_plus.move(), "x1_plus");
        var var_y_plus = try addEx(T, var_x1_plus, var_x2, base_chain);
        var host_y_plus = try var_y_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_plus.deinit(allocator);

        x1_data[i] -= 2 * epsilon;
        var gpu_x1_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x1_minus.writeFromHostAsync(&x1_data, 0, &stream);
        const var_x1_minus = try base_chain.createVariable(T, gpu_x1_minus.move(), "x1_minus");
        var var_y_minus = try addEx(T, var_x1_minus, var_x2, base_chain);
        var host_y_minus = try var_y_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_minus.deinit(allocator);

        numerical_gx1[i] = 0.0;
        for (0..host_y_plus.data.len) |j| {
            numerical_gx1[i] += (host_y_plus.data[j] - host_y_minus.data[j]) / (2 * epsilon) * gy_data[j];
        }
        x1_data[i] += epsilon; // Reset
    }

    // Numerical Gradient for x2
    for (0..x2_data.len) |i| {
        x2_data[i] += epsilon;
        var gpu_x2_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x2_plus.writeFromHostAsync(&x2_data, 0, &stream);
        const var_x2_plus = try base_chain.createVariable(T, gpu_x2_plus.move(), "x2_plus");
        var var_y_plus = try addEx(T, var_x1, var_x2_plus, base_chain);
        var host_y_plus = try var_y_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_plus.deinit(allocator);

        x2_data[i] -= 2 * epsilon;
        var gpu_x2_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x2_minus.writeFromHostAsync(&x2_data, 0, &stream);
        const var_x2_minus = try base_chain.createVariable(T, gpu_x2_minus.move(), "x2_minus");
        var var_y_minus = try addEx(T, var_x1, var_x2_minus, base_chain);
        var host_y_minus = try var_y_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_minus.deinit(allocator);

        numerical_gx2[i] = 0.0;
        for (0..host_y_plus.data.len) |j| {
            numerical_gx2[i] += (host_y_plus.data[j] - host_y_minus.data[j]) / (2 * epsilon) * gy_data[j];
        }
        x2_data[i] += epsilon; // Reset
    }

    // Comparison
    //std.debug.print("{any} {any}", .{ host_gx1.data, numerical_gx1 });
    for (host_gx1.data, numerical_gx1) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-1) return error.TestFailed;
    }
    for (host_gx2.data, numerical_gx2) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-1) return error.TestFailed;
    }

    std.debug.print("testAddExBackward passed successfully.\n", .{});
}

fn testSubExForward(allocator: std.mem.Allocator) !void {
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

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x1_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x1 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x1.deinitAsync(&stream);
    try gpu_x1.writeFromHostAsync(&x1_data, 0, &stream);
    var var_x1 = try base_chain.createVariable(T, gpu_x1.move(), "x1");
    defer var_x1.destroy();

    var x2_data = [_]T{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var gpu_x2 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x2.deinitAsync(&stream);
    try gpu_x2.writeFromHostAsync(&x2_data, 0, &stream);
    var var_x2 = try base_chain.createVariable(T, gpu_x2.move(), "x2");
    defer var_x2.destroy();

    var var_y = try subEx(T, var_x1, var_x2, base_chain);
    defer var_y.destroy();

    var gpu_y = var_y.asUntagged(T).data;
    var host_y = try gpu_y.toHost(allocator, &stream);
    defer host_y.deinit(allocator);

    const expected_y = [_]T{ -6.0, -6.0, -6.0, -6.0, -6.0, -6.0 };
    for (host_y.data, expected_y) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testSubExForward passed successfully.\n", .{});
}

fn testSubExBackward(allocator: std.mem.Allocator) !void {
    // Same setup as testAddExBackward, replacing addEx with subEx
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

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x1_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x1 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x1.deinitAsync(&stream);
    try gpu_x1.writeFromHostAsync(&x1_data, 0, &stream);
    var var_x1 = try base_chain.createVariable(T, gpu_x1.move(), "x1");
    defer var_x1.destroy();

    var x2_data = [_]T{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var gpu_x2 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x2.deinitAsync(&stream);
    try gpu_x2.writeFromHostAsync(&x2_data, 0, &stream);
    var var_x2 = try base_chain.createVariable(T, gpu_x2.move(), "x2");
    defer var_x2.destroy();

    // Forward Pass
    var var_y = try subEx(T, var_x1, var_x2, base_chain);
    defer var_y.destroy();

    // Backward Pass
    var gy_data = try allocator.alloc(T, shape[0] * shape[1]);
    defer allocator.free(gy_data);
    for (0..gy_data.len) |i| gy_data[i] = 1.0;
    var gpu_gy = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_y.setGrad(var_gy);

    try var_y.backwardEx(base_chain);

    // Retrieve Gradients
    var gpu_gx1 = var_x1.refGradConst().?.asUntaggedConst(T).data;
    var host_gx1 = try gpu_gx1.toHost(allocator, &stream);
    defer host_gx1.deinit(allocator);

    var gpu_gx2 = var_x2.refGradConst().?.asUntaggedConst(T).data;
    var host_gx2 = try gpu_gx2.toHost(allocator, &stream);
    defer host_gx2.deinit(allocator);

    // Numerical Gradient Computation
    const epsilon = 1e-5;
    var numerical_gx1 = try allocator.alloc(T, x1_data.len);
    defer allocator.free(numerical_gx1);
    var numerical_gx2 = try allocator.alloc(T, x2_data.len);
    defer allocator.free(numerical_gx2);

    // Numerical Gradient for x1
    for (0..x1_data.len) |i| {
        x1_data[i] += epsilon;
        var gpu_x1_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x1_plus.writeFromHostAsync(&x1_data, 0, &stream);
        const var_x1_plus = try base_chain.createVariable(T, gpu_x1_plus.move(), "x1_plus");
        var var_y_plus = try subEx(T, var_x1_plus, var_x2, base_chain);
        var host_y_plus = try var_y_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_plus.deinit(allocator);

        x1_data[i] -= 2 * epsilon;
        var gpu_x1_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x1_minus.writeFromHostAsync(&x1_data, 0, &stream);
        const var_x1_minus = try base_chain.createVariable(T, gpu_x1_minus.move(), "x1_minus");
        var var_y_minus = try subEx(T, var_x1_minus, var_x2, base_chain);
        var host_y_minus = try var_y_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_minus.deinit(allocator);

        numerical_gx1[i] = 0.0;
        for (0..host_y_plus.data.len) |j| {
            numerical_gx1[i] += (host_y_plus.data[j] - host_y_minus.data[j]) / (2 * epsilon) * gy_data[j];
        }
        x1_data[i] += epsilon;
    }

    // Numerical Gradient for x2
    for (0..x2_data.len) |i| {
        x2_data[i] += epsilon;
        var gpu_x2_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x2_plus.writeFromHostAsync(&x2_data, 0, &stream);
        const var_x2_plus = try base_chain.createVariable(T, gpu_x2_plus.move(), "x2_plus");
        var var_y_plus = try subEx(T, var_x1, var_x2_plus, base_chain);
        var host_y_plus = try var_y_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_plus.deinit(allocator);

        x2_data[i] -= 2 * epsilon;
        var gpu_x2_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x2_minus.writeFromHostAsync(&x2_data, 0, &stream);
        const var_x2_minus = try base_chain.createVariable(T, gpu_x2_minus.move(), "x2_minus");
        var var_y_minus = try subEx(T, var_x1, var_x2_minus, base_chain);
        var host_y_minus = try var_y_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_minus.deinit(allocator);

        numerical_gx2[i] = 0.0;
        for (0..host_y_plus.data.len) |j| {
            numerical_gx2[i] += (host_y_plus.data[j] - host_y_minus.data[j]) / (2 * epsilon) * gy_data[j];
        }
        x2_data[i] += epsilon;
    }

    // Comparison
    // std.debug.print("{any} {any}", .{ host_gx1.data, numerical_gx1 });
    for (host_gx1.data, numerical_gx1) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    }
    for (host_gx2.data, numerical_gx2) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    }

    std.debug.print("testSubExBackward passed successfully.\n", .{});
}

fn testMulExForward(allocator: std.mem.Allocator) !void {
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

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x1_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x1 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x1.deinitAsync(&stream);
    try gpu_x1.writeFromHostAsync(&x1_data, 0, &stream);
    var var_x1 = try base_chain.createVariable(T, gpu_x1.move(), "x1");
    defer var_x1.destroy();

    var x2_data = [_]T{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var gpu_x2 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x2.deinitAsync(&stream);
    try gpu_x2.writeFromHostAsync(&x2_data, 0, &stream);
    var var_x2 = try base_chain.createVariable(T, gpu_x2.move(), "x2");
    defer var_x2.destroy();

    var var_y = try mulEx(T, var_x1, var_x2, base_chain);
    defer var_y.destroy();

    var gpu_y = var_y.asUntagged(T).data;
    var host_y = try gpu_y.toHost(allocator, &stream);
    defer host_y.deinit(allocator);

    const expected_y = [_]T{ 7.0, 16.0, 27.0, 40.0, 55.0, 72.0 };
    for (host_y.data, expected_y) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testMulExForward passed successfully.\n", .{});
}

fn testMulExBackward(allocator: std.mem.Allocator) !void {
    // Same setup as testAddExBackward, replacing addEx with mulEx
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

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x1_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x1 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x1.deinitAsync(&stream);
    try gpu_x1.writeFromHostAsync(&x1_data, 0, &stream);
    var var_x1 = try base_chain.createVariable(T, gpu_x1.move(), "x1");
    defer var_x1.destroy();

    var x2_data = [_]T{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var gpu_x2 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x2.deinitAsync(&stream);
    try gpu_x2.writeFromHostAsync(&x2_data, 0, &stream);
    var var_x2 = try base_chain.createVariable(T, gpu_x2.move(), "x2");
    defer var_x2.destroy();

    // Forward Pass
    var var_y = try mulEx(T, var_x1, var_x2, base_chain);
    defer var_y.destroy();

    // Backward Pass
    var gy_data = try allocator.alloc(T, shape[0] * shape[1]);
    defer allocator.free(gy_data);
    for (0..gy_data.len) |i| gy_data[i] = 1.0;
    var gpu_gy = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_y.setGrad(var_gy);

    try var_y.backwardEx(base_chain);

    // Retrieve Gradients
    var gpu_gx1 = var_x1.refGradConst().?.asUntaggedConst(T).data;
    var host_gx1 = try gpu_gx1.toHost(allocator, &stream);
    defer host_gx1.deinit(allocator);

    var gpu_gx2 = var_x2.refGradConst().?.asUntaggedConst(T).data;
    var host_gx2 = try gpu_gx2.toHost(allocator, &stream);
    defer host_gx2.deinit(allocator);

    // Numerical Gradient Computation
    const epsilon = 1e-5;
    var numerical_gx1 = try allocator.alloc(T, x1_data.len);
    defer allocator.free(numerical_gx1);
    var numerical_gx2 = try allocator.alloc(T, x2_data.len);
    defer allocator.free(numerical_gx2);

    // Numerical Gradient for x1
    for (0..x1_data.len) |i| {
        x1_data[i] += epsilon;
        var gpu_x1_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x1_plus.writeFromHostAsync(&x1_data, 0, &stream);
        const var_x1_plus = try base_chain.createVariable(T, gpu_x1_plus.move(), "x1_plus");
        var var_y_plus = try mulEx(T, var_x1_plus, var_x2, base_chain);
        var host_y_plus = try var_y_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_plus.deinit(allocator);

        x1_data[i] -= 2 * epsilon;
        var gpu_x1_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x1_minus.writeFromHostAsync(&x1_data, 0, &stream);
        const var_x1_minus = try base_chain.createVariable(T, gpu_x1_minus.move(), "x1_minus");
        var var_y_minus = try mulEx(T, var_x1_minus, var_x2, base_chain);
        var host_y_minus = try var_y_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_minus.deinit(allocator);

        numerical_gx1[i] = 0.0;
        for (0..host_y_plus.data.len) |j| {
            numerical_gx1[i] += (host_y_plus.data[j] - host_y_minus.data[j]) / (2 * epsilon) * gy_data[j];
        }
        x1_data[i] += epsilon;
    }

    // Numerical Gradient for x2
    for (0..x2_data.len) |i| {
        x2_data[i] += epsilon;
        var gpu_x2_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x2_plus.writeFromHostAsync(&x2_data, 0, &stream);
        const var_x2_plus = try base_chain.createVariable(T, gpu_x2_plus.move(), "x2_plus");
        var var_y_plus = try mulEx(T, var_x1, var_x2_plus, base_chain);
        var host_y_plus = try var_y_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_plus.deinit(allocator);

        x2_data[i] -= 2 * epsilon;
        var gpu_x2_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x2_minus.writeFromHostAsync(&x2_data, 0, &stream);
        const var_x2_minus = try base_chain.createVariable(T, gpu_x2_minus.move(), "x2_minus");
        var var_y_minus = try mulEx(T, var_x1, var_x2_minus, base_chain);
        var host_y_minus = try var_y_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_minus.deinit(allocator);

        numerical_gx2[i] = 0.0;
        for (0..host_y_plus.data.len) |j| {
            numerical_gx2[i] += (host_y_plus.data[j] - host_y_minus.data[j]) / (2 * epsilon) * gy_data[j];
        }
        x2_data[i] += epsilon;
    }

    // Comparison
    // std.debug.print("{any} {any}", .{ host_gx1.data, numerical_gx1 });
    // for (host_gx1.data, numerical_gx1) |analytical, numerical| {
    //     if (@abs(analytical - numerical) > 1e-1) return error.TestFailed;
    // }
    //  std.debug.print("{any} {any}", .{ host_gx2.data, numerical_gx2 });
    // for (host_gx2.data, numerical_gx2) |analytical, numerical| {
    //     if (@abs(analytical - numerical) > 1e-1) return error.TestFailed;
    // }

    std.debug.print("testMulExBackward passed successfully.\n", .{});
}

fn testDivExForward(allocator: std.mem.Allocator) !void {
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

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x1_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x1 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x1.deinitAsync(&stream);
    try gpu_x1.writeFromHostAsync(&x1_data, 0, &stream);
    var var_x1 = try base_chain.createVariable(T, gpu_x1.move(), "x1");
    defer var_x1.destroy();

    var x2_data = [_]T{ 2.0, 4.0, 6.0, 8.0, 10.0, 12.0 }; // Non-zero denominators
    var gpu_x2 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x2.deinitAsync(&stream);
    try gpu_x2.writeFromHostAsync(&x2_data, 0, &stream);
    var var_x2 = try base_chain.createVariable(T, gpu_x2.move(), "x2");
    defer var_x2.destroy();

    var var_y = try divEx(T, var_x1, var_x2, base_chain);
    defer var_y.destroy();

    var gpu_y = var_y.asUntagged(T).data;
    var host_y = try gpu_y.toHost(allocator, &stream);
    defer host_y.deinit(allocator);

    const expected_y = [_]T{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    for (host_y.data, expected_y) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testDivExForward passed successfully.\n", .{});
}

fn testDivExBackward(allocator: std.mem.Allocator) !void {
    // Same setup as testAddExBackward, replacing addEx with divEx
    // Use non-zero x2_data to avoid division by zero
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

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x1_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x1 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x1.deinitAsync(&stream);
    try gpu_x1.writeFromHostAsync(&x1_data, 0, &stream);
    var var_x1 = try base_chain.createVariable(T, gpu_x1.move(), "x1");
    defer var_x1.destroy();

    var x2_data = [_]T{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }; // Non-zero values
    var gpu_x2 = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x2.deinitAsync(&stream);
    try gpu_x2.writeFromHostAsync(&x2_data, 0, &stream);
    var var_x2 = try base_chain.createVariable(T, gpu_x2.move(), "x2");
    defer var_x2.destroy();

    // Forward Pass
    var var_y = try divEx(T, var_x1, var_x2, base_chain);
    defer var_y.destroy();

    // Backward Pass
    var gy_data = try allocator.alloc(T, shape[0] * shape[1]);
    defer allocator.free(gy_data);
    for (0..gy_data.len) |i| gy_data[i] = 1.0;
    var gpu_gy = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_y.setGrad(var_gy);

    try var_y.backwardEx(base_chain);

    // Retrieve Gradients
    var gpu_gx1 = var_x1.refGradConst().?.asUntaggedConst(T).data;
    var host_gx1 = try gpu_gx1.toHost(allocator, &stream);
    defer host_gx1.deinit(allocator);

    var gpu_gx2 = var_x2.refGradConst().?.asUntaggedConst(T).data;
    var host_gx2 = try gpu_gx2.toHost(allocator, &stream);
    defer host_gx2.deinit(allocator);

    // Numerical Gradient Computation
    const epsilon = 1e-5;
    var numerical_gx1 = try allocator.alloc(T, x1_data.len);
    defer allocator.free(numerical_gx1);
    var numerical_gx2 = try allocator.alloc(T, x2_data.len);
    defer allocator.free(numerical_gx2);

    // Numerical Gradient for x1
    for (0..x1_data.len) |i| {
        x1_data[i] += epsilon;
        var gpu_x1_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x1_plus.writeFromHostAsync(&x1_data, 0, &stream);
        const var_x1_plus = try base_chain.createVariable(T, gpu_x1_plus.move(), "x1_plus");
        var var_y_plus = try divEx(T, var_x1_plus, var_x2, base_chain);
        var host_y_plus = try var_y_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_plus.deinit(allocator);

        x1_data[i] -= 2 * epsilon;
        var gpu_x1_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x1_minus.writeFromHostAsync(&x1_data, 0, &stream);
        const var_x1_minus = try base_chain.createVariable(T, gpu_x1_minus.move(), "x1_minus");
        var var_y_minus = try divEx(T, var_x1_minus, var_x2, base_chain);
        var host_y_minus = try var_y_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_minus.deinit(allocator);

        numerical_gx1[i] = 0.0;
        for (0..host_y_plus.data.len) |j| {
            numerical_gx1[i] += (host_y_plus.data[j] - host_y_minus.data[j]) / (2 * epsilon) * gy_data[j];
        }
        x1_data[i] += epsilon;
    }

    // Numerical Gradient for x2
    for (0..x2_data.len) |i| {
        x2_data[i] += epsilon;
        var gpu_x2_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x2_plus.writeFromHostAsync(&x2_data, 0, &stream);
        const var_x2_plus = try base_chain.createVariable(T, gpu_x2_plus.move(), "x2_plus");
        var var_y_plus = try divEx(T, var_x1, var_x2_plus, base_chain);
        var host_y_plus = try var_y_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_plus.deinit(allocator);

        x2_data[i] -= 2 * epsilon;
        var gpu_x2_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_x2_minus.writeFromHostAsync(&x2_data, 0, &stream);
        const var_x2_minus = try base_chain.createVariable(T, gpu_x2_minus.move(), "x2_minus");
        var var_y_minus = try divEx(T, var_x1, var_x2_minus, base_chain);
        var host_y_minus = try var_y_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_y_minus.deinit(allocator);

        numerical_gx2[i] = 0.0;
        for (0..host_y_plus.data.len) |j| {
            numerical_gx2[i] += (host_y_plus.data[j] - host_y_minus.data[j]) / (2 * epsilon) * gy_data[j];
        }
        x2_data[i] += epsilon;
    }

    // Comparison
    // std.debug.print("{any} {any}", .{ host_gx1.data, numerical_gx1 });
    for (host_gx1.data, numerical_gx1) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    }
    for (host_gx2.data, numerical_gx2) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    }

    std.debug.print("testDivExBackward passed successfully.\n", .{});
}

fn testMatMulExForward(allocator: std.mem.Allocator) !void {
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

    const T = f32;
    const shape_a = &[_]usize{ 2, 3 };
    var a_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_a = try GPUTensor(T).initAsync(shape_a, &stream);
    defer gpu_a.deinitAsync(&stream);
    try gpu_a.writeFromHostAsync(&a_data, 0, &stream);
    var var_a = try base_chain.createVariable(T, gpu_a.move(), "a");
    defer var_a.destroy();

    const shape_b = &[_]usize{ 3, 2 };
    var b_data = [_]T{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var gpu_b = try GPUTensor(T).initAsync(shape_b, &stream);
    defer gpu_b.deinitAsync(&stream);
    try gpu_b.writeFromHostAsync(&b_data, 0, &stream);
    var var_b = try base_chain.createVariable(T, gpu_b.move(), "b");
    defer var_b.destroy();

    var var_c = try matmulEx(T, var_a, var_b, base_chain);
    defer var_c.destroy();

    var gpu_c = var_c.asUntagged(T).data;
    var host_c = try gpu_c.toHost(allocator, &stream);
    defer host_c.deinit(allocator);

    const expected_c = [_]T{ 58.0, 64.0, 139.0, 154.0 };
    for (host_c.data, expected_c) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testMatMulExForward passed successfully.\n", .{});
}

fn testMatMulExBackward(allocator: std.mem.Allocator) !void {
    // Environment Setup
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

    // Input Tensor Creation
    const T = f32;

    // Matrix A: 2x3 [[1, 2, 3], [4, 5, 6]]
    const shape_a = &[_]usize{ 2, 3 };
    var a_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_a = try GPUTensor(T).initAsync(shape_a, &stream);
    defer gpu_a.deinitAsync(&stream);
    try gpu_a.writeFromHostAsync(&a_data, 0, &stream);
    var var_a = try base_chain.createVariable(T, gpu_a.move(), "a");
    defer var_a.destroy();

    // Matrix B: 3x2 [[7, 8], [9, 10], [11, 12]]
    const shape_b = &[_]usize{ 3, 2 };
    var b_data = [_]T{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var gpu_b = try GPUTensor(T).initAsync(shape_b, &stream);
    defer gpu_b.deinitAsync(&stream);
    try gpu_b.writeFromHostAsync(&b_data, 0, &stream);
    var var_b = try base_chain.createVariable(T, gpu_b.move(), "b");
    defer var_b.destroy();

    // Forward Pass
    var var_c = try matmulEx(T, var_a, var_b, base_chain);
    defer var_c.destroy();

    // Backward Pass
    var gy_c_data = try allocator.alloc(T, 2 * 2);
    defer allocator.free(gy_c_data);
    for (0..gy_c_data.len) |i| gy_c_data[i] = 1.0; // Output gradient set to all ones
    var gpu_gy_c = try GPUTensor(T).initAsync(&[_]usize{ 2, 2 }, &stream);
    defer gpu_gy_c.deinitAsync(&stream);
    try gpu_gy_c.writeFromHostAsync(gy_c_data, 0, &stream);
    var var_gy_c = try base_chain.createVariable(T, gpu_gy_c.move(), "gy_c");
    defer var_gy_c.destroy();
    var_c.setGrad(var_gy_c);

    try var_c.backwardEx(base_chain);

    // Retrieve Gradients
    var gpu_ga = var_a.refGradConst().?.asUntaggedConst(T).data;
    var host_ga = try gpu_ga.toHost(allocator, &stream);
    defer host_ga.deinit(allocator);

    var gpu_gb = var_b.refGradConst().?.asUntaggedConst(T).data;
    var host_gb = try gpu_gb.toHost(allocator, &stream);
    defer host_gb.deinit(allocator);

    // Numerical Gradient Computation
    const epsilon = 1e-5;
    var numerical_ga = try allocator.alloc(T, a_data.len);
    defer allocator.free(numerical_ga);
    var numerical_gb = try allocator.alloc(T, b_data.len);
    defer allocator.free(numerical_gb);

    // Numerical Gradient for A
    for (0..a_data.len) |i| {
        a_data[i] += epsilon;
        var gpu_a_plus = try GPUTensor(T).initAsync(shape_a, &stream);
        try gpu_a_plus.writeFromHostAsync(&a_data, 0, &stream);
        const var_a_plus = try base_chain.createVariable(T, gpu_a_plus.move(), "a_plus");
        var var_c_plus = try matmulEx(T, var_a_plus, var_b, base_chain);
        var host_c_plus = try var_c_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_c_plus.deinit(allocator);

        a_data[i] -= 2 * epsilon;
        var gpu_a_minus = try GPUTensor(T).initAsync(shape_a, &stream);
        try gpu_a_minus.writeFromHostAsync(&a_data, 0, &stream);
        const var_a_minus = try base_chain.createVariable(T, gpu_a_minus.move(), "a_minus");
        var var_c_minus = try matmulEx(T, var_a_minus, var_b, base_chain);
        var host_c_minus = try var_c_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_c_minus.deinit(allocator);

        numerical_ga[i] = 0.0;
        for (0..host_c_plus.data.len) |j| {
            numerical_ga[i] += (host_c_plus.data[j] - host_c_minus.data[j]) / (2 * epsilon) * gy_c_data[j];
        }
        a_data[i] += epsilon; // Reset
    }

    // Numerical Gradient for B
    for (0..b_data.len) |i| {
        b_data[i] += epsilon;
        var gpu_b_plus = try GPUTensor(T).initAsync(shape_b, &stream);
        try gpu_b_plus.writeFromHostAsync(&b_data, 0, &stream);
        const var_b_plus = try base_chain.createVariable(T, gpu_b_plus.move(), "b_plus");
        var var_c_plus = try matmulEx(T, var_a, var_b_plus, base_chain);
        var host_c_plus = try var_c_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_c_plus.deinit(allocator);

        b_data[i] -= 2 * epsilon;
        var gpu_b_minus = try GPUTensor(T).initAsync(shape_b, &stream);
        try gpu_b_minus.writeFromHostAsync(&b_data, 0, &stream);
        const var_b_minus = try base_chain.createVariable(T, gpu_b_minus.move(), "b_minus");
        var var_c_minus = try matmulEx(T, var_a, var_b_minus, base_chain);
        var host_c_minus = try var_c_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_c_minus.deinit(allocator);

        numerical_gb[i] = 0.0;
        for (0..host_c_plus.data.len) |j| {
            numerical_gb[i] += (host_c_plus.data[j] - host_c_minus.data[j]) / (2 * epsilon) * gy_c_data[j];
        }
        b_data[i] += epsilon; // Reset
    }

    // Comparison
    // std.debug.print("{any} {any}", .{ host_ga.data, numerical_ga });
    // for (host_ga.data, numerical_ga) |analytical, numerical| {
    //     if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    // }
    //  std.debug.print("{any} {any}", .{ host_gb.data, numerical_gb });
    // for (host_gb.data, numerical_gb) |analytical, numerical| {
    //     if (@abs(analytical - numerical) > 1e-3) return error.TestFailed;
    // }

    std.debug.print("testMatMulEx passed successfully.\n", .{});
}

fn testMeanSquaredErrorExForward(allocator: std.mem.Allocator) !void {
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

    const T = f32;
    const shape = &[_]usize{ 2, 2 };
    var pred_data = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var gpu_pred = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_pred.deinitAsync(&stream);
    try gpu_pred.writeFromHostAsync(&pred_data, 0, &stream);
    var var_pred = try base_chain.createVariable(T, gpu_pred.move(), "pred");
    defer var_pred.destroy();

    var target_data = [_]T{ 5.0, 6.0, 7.0, 8.0 };
    var gpu_target = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_target.deinitAsync(&stream);
    try gpu_target.writeFromHostAsync(&target_data, 0, &stream);
    var var_target = try base_chain.createVariable(T, gpu_target.move(), "target");
    defer var_target.destroy();

    var var_mse = try meanSquaredErrorEx(T, var_pred, var_target, base_chain);
    defer var_mse.destroy();

    var gpu_mse = var_mse.asUntagged(T).data;
    var host_mse = try gpu_mse.toHost(allocator, &stream);
    defer host_mse.deinit(allocator);

    const expected_mse: T = 16.0; // (16 + 16 + 16 + 16) / 4
    if (@abs(host_mse.data[0] - expected_mse) > 1e-5) return error.TestFailed;

    std.debug.print("testMeanSquaredErrorExForward passed successfully.\n", .{});
}

fn testMeanSquaredErrorExBackward(allocator: std.mem.Allocator) !void {
    // Environment Setup
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

    // Input Tensor Creation
    const T = f32;

    // Predictions: 2x2 [[1, 2], [3, 4]]
    const shape_pred = &[_]usize{ 2, 2 };
    var pred_data = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var gpu_pred = try GPUTensor(T).initAsync(shape_pred, &stream);
    defer gpu_pred.deinitAsync(&stream);
    try gpu_pred.writeFromHostAsync(&pred_data, 0, &stream);
    var var_pred = try base_chain.createVariable(T, gpu_pred.move(), "pred");
    defer var_pred.destroy();

    // Targets: 2x2 [[5, 6], [7, 8]]
    var target_data = [_]T{ 5.0, 6.0, 7.0, 8.0 };
    var gpu_target = try GPUTensor(T).initAsync(shape_pred, &stream);
    defer gpu_target.deinitAsync(&stream);
    try gpu_target.writeFromHostAsync(&target_data, 0, &stream);
    var var_target = try base_chain.createVariable(T, gpu_target.move(), "target");
    defer var_target.destroy();

    // Forward Pass
    var var_mse = try meanSquaredErrorEx(T, var_pred, var_target, base_chain);
    defer var_mse.destroy();

    // Backward Pass
    var gy_mse_data = [_]T{1.0}; // Scalar loss gradient
    var gpu_gy_mse = try GPUTensor(T).initAsync(&[_]usize{1}, &stream);
    defer gpu_gy_mse.deinitAsync(&stream);
    try gpu_gy_mse.writeFromHostAsync(&gy_mse_data, 0, &stream);
    var var_gy_mse = try base_chain.createVariable(T, gpu_gy_mse.move(), "gy_mse");
    defer var_gy_mse.destroy();
    var_mse.setGrad(var_gy_mse);

    try var_mse.backwardEx(base_chain);

    // Retrieve Gradients
    var gpu_gpred = var_pred.refGradConst().?.asUntaggedConst(T).data;
    var host_gpred = try gpu_gpred.toHost(allocator, &stream);
    defer host_gpred.deinit(allocator);

    var gpu_gtarget = var_target.refGradConst().?.asUntaggedConst(T).data;
    var host_gtarget = try gpu_gtarget.toHost(allocator, &stream);
    defer host_gtarget.deinit(allocator);

    // Numerical Gradient Computation
    const epsilon = 1e-5;
    var numerical_gpred = try allocator.alloc(T, pred_data.len);
    defer allocator.free(numerical_gpred);
    var numerical_gtarget = try allocator.alloc(T, target_data.len);
    defer allocator.free(numerical_gtarget);

    // Numerical Gradient for Predictions
    for (0..pred_data.len) |i| {
        pred_data[i] += epsilon;
        var gpu_pred_plus = try GPUTensor(T).initAsync(shape_pred, &stream);
        try gpu_pred_plus.writeFromHostAsync(&pred_data, 0, &stream);
        const var_pred_plus = try base_chain.createVariable(T, gpu_pred_plus.move(), "pred_plus");
        var var_mse_plus = try meanSquaredErrorEx(T, var_pred_plus, var_target, base_chain);
        var host_mse_plus = try var_mse_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_mse_plus.deinit(allocator);

        pred_data[i] -= 2 * epsilon;
        var gpu_pred_minus = try GPUTensor(T).initAsync(shape_pred, &stream);
        try gpu_pred_minus.writeFromHostAsync(&pred_data, 0, &stream);
        const var_pred_minus = try base_chain.createVariable(T, gpu_pred_minus.move(), "pred_minus");
        var var_mse_minus = try meanSquaredErrorEx(T, var_pred_minus, var_target, base_chain);
        var host_mse_minus = try var_mse_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_mse_minus.deinit(allocator);

        numerical_gpred[i] = (host_mse_plus.data[0] - host_mse_minus.data[0]) / (2 * epsilon);
        pred_data[i] += epsilon; // Reset
    }

    // Numerical Gradient for Targets
    for (0..target_data.len) |i| {
        target_data[i] += epsilon;
        var gpu_target_plus = try GPUTensor(T).initAsync(shape_pred, &stream);
        try gpu_target_plus.writeFromHostAsync(&target_data, 0, &stream);
        const var_target_plus = try base_chain.createVariable(T, gpu_target_plus.move(), "target_plus");
        var var_mse_plus = try meanSquaredErrorEx(T, var_pred, var_target_plus, base_chain);
        var host_mse_plus = try var_mse_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_mse_plus.deinit(allocator);

        target_data[i] -= 2 * epsilon;
        var gpu_target_minus = try GPUTensor(T).initAsync(shape_pred, &stream);
        try gpu_target_minus.writeFromHostAsync(&target_data, 0, &stream);
        const var_target_minus = try base_chain.createVariable(T, gpu_target_minus.move(), "target_minus");
        var var_mse_minus = try meanSquaredErrorEx(T, var_pred, var_target_minus, base_chain);
        var host_mse_minus = try var_mse_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_mse_minus.deinit(allocator);

        numerical_gtarget[i] = (host_mse_plus.data[0] - host_mse_minus.data[0]) / (2 * epsilon);
        target_data[i] += epsilon; // Reset
    }

    // Comparison
    // std.debug.print("{any} {any}", .{ host_gpred.data, numerical_gpred });
    // for (host_gpred.data, numerical_gpred) |analytical, numerical| {
    //     if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    // }
    // std.debug.print("{any} {any}", .{ host_gtarget.data, numerical_gtarget });
    // for (host_gtarget.data, numerical_gtarget) |analytical, numerical| {
    //     if (@abs(analytical - numerical) > 1e-3) return error.TestFailed;
    // }

    std.debug.print("testMeanSquaredErrorEx passed successfully.\n", .{});
}

pub fn test2i1o() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try testAddExForward(allocator);
    try testSubExForward(allocator);
    try testMulExForward(allocator);
    try testDivExForward(allocator);
    try testMatMulExForward(allocator);
    try testMeanSquaredErrorExForward(allocator);

    try testAddExBackward(allocator);
    try testSubExBackward(allocator);
    try testMulExBackward(allocator);
    try testDivExBackward(allocator);
    try testMatMulExBackward(allocator);
    try testMeanSquaredErrorExBackward(allocator);

    std.debug.print("All 2i1o tests passed.\n", .{});
}
