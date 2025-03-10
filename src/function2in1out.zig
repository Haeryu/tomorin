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
const powEx = @import("function1scalar1in1out.zig").powEx;
const scaleEx = @import("function1scalar1in1out.zig").scaleEx;
const squareEx = @import("function1in1out.zig").squareEx;
const transposeEx = @import("function1in1out.zig").transposeEx;
const broadcastToEx = @import("function1shape1in1out.zig").broadcastToEx;

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
            var y = try x1.linear(x2, null, context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const gx = try matmulEx(T, gy, try transposeEx(T, self.in2.?, self.base.chain), self.base.chain);
            const gw = try matmulEx(T, try transposeEx(T, self.in1.?, self.base.chain), gy, self.base.chain);

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
