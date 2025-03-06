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
const makefunc2in1outBase = @import("function.zig").makefunc2in1outBase;

const neg = @import("function1in1out.zig").neg;
const pow = @import("function1scalar1in1out.zig").pow;
const scale = @import("function1scalar1in1out.zig").scale;
const square = @import("function1in1out.zig").square;
const transpose = @import("function1in1out.zig").transpose;
const broadcastTo = @import("function1shape1in1out.zig").broadcastTo;

pub fn FuncDecorator2in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator2in1outBase(Self);

        pub fn create(context: *Context) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try context.registerFunction(.{
                .ptr = self,
                .vtable = &.{
                    .forward = &Base.forwardDecorated,
                    .backward = &Base.backwardDecorated,
                    .destroy = &Base.destroy,
                    .get_generation = &Base.getGeneration,
                    .enqueue = &Base.enqueue,
                    .get_dot_alloc = &getDotAlloc,
                },
            });

            self.* = .{
                .in1 = null,
                .in2 = null,
                .out = null,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
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
) !*TaggedVar {
    const funckey = try F.create(x1.getContext());

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

        pub const ref_in1_at_back = false;
        pub const ref_in2_at_back = false;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Add(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            if (x1.ptr == x2.ptr) {
                var new_x1 = try x1.cloneAsync(context.stream);
                defer new_x1.deinitAsync(context.stream);

                const y = try new_x1.add(x2, context.cuda_context, context.stream);

                return y;
            } else {
                const y = try x1.add(x2, context.cuda_context, context.stream);

                return y;
            }
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
    return try makefunc(Add(T), x1, x2);
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

        pub const ref_in1_at_back = false;
        pub const ref_in2_at_back = false;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Sub(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            if (x1 == x2) {
                var new_x1 = try x1.cloneAsync(context.stream);
                defer new_x1.deinitAsync(context.stream);

                const y = try new_x1.sub(x2, context.cuda_context, context.stream);

                return y;
            } else {
                const y = try x1.sub(x2, context.cuda_context, context.stream);

                return y;
            }
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            return .{ gy, try neg(Self.In2, gy) };
        }
    };
}

pub fn sub(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try makefunc(Sub(T), x1, x2);
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

        pub const ref_in1_at_back = true;
        pub const ref_in2_at_back = true;

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
            return .{ try mul(In1, gy, self.in2.?), try mul(In2, gy, self.in1.?) };
        }
    };
}

pub fn mul(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try makefunc(Mul(T), x1, x2);
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

        pub const ref_in1_at_back = true;
        pub const ref_in2_at_back = true;

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

            const x2_sq = try square(T, self.in2.?);
            const minus_x1 = try neg(T, self.in1.?);
            const denom = try div(T, minus_x1, x2_sq);
            const gx1 = try mul(T, gy, denom);

            return .{ gx0, gx1 };
        }
    };
}

pub fn div(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try makefunc(Div(T), x1, x2);
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

        pub const ref_in1_at_back = true;
        pub const ref_in2_at_back = true;

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
            var y = try x1.matmul(
                false,
                x2,
                false,
                null,
                false,
                tomo.c.CUBLAS_COMPUTE_32F,
                1.0,
                1.0,
                tomo.tensor.matmul_epilogue.Epilogue(void, void),
                .{},
                context.stream,
                context.cuda_context,
                T,
            );
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const gx = try matmul(T, gy, try transpose(T, self.in2.?));
            const gw = try matmul(T, try transpose(T, self.in1.?), gy);

            return .{ gx, gw };
        }
    };
}

pub fn matmul(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try makefunc(MatMul(T), x1, x2);
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

        pub const ref_in1_at_back = true;
        pub const ref_in2_at_back = true;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = MeanSquaredError(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            if (x1 == x2) {
                var new_x1 = try x1.cloneAsync(context.stream);
                defer new_x1.deinitAsync(context.stream);

                var diff = try new_x1.sub(x2, context.cuda_context, context.stream);
                defer diff.deinitAsync(context.stream);

                try diff.product(&diff, context.stream);

                var sum = try diff.sum(context.allocator, &.{}, true, context.stream);
                defer sum.deinitAsync(context.stream);

                return try sum.scale(1.0 / @as(T, @floatFromInt(diff.base.countElem())), context.cuda_context, context.stream);
            } else {
                var diff = try x1.sub(x2, context.cuda_context, context.stream);
                defer diff.deinitAsync(context.stream);

                try diff.product(&diff, context.stream);

                var sum = try diff.sum(context.allocator, &.{}, true, context.stream);
                defer sum.deinitAsync(context.stream);

                return try sum.scale(1.0 / @as(T, @floatFromInt(diff.base.countElem())), context.cuda_context, context.stream);
            }
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const diff = try sub(T, self.in1.?, self.in2.?);
            const gy_broad = try broadcastTo(T, gy, diff.asUntaggedConst(T).data.base.getShapeConst());
            const gy_diff = try mul(T, gy_broad, diff);
            const gx0 = try scale(T, gy_diff, 2.0 / @as(T, @floatFromInt(diff.len())));
            const gx1 = try neg(T, gx0);
            return .{ gx0, gx1 };
        }
    };
}

pub fn meanSquaredError(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try makefunc(MeanSquaredError(T), x1, x2);
}
