const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;
const VarKey = @import("context.zig").VarKey;
const FuncKey = @import("context.zig").FuncKey;

const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;

const Function = @import("function.zig").Function;
const FunctionBase = @import("function.zig").FunctionBase;

const add = @import("function2in1out.zig").add;
const mul = @import("function2in1out.zig").mul;

pub fn FuncDecorator1Scalar1in1out(comptime Self: type) type {
    return struct {
        pub fn create(context: *Context, scalar: Self.Scalar) !FuncKey {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const self_key = try context.registerFunction(.{
                .ptr = self,
                .vtable = &.{
                    .forward = &forwardDecorated,
                    .backward = &backwardDecorated,
                    .destroy = &destroy,
                    .get_generation = &getGeneration,
                    .enqueue = &enqueue,
                },
            });

            self.* = .{
                .in = null,
                .out = null,
                .scalar = scalar,
                .base = .{
                    .self_key = self_key,
                    .context = context,
                },
            };

            return self_key;
        }

        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.base.context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []const VarKey) ![]const VarKey {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in = args[0];

            const in = self.base.context.getVariable(args[0]).asUntaggedConst(Self.In);

            var y = try self.forward(&in.data);
            errdefer y.deinitAsync(self.base.context.stream);

            const var_y = try self.base.context.createVariable(Self.Out, y, null);
            self.out = var_y;

            self.base.generation = in.generation;
            self.base.context.getVariable(var_y).asUntagged(Self.Out).setCreator(
                self.base.self_key,
                self.base.generation,
            );

            return &.{var_y};
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const out = self.base.context.getVariable(self.out.?).asUntaggedConst(Self.Out);

            const gx = try self.backward(out.grad.?);

            const in = self.base.context.getVariable(self.in.?).asUntagged(Self.In);

            if (in.grad) |in_grad| {
                in.grad = try add(Self.Out, in_grad, gx, self.base.context);
            } else {
                in.grad = gx;
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const in = self.base.context.getVariable(self.in.?).asUntagged(Self.In);

            if (in.creator) |creator| {
                const in_creator = self.base.context.getFunction(creator);
                if (!seen_set.contains(in_creator)) {
                    try seen_set.put(in_creator, {});
                    try queue.add(in_creator);
                }
            }
        }
    };
}

fn makefunc(
    comptime F: type,
    x: VarKey,
    scalar: F.Scalar,
    context: *Context,
) !VarKey {
    const funckey = try F.create(context, scalar);

    return (try context.getFunction(funckey).forward(&.{x}))[0];
}

pub fn Scale(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar: T,
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Scale(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            return try x.scale(self.scalar, self.base.context.cuda_context, self.base.context.stream);
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            return try scale(T, gy, self.scalar, self.base.context);
        }
    };
}

pub fn scale(
    comptime T: type,
    x: VarKey,
    scalar: T,
    context: *Context,
) !VarKey {
    return try makefunc(Scale(T), x, scalar, context);
}

pub fn Powf(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar: T,
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Powf(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            var y = try x.cloneAsync(self.base.context.stream);
            errdefer y.deinitAsync(self.base.context.stream);
            return try y.powf(self.scalar, self.base.context.cuda_context, self.base.context.stream);
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            const x_cmin1 = try powf(T, gy, self.scalar - 1.0, self.base.context);
            const c_x_cmin1 = try scale(T, x_cmin1, self.scalar, self.base.context);
            return try mul(T, c_x_cmin1, gy, self.base.context);
        }
    };
}

pub fn powf(
    comptime T: type,
    x: VarKey,
    scalar: T,
    context: *Context,
) !VarKey {
    return try makefunc(Powf(T), x, scalar, context);
}

pub fn Pow(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar: i32,
        base: FunctionBase,

        const Scalar = i32;
        const In = T;
        const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Pow(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            var y = try x.cloneAsync(self.base.context.stream);
            errdefer y.deinitAsync(self.base.context.stream);
            try y.pow(self.scalar, self.base.context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            const x_cmin1 = try pow(T, gy, self.scalar, self.base.context);
            const c_x_cmin1 = try scale(T, x_cmin1, @floatFromInt(self.scalar), self.base.context);
            return try mul(T, c_x_cmin1, gy, self.base.context);
        }
    };
}

pub fn pow(
    comptime T: type,
    x: VarKey,
    scalar: i32,
    context: *Context,
) !VarKey {
    return try makefunc(Pow(T), x, scalar, context);
}
