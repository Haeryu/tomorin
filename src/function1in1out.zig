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
const scale = @import("function1scalar1in1out.zig").scale;
const mul = @import("function2in1out.zig").mul;

// TODO: 1in1outBase -> 1in1scalar, 1in2scalar ...
pub fn FuncDecorator1in1out(comptime Self: type) type {
    return struct {
        pub fn create(context: *Context) !FuncKey {
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
                .base = .{
                    .self_key = self_key,
                },
            };

            return self_key;
        }

        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;

            if (Self.owns_in) {
                context.releaseVariable(self.in.?);
            }
            if (Self.owns_out) {
                context.releaseVariable(self.out.?);
            }
            context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []const VarKey, out: []?VarKey) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;
            self.in = args[0];

            var in = context.acquireVariable(args[0]).asUntaggedConst(Self.In);
            defer {
                if (!Self.owns_in) {
                    context.releaseVariable(self.in.?);
                }
            }

            var y = try self.forward(&in.data);
            errdefer y.deinitAsync(context.stream);

            const var_y = try context.createVariable(Self.Out, y, null);
            defer context.releaseVariable(var_y);
            self.out = var_y;

            in = context.acquireVariable(args[0]).asUntaggedConst(Self.In);
            self.base.generation = in.generation;
            context.acquireVariable(var_y).asUntagged(Self.Out).setCreator(
                self.base.self_key,
                self.base.generation,
            );

            defer {
                if (!Self.owns_out) {
                    context.releaseVariable(self.out.?);
                }
            }

            out[0] = var_y;
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;

            const out = context.refVariable(self.out.?).asUntaggedConst(Self.Out);

            const gx = try self.backward(out.grad.?);

            const in = context.refVariable(self.in.?).asUntagged(Self.In);

            if (in.grad) |in_grad| {
                in.grad = try add(Self.Out, in_grad, gx);
            } else {
                in.grad = gx;
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;
            const in = context.refVariable(self.in.?).asUntagged(Self.In);

            if (in.creator) |creator| {
                const in_creator = context.refFunction(creator);
                if (!seen_set.contains(in_creator)) {
                    try seen_set.put(in_creator, {});
                    try queue.add(in_creator);
                }
            }
        }
    };
}

fn makefunc(comptime F: type, x: VarKey) !VarKey {
    const funckey = try F.create(x.context);
    var out: [Function.max_out]?VarKey = .{null} ** Function.max_out;
    try x.context.refFunction(funckey).forward(&.{x}, &out);
    return out[0].?;
}

pub fn Neg(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        base: FunctionBase,

        const In = T;
        const Out = T;

        const owns_in = false;
        const owns_out = false;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Neg(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);

            return try y.scale(-1.0, context.cuda_context, context.stream);
        }

        pub fn backward(_: *Self, gy: VarKey) !VarKey {
            return try neg(T, gy);
        }
    };
}

pub fn neg(
    comptime T: type,
    x: VarKey,
) !VarKey {
    return try makefunc(Neg(T), x);
}

pub fn Square(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        base: FunctionBase,

        const In = T;
        const Out = T;

        const owns_in = true;
        const owns_out = false;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Square(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            try y.product(x, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            return try mul(T, try scale(T, self.in.?, 2.0), gy);
        }
    };
}

pub fn square(
    comptime T: type,
    x: VarKey,
) !VarKey {
    return try makefunc(Square(T), x);
}

pub fn Exp(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        base: FunctionBase,

        const In = T;
        const Out = T;

        const owns_in = true;
        const owns_out = false;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Exp(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            try y.exp(x, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            return try mul(T, self.in.?, gy);
        }
    };
}

pub fn exp(comptime T: type, x: VarKey) !VarKey {
    return try makefunc(Exp(T), x);
}
