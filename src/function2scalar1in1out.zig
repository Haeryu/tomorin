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

const FuncDecorator1in1outBase = @import("function1in1out.zig").FuncDecorator1in1outBase;
const add = @import("function2in1out.zig").add;
const mul = @import("function2in1out.zig").mul;

pub fn FuncDecorator2Scalar1in1out(comptime Self: type) type {
    return struct {
        pub fn create(context: *Context, scalar1: Self.Scalar1, scalar2: Self.Scalar2) !FuncKey {
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
                .scalar1 = scalar1,
                .scalar2 = scalar2,
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
                in.grad = try add(Self.Out, in_grad, gx, context);
            } else {
                in.grad = gx;
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;
            const in = self.base.context.refVariable(self.in.?).asUntagged(Self.In);

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

fn makefunc(
    comptime F: type,
    x: VarKey,
    scalar1: F.Scalar1,
    scalar2: F.Scalar2,
) !VarKey {
    const funckey = try F.create(x.context, scalar1, scalar2);
    var out: [Function.max_out]?VarKey = .{null} ** Function.max_out;
    try x.context.refFunction(funckey).forward(&.{x}, &out);

    return out[0];
}

pub fn ScaleShift(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar1: T,
        scalar2: T,
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        const owns_in = false;
        const owns_out = false;

        pub usingnamespace FuncDecorator2Scalar1in1out(Self);

        const Self = ScaleShift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.scaleShift(self.scalar1, self.scalar2, context.stream);
            return y;
        }

        pub fn backward(_: *Self, gy: VarKey) !VarKey {
            return gy;
        }
    };
}

pub fn scaleShift(comptime T: type, x: VarKey, scale: T, shift: T) !VarKey {
    return try makefunc(ScaleShift(T), x, scale, shift);
}
