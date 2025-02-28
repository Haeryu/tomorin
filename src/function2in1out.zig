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

pub fn FuncDecorator2in1out(comptime Self: type) type {
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
                .in1 = null,
                .in2 = null,
                .out = null,
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
            self.in1 = args[0];
            self.in2 = args[1];

            const in1 = self.base.context.getVariable(args[0]).asUntaggedConst(Self.In1);
            const in2 = self.base.context.getVariable(args[1]).asUntaggedConst(Self.In2);

            var y = try self.forward(&in1.data, &in2.data);
            errdefer y.deinitAsync(self.base.context.stream);

            const var_y = try self.base.context.createVariable(Self.Out, y, null);
            self.out = var_y;

            self.base.generation = @max(in1.generation, in2.generation);
            self.base.context.getVariable(var_y).asUntagged(Self.Out).setCreator(
                self.base.self_key,
                self.base.generation,
            );

            return &.{var_y};
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const out = self.base.context.getVariable(self.out.?).asUntaggedConst(Self.Out);

            const gx1, const gx2 = try self.backward(out.grad.?);

            const in1 = self.base.context.getVariable(self.in1.?).asUntagged(Self.In1);
            const in2 = self.base.context.getVariable(self.in2.?).asUntagged(Self.In2);
            if (in1.grad) |in1_grad| {
                in1.grad = try add(Self.Out, in1_grad, gx1, self.base.context);
            } else {
                in1.grad = gx1;
            }
            if (in2.grad) |in2_grad| {
                in2.grad = try add(Self.Out, in2_grad, gx2, self.base.context);
            } else {
                in2.grad = gx2;
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const in1 = self.base.context.getVariable(self.in1.?).asUntagged(Self.In1);

            if (in1.creator == null) return;

            const in1_creator = self.base.context.getFunction(in1.creator.?);
            if (!seen_set.contains(in1_creator)) {
                try seen_set.put(in1_creator, {});
                try queue.add(in1_creator);
            }

            const in2 = self.base.context.getVariable(self.in2.?).asUntagged(Self.In2);

            if (in2.creator == null) return;

            const in2_creator = self.base.context.getFunction(in2.creator.?);
            if (!seen_set.contains(in2_creator)) {
                try seen_set.put(in2_creator, {});
                try queue.add(in2_creator);
            }
        }
    };
}

fn makefunc(
    comptime F: type,
    x1: VarKey,
    x2: VarKey,
    context: *Context,
) !VarKey {
    const funckey = try F.create(context);

    return (try context.getFunction(funckey).forward(&.{ x1, x2 }))[0];
}

pub fn Add(comptime T: type) type {
    return struct {
        in1: ?VarKey,
        in2: ?VarKey,
        out: ?VarKey,
        base: FunctionBase,

        const In1 = T;
        const In2 = T;
        const Out = T;

        pub usingnamespace FuncDecorator2in1out(Add(T));

        const Self = Add(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            if (x1 == x2) {
                var new_x1 = try x1.cloneAsync(self.base.context.stream);
                defer new_x1.deinitAsync(self.base.context.stream);

                const y = try new_x1.add(x2, self.base.context.cuda_context, self.base.context.stream);

                return y;
            } else {
                const y = try x1.add(x2, self.base.context.cuda_context, self.base.context.stream);

                return y;
            }
        }

        pub fn backward(_: *Self, gy: VarKey) !std.meta.Tuple(&.{ VarKey, VarKey }) {
            return .{ gy, gy };
        }
    };
}

pub fn add(
    comptime T: type,
    x1: VarKey,
    x2: VarKey,
    context: *Context,
) !VarKey {
    return try makefunc(Add(T), x1, x2, context);
}
