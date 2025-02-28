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
const PTaggedVar = @import("variable.zig").PTaggedVar;
const Variable = @import("variable.zig").Variable;

const Function = @import("function.zig").Function;
const FunctionBase = @import("function.zig").FunctionBase;

const add = @import("function2in1out.zig").add;

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
                },
            });

            self.* = .{
                .in = null,
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

        pub fn forwardDecorated(ctx: *anyopaque, args: []VarKey) ![]VarKey {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in = args[0];

            const in = self.base.context.getVariable(args[0]).asUntaggedConst(Self.In);

            var y = try self.forward(&in.data);
            errdefer y.deinitAsync(self.base.context.stream);

            const var_y = try self.base.context.makeVariable(Self.Out, y, null);
            self.out = var_y;

            self.base.generation = in.generation;
            self.base.context.getVariable(var_y).asUntagged(Self.Out).setCreator(
                self.base.self_key,
                self.base.generation,
            );

            var vars = [1]VarKey{var_y};

            return &vars;
        }

        pub fn backwardDecorated(ctx: *anyopaque, args: []VarKey) ![]VarKey {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const gy = args[0];

            const gx = try self.backward(gy);

            var tagged_gx = [1]VarKey{gx};

            return &tagged_gx;
        }
    };
}

fn makefunc(
    comptime F: type,
    x: VarKey,
    context: *Context,
) !VarKey {
    const funckey = try F.create(context);

    var varkeys = [1]VarKey{x};

    return (try context.getFunction(funckey).forward(&varkeys))[0];
}

pub fn Neg(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        base: FunctionBase,

        const In = T;
        const Out = T;

        pub usingnamespace FuncDecorator1in1out(Neg(T));

        const Self = Neg(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            var y = try x.cloneAsync(self.base.context.stream);

            return try y.scale(-1.0, self.base.context.cuda_context, self.base.context.stream);
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            return try neg(T, gy, self.base.context);
        }
    };
}

pub fn neg(
    comptime T: type,
    x: VarKey,
    context: *Context,
) !VarKey {
    return try makefunc(Neg(T), x, context);
}
