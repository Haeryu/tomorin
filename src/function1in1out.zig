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
const PTaggedVar = @import("variable.zig").PTaggedVar;
const Variable = @import("variable.zig").Variable;

const Function = @import("function.zig").Function;

const add = @import("function2in1out.zig").add;

pub fn FuncDecorator1in1out(comptime Self: type) type {
    return struct {
        pub fn create(context: *Context) !Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            self.* = .{
                // .in = null,
                // .out = null,
                .context = context,
            };

            return .{
                .ptr = self,
                .vtable = &.{
                    .forward = &forwardDecorated,
                    .backward = &backwardDecorated,
                    .destroy = &destroy,
                },
            };
        }

        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.context.allocator.destroy(self);
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []*TaggedVar) ![]*TaggedVar {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const in = args[0].asUntaggedConst(Self.In);

            var y = try self.forward(&in.data);
            errdefer y.deinitAsync(self.context.stream);

            const var_y: Variable(Self.Out) = .{
                .data = y,
                .level = self.context.getCurrentLevel(),
                .name = null,
                .context = self.context,
            };

            const tagged_y = TaggedVar.init(Self.Out, var_y);

            try self.context.pushTaggedVarAtCurrentLevelTop(tagged_y);

            var vars = [1]*TaggedVar{self.context.getCurrentLevelTopTaggedVar()};

            return &vars;
        }

        pub fn backwardDecorated(ctx: *anyopaque, args: []PTaggedVar) ![]PTaggedVar {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const gy = args[0].asUntaggedConst(Self.Out);

            const gx = try self.backward(gy);

            var tagged_gx = [1]PTaggedVar{PTaggedVar.init(Self.In, gx)};

            return &tagged_gx;
        }
    };
}

fn makefunc(
    comptime F: type,
    x: *const Variable(F.In),
) !*Variable(F.Out) {
    var func = try F.create(x.context);
    errdefer func.destroy();

    try x.context.pushFunctionAtCurrentLevelTop(func);

    var tagged_x = TaggedVar.init(F.In, x.*);
    var xs = [1]*TaggedVar{&tagged_x};

    const y = try func.forward(&xs);

    return y[0].asUntagged(F.Out);
}

pub fn Neg(comptime T: type) type {
    return struct {
        context: *Context,
        const In = T;
        const Out = T;

        pub usingnamespace FuncDecorator1in1out(Neg(T));

        const Self = Neg(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            var y = try x.cloneAsync(self.context.stream);

            return try y.scale(-1.0, self.context.cuda_context, self.context.stream);
        }

        pub fn backward(_: *Self, gy: *const Variable(T)) !*Variable(T) {
            return try neg(T, gy);
        }
    };
}

pub fn neg(
    comptime T: type,
    x: *const Variable(T),
) !*Variable(T) {
    return try makefunc(Neg(T), x);
}
