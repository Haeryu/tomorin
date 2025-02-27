const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;

const PVariable = @import("variable.zig").PVariable;
const PVarTagged = @import("variable.zig").PTaggedVar;
const Variable = @import("variable.zig").Variable;
const PVariableWeak = @import("variable.zig").PVariableWeak;

const PFunction = @import("function.zig").PFunction;
const Function = @import("function.zig").Function;

const add = @import("function2in1out.zig").add;
const mul = @import("function2in1out.zig").mul;

pub fn FuncDecorator1scalar1in1out(comptime Self: type) type {
    return struct {
        pub fn create(scalar: Self.Scalar, context: *const Context) !PFunction {
            const self = try context.function_allocator.create(Self);
            errdefer context.function_allocator.destroy(self);

            self.* = .{
                .in = null,
                .out = null,
                .scalar = scalar,
                .context = context,
            };

            const function: Function = .{
                .ptr = self,
                .vtable = &.{
                    .forward = &forwardDecorated,
                    .backward = &backwardDecorated,
                    .destroy = &destroy,
                    .add_inputs_creators = &addInputsCreators,
                },
                .generation = 0,
            };

            var pfn = try Rc(Function, Function.Destructor).create(
                context.function_allocator,
                function,
                .{},
            );
            defer pfn.release(context.function_allocator);

            return .{
                .pfn = pfn.move(),
            };
        }

        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.in) |*in| {
                in.release(self.context.variable_allocator);
                self.in = null;
            }
            if (self.out) |out| {
                var outmut = out;
                outmut.release(self.context.variable_allocator);
                self.out = null;
            }
            self.context.function_allocator.destroy(self);
        }

        pub fn forwardDecorated(
            ctx: *anyopaque,
            pcreator: PFunction,
            pxs_tagged: []PVarTagged,
        ) ![Function.max_args_out]?PVarTagged {
            const self: *Self = @ptrCast(@alignCast(ctx));
            std.debug.assert(self.in == null);
            self.in = pxs_tagged[0].untagMove(Self.In);
            std.debug.assert(self.context == self.in.?.get().?.context);

            var y = try self.forward(&self.in.?.get().?.data);
            defer y.deinitAsync(self.context.stream);

            var py = try Variable(Self.Out).create(
                y.move(),
                null,
                self.context,
            );
            defer py.release(self.context.variable_allocator);

            py.get().?.setCreator(pcreator.move());

            self.out = py.downgrade();

            return .{PVarTagged.init(Self.Out, py.move())} ++ .{null} ** (Function.max_args_out - 1);
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            var pgy = self.out.?.upgrade().?;
            defer {
                if (!self.context.enable_backprop_graph) {
                    pgy.get().?.cleargrad();
                }
                pgy.release(self.context.variable_allocator);
            }

            var gx = try self.backward(pgy.get().?.grad.?.clone());
            defer gx.release(self.context.variable_allocator);

            if (self.in.?.get().?.grad) |*grad| {
                //var mutgrad = grad;
                var new_grad = try add(Self.In, grad.move(), gx.move(), self.context);
                defer new_grad.release(self.context.variable_allocator);

                self.in.?.get().?.grad = new_grad.move();
            } else {
                self.in.?.get().?.grad = gx.move();
            }
        }

        pub fn addInputsCreators(
            ctx: *anyopaque,
            queue: *PFunction.Queue,
            seen_set: *std.AutoHashMap(*const Function, void),
        ) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.in.?.getConst().?.creator) |creator| {
                if (!seen_set.contains(creator.getConst().?)) {
                    var mut_creator = creator;
                    try queue.add(mut_creator.get().?);
                    try seen_set.put(creator.getConst().?, {});
                }
            }
        }
    };
}

fn makefunc(
    comptime F: type,
    x: PVariable(F.In),
    scalar: F.Scalar,
    context: *const Context,
) !PVariable(F.Out) {
    var mutx = x;

    var self = try F.create(scalar, context);
    errdefer self.release(context.function_allocator);

    var tagged = [1]PVarTagged{PVarTagged.init(F.In, mutx.move())};

    var y = (try self.forward(&tagged))[0].?;
    defer y.release(context.variable_allocator);

    return y.untagMove(F.Out);
}

pub fn Scale(comptime T: type) type {
    return struct {
        in: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        scalar: T,
        context: *const Context,
        const In = T;
        const Out = T;
        const Scalar = T;

        pub usingnamespace FuncDecorator1scalar1in1out(Scale(T));

        const Self = Scale(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            var y = try x.cloneAsync(self.context.stream);
            return try y.scale(self.scalar, self.context.cuda_context, self.context.stream);
        }

        pub fn backward(
            self: *Self,
            gy: PVariable(T),
        ) !PVariable(T) {
            var mutgy = gy;
            return try scale(T, mutgy.move(), self.scalar, self.context);
        }
    };
}

pub fn scale(
    comptime T: type,
    x: PVariable(T),
    scalar: T,
    context: *const Context,
) !PVariable(T) {
    var mutx = x;
    return try makefunc(Scale(T), mutx.move(), scalar, context);
}

pub fn Pow(comptime T: type) type {
    return struct {
        in: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        scalar: T,
        context: *const Context,
        const In = T;
        const Out = T;
        const Scalar = T;

        pub usingnamespace FuncDecorator1scalar1in1out(Pow(T));

        const Self = Pow(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            var y = try x.cloneAsync(self.context.stream);
            defer y.deinitAsync(self.context.stream);

            try y.powf(self.scalar, self.context.stream);

            return y.move();
        }

        pub fn backward(
            self: *Self,
            gy: PVariable(T),
        ) !PVariable(T) {
            var mutgy = gy;

            var x = self.in.?.clone();
            defer x.release(self.context.variable_allocator);

            var cx = try scale(T, x.move(), self.scalar, self.context);
            defer cx.release(self.context.variable_allocator);

            var cx_xminus1 = try pow(T, cx.move(), self.scalar - 1.0, self.context);
            defer cx_xminus1.release(self.context.variable_allocator);

            return try mul(T, cx_xminus1.move(), mutgy.move(), self.context);
        }
    };
}

pub fn pow(
    comptime T: type,
    x: PVariable(T),
    scalar: T,
    context: *const Context,
) !PVariable(T) {
    var mutx = x;
    return try makefunc(Pow(T), mutx.move(), scalar, context);
}
