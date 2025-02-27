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

// const clone = @import("function1in2out.zig").clone;
const neg = @import("function1in1out.zig").neg;

pub fn FuncDecorator2in1out(comptime Self: type) type {
    return struct {
        pub fn create(context: *const Context) !PFunction {
            const self = try context.function_allocator.create(Self);
            errdefer context.function_allocator.destroy(self);

            self.* = .{
                .in1 = null,
                .in2 = null,
                .out = null,
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

            if (self.in1) |*in1| {
                in1.release(self.context.variable_allocator);
                self.in1 = null;
            }
            if (self.in2) |*in2| {
                in2.release(self.context.variable_allocator);
                self.in2 = null;
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
            std.debug.assert(self.in1 == null);
            std.debug.assert(self.in2 == null);

            self.in1 = pxs_tagged[0].untagMove(Self.In1);
            self.in2 = pxs_tagged[1].untagMove(Self.In2);

            std.debug.assert(self.in1.?.get().?.context == self.context);
            std.debug.assert(self.in2.?.get().?.context == self.context);

            var y = try self.forward(&self.in1.?.get().?.data, &self.in2.?.get().?.data);
            errdefer y.deinitAsync(self.context.stream);

            var py = try Variable(Self.Out).create(
                y,
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

            var gx1, var gx2 = try self.backward(pgy.get().?.grad.?.clone());
            defer gx1.release(self.context.variable_allocator);
            defer gx2.release(self.context.variable_allocator);

            if (self.in1.?.get().?.grad) |*grad1| {
                //var mutgrad = grad;
                var new_grad = try add(Self.In1, grad1.move(), gx1.move(), self.context);
                defer new_grad.release(self.context.function_allocator);

                self.in1.?.get().?.grad = new_grad.move();
            } else {
                self.in1.?.get().?.grad = gx1.move();
            }

            if (self.in2.?.get().?.grad) |*grad2| {
                //var mutgrad = grad;
                var new_grad = try add(Self.In2, grad2.move(), gx2.move(), self.context);
                defer new_grad.release(self.context.function_allocator);

                self.in2.?.get().?.grad = new_grad.move();
            } else {
                self.in2.?.get().?.grad = gx2.move();
            }
        }

        pub fn addInputsCreators(
            ctx: *anyopaque,
            queue: *PFunction.Queue,
            seen_set: *std.AutoHashMap(*const Function, void),
        ) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.in1.?.getConst().?.creator) |creator1| {
                if (!seen_set.contains(creator1.getConst().?)) {
                    var mut_creator1 = creator1;
                    try queue.add(mut_creator1.get().?);
                    try seen_set.put(creator1.getConst().?, {});
                }
            }
            if (self.in2.?.getConst().?.creator) |creator2| {
                if (!seen_set.contains(creator2.getConst().?)) {
                    var mut_creator2 = creator2;
                    try queue.add(mut_creator2.get().?);
                    try seen_set.put(creator2.getConst().?, {});
                }
            }
        }
    };
}

fn makefunc(
    comptime F: type,
    x1: PVariable(F.In1),
    x2: PVariable(F.In2),
    context: *const Context,
) !PVariable(F.Out) {
    var mutx1 = x1;
    var mutx2 = x2;

    var self = try F.create(context);
    errdefer self.release(context.function_allocator);

    var tagged = [2]PVarTagged{ PVarTagged.init(F.In1, mutx1.move()), PVarTagged.init(F.In2, mutx2.move()) };

    var y = (try self.move().forward(&tagged))[0].?;
    defer y.release(context.variable_allocator);

    return y.untagMove(F.Out);
}

pub fn Add(comptime T: type) type {
    return struct {
        in1: ?PVariable(T) = null,
        in2: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        context: *const Context,

        const In1 = T;
        const In2 = T;
        const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Add(T);

        pub fn forward(
            self: *Self,
            x1: *const GPUTensor(T),
            x2: *const GPUTensor(T),
        ) !GPUTensor(T) {
            if (x1.ptr == x2.ptr) {
                var new_x2 = try x2.cloneAsync(self.context.stream);
                defer new_x2.deinitAsync(self.context.stream);
                return try x1.add(&new_x2, self.context.cuda_context, self.context.stream);
            } else {
                return try x1.add(x2, self.context.cuda_context, self.context.stream);
            }
        }

        pub fn backward(
            _: *Self,
            gy: PVariable(T),
        ) !std.meta.Tuple(&.{ PVariable(T), PVariable(T) }) {
            var gymut = gy;
            return .{ gymut.clone(), gymut.move() };
        }
    };
}

pub fn add(
    comptime T: type,
    x1: PVariable(T),
    x2: PVariable(T),
    context: *const Context,
) !PVariable(T) {
    var mutx1 = x1;
    var mutx2 = x2;
    return try makefunc(Add(T), mutx1.move(), mutx2.move(), context);
}

pub fn Sub(comptime T: type) type {
    return struct {
        in1: ?PVariable(T) = null,
        in2: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        context: *const Context,
        const In1 = T;
        const In2 = T;
        const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Sub(T);

        pub fn forward(
            self: *Self,
            x1: *const GPUTensor(T),
            x2: *const GPUTensor(T),
        ) !GPUTensor(T) {
            if (x1.ptr == x2.ptr) {
                var new_x2 = try x2.cloneAsync(self.context.stream);
                defer new_x2.deinitAsync(self.context.stream);
                return try x1.sub(&new_x2, self.context.cuda_context, self.context.stream);
            } else {
                return try x1.sub(x2, self.context.cuda_context, self.context.stream);
            }
        }

        pub fn backward(
            self: *Self,
            gy: PVariable(T),
        ) !std.meta.Tuple(&.{ PVariable(T), PVariable(T) }) {
            var mutgy = gy;

            var gy1 = mutgy.clone();
            defer gy1.release(self.context.variable_allocator);
            var gy2 = mutgy.move();
            defer gy2.release(self.context.variable_allocator);

            return .{ gy1.move(), try neg(T, gy2.move(), self.context) };
        }
    };
}

pub fn sub(
    comptime T: type,
    x1: PVariable(T),
    x2: PVariable(T),
    context: *const Context,
) !PVariable(T) {
    var mutx1 = x1;
    var mutx2 = x2;
    return try makefunc(Sub(T), mutx1.move(), mutx2.move(), context);
}

pub fn Mul(comptime T: type) type {
    return struct {
        in1: ?PVariable(T) = null,
        in2: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        context: *const Context,
        const In1 = T;
        const In2 = T;
        const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Mul(T);

        pub fn forward(
            self: *Self,
            x1: *const GPUTensor(T),
            x2: *const GPUTensor(T),
        ) !GPUTensor(T) {
            var y = try x1.cloneAsync(self.context.stream);
            defer y.deinitAsync(self.context.stream);
            try y.product(x2, self.context.stream);

            return y.move();
        }

        pub fn backward(
            self: *Self,
            gy: PVariable(T),
        ) !std.meta.Tuple(&.{ PVariable(T), PVariable(T) }) {
            var x1 = try self.in1.?.getConst().?.data.cloneAsync(self.context.stream);
            defer x1.deinitAsync(self.context.stream);

            var x2 = try self.in2.?.getConst().?.data.cloneAsync(self.context.stream);
            defer x2.deinitAsync(self.context.stream);

            var v1 = try Variable(Self.In1).create(
                x1.move(),
                null,
                self.context,
            );
            defer v1.release(self.context.variable_allocator);

            var v2 = try Variable(Self.In2).create(
                x2.move(),
                null,
                self.context,
            );
            defer v2.release(self.context.variable_allocator);

            var mutgy = gy;

            var gy1 = mutgy.clone();
            defer gy1.release(self.context.variable_allocator);
            var gy2 = mutgy.move();
            defer gy2.release(self.context.variable_allocator);

            var gx1 = try mul(T, gy1.clone(), v2.move(), self.context);
            defer gx1.release(self.context.variable_allocator);

            var gx2 = try mul(T, gy2.move(), v1.move(), self.context);
            defer gx2.release(self.context.variable_allocator);

            return .{ gx1.move(), gx2.move() };
        }
    };
}

pub fn mul(
    comptime T: type,
    x1: PVariable(T),
    x2: PVariable(T),
    context: *const Context,
) !PVariable(T) {
    var mutx1 = x1;
    var mutx2 = x2;
    return try makefunc(Mul(T), mutx1.move(), mutx2.move(), context);
}
