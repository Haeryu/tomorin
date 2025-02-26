const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;

const PVariable = @import("variable.zig").PVariable;
const PVarTagged = @import("variable.zig").PVarTagged;
const Variable = @import("variable.zig").Variable;
const PVariableWeak = @import("variable.zig").PVariableWeak;

const PFunction = @import("function.zig").PFunction;
const Function = @import("function.zig").Function;

// const clone = @import("function1in2out.zig").clone;
const neg = @import("function1in1out.zig").neg;

pub fn FuncDecorator2in1out(comptime Self: type) type {
    return struct {
        pub fn create(allocator: std.mem.Allocator) !PFunction {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .in1 = null,
                .in2 = null,
                .out = null,
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
                allocator,
                function,
                .{ .allocator = allocator },
            );
            defer pfn.release(allocator);

            return .{
                .pfn = pfn.move(),
            };
        }

        pub fn destroy(ctx: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.in1) |*in1| {
                in1.release(allocator);
                self.in1 = null;
            }
            if (self.in2) |*in2| {
                in2.release(allocator);
                self.in2 = null;
            }
            if (self.out) |out| {
                var outmut = out;
                // if (outmut.upgrade()) |strong| {
                //     var strongmut = strong;
                //     strongmut.release(allocator);
                // }
                outmut.release(allocator);
                self.out = null;
            }

            allocator.destroy(self);
        }

        pub fn forwardDecorated(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            pcreator: PFunction,
            pxs_tagged: []?PVarTagged,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) ![Function.max_args_out]?PVarTagged {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in1 = pxs_tagged[0].?.untagMove(Self.In1);
            self.in2 = pxs_tagged[1].?.untagMove(Self.In2);

            var y = try self.forward(&self.in1.?.get().?.data, &self.in2.?.get().?.data, cuda_context, stream);
            errdefer y.deinitAsync(stream);

            var py = try Variable(Self.Out).create(allocator, y.move(), stream);
            defer py.release(allocator);

            py.get().?.setCreator(pcreator.move());

            self.out = py.downgrade();

            return .{PVarTagged.init(Self.Out, py.move())} ++ .{null} ** (Function.max_args_out - 1);
        }

        pub fn backwardDecorated(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            var pgy = self.out.?.upgrade().?;
            defer pgy.release(allocator);

            var gx1, var gx2 = try self.backward(allocator, pgy.get().?.grad.?.clone(), cuda_context, stream);
            errdefer gx1.release(allocator);
            errdefer gx2.release(allocator);

            if (self.in1.?.get().?.grad) |*grad1| {
                //var mutgrad = grad;
                var new_grad = try add(Self.In1, allocator, grad1.move(), gx1.move(), cuda_context, stream);
                defer new_grad.release(allocator);

                self.in1.?.get().?.grad = new_grad.move();
            } else {
                self.in1.?.get().?.grad = gx1.move();
            }

            if (self.in2.?.get().?.grad) |*grad2| {
                //var mutgrad = grad;
                var new_grad = try add(Self.In2, allocator, grad2.move(), gx2.move(), cuda_context, stream);
                defer new_grad.release(allocator);

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
                    var mut_creator = creator1;
                    try queue.add(mut_creator.get().?);
                    try seen_set.put(creator1.getConst().?, {});
                }
            }
            if (self.in2.?.getConst().?.creator) |creator2| {
                if (!seen_set.contains(creator2.getConst().?)) {
                    var mut_creator = creator2;
                    try queue.add(mut_creator.get().?);
                    try seen_set.put(creator2.getConst().?, {});
                }
            }
        }
    };
}

fn makefunc(
    comptime F: type,
    allocator: std.mem.Allocator,
    x1: PVariable(F.In1),
    x2: PVariable(F.In2),
    cuda_context: *const CudaContext,
    stream: *const Stream,
) !PVariable(F.Out) {
    var mutx1 = x1;
    var mutx2 = x2;

    errdefer mutx1.release(allocator);
    errdefer mutx2.release(allocator);

    var self = try F.create(allocator);
    errdefer self.release(allocator);

    var tagged = [2]?PVarTagged{ PVarTagged.init(F.In1, mutx1.move()), PVarTagged.init(F.In2, mutx2.move()) };

    var y = (try self.move().forward(allocator, &tagged, cuda_context, stream))[0].?;
    defer y.release(allocator);

    return y.untagMove(F.Out);
}

pub fn Add(comptime T: type) type {
    return struct {
        in1: ?PVariable(T) = null,
        in2: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        const In1 = T;
        const In2 = T;
        const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Add(T);

        pub fn forward(
            _: *Self,
            x1: *const GPUTensor(T),
            x2: *const GPUTensor(T),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !GPUTensor(T) {
            if (x1.ptr == x2.ptr) {
                var new_x2 = try x2.cloneAsync(stream);
                defer new_x2.deinitAsync(stream);
                return try x1.add(&new_x2, cuda_context, stream);
            } else {
                return try x1.add(x2, cuda_context, stream);
            }
        }

        pub fn backward(
            _: *Self,
            _: std.mem.Allocator,
            gy: PVariable(T),
            _: *const CudaContext,
            _: *const Stream,
        ) !std.meta.Tuple(&.{ PVariable(T), PVariable(T) }) {
            var gymut = gy;
            //return try clone(T, allocator, gymut.move(), cuda_context, stream);
            return .{ gymut.clone(), gymut.move() };
        }
    };
}

pub fn add(
    comptime T: type,
    allocator: std.mem.Allocator,
    x1: PVariable(T),
    x2: PVariable(T),
    cuda_context: *const CudaContext,
    stream: *const Stream,
) !PVariable(T) {
    var mutx1 = x1;
    var mutx2 = x2;
    return try makefunc(Add(T), allocator, mutx1.move(), mutx2.move(), cuda_context, stream);
}

pub fn Sub(comptime T: type) type {
    return struct {
        in1: ?PVariable(T) = null,
        in2: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        const In1 = T;
        const In2 = T;
        const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Sub(T);

        pub fn forward(
            _: *Self,
            x1: *const GPUTensor(T),
            x2: *const GPUTensor(T),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !GPUTensor(T) {
            if (x1.ptr == x2.ptr) {
                var new_x2 = try x2.cloneAsync(stream);
                defer new_x2.deinitAsync(stream);
                return try x1.sub(&new_x2, cuda_context, stream);
            } else {
                return try x1.sub(x2, cuda_context, stream);
            }
        }

        pub fn backward(
            _: *Self,
            allocator: std.mem.Allocator,
            gy: PVariable(T),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !std.meta.Tuple(&.{ PVariable(T), PVariable(T) }) {
            var mutgy = gy;
            var gy1 = mutgy.clone();
            defer gy1.release(allocator);
            var gy2 = mutgy.move();
            defer gy2.release(allocator);

            return .{ gy1.move(), try neg(T, allocator, gy2.move(), cuda_context, stream) };
        }
    };
}

pub fn sub(
    comptime T: type,
    allocator: std.mem.Allocator,
    x1: PVariable(T),
    x2: PVariable(T),
    cuda_context: *const CudaContext,
    stream: *const Stream,
) !PVariable(T) {
    var mutx1 = x1;
    var mutx2 = x2;
    return try makefunc(Sub(T), allocator, mutx1.move(), mutx2.move(), cuda_context, stream);
}

pub fn Mul(comptime T: type) type {
    return struct {
        in1: ?PVariable(T) = null,
        in2: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        const In1 = T;
        const In2 = T;
        const Out = T;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Mul(T);

        pub fn forward(
            _: *Self,
            x1: *const GPUTensor(T),
            x2: *const GPUTensor(T),
            _: *const CudaContext,
            stream: *const Stream,
        ) !GPUTensor(T) {
            var y = try x1.cloneAsync(stream);
            errdefer y.deinitAsync(stream);
            try y.product(x2, stream);

            return y.move();
        }

        pub fn backward(
            self: *Self,
            allocator: std.mem.Allocator,
            gy: PVariable(T),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !std.meta.Tuple(&.{ PVariable(T), PVariable(T) }) {
            var x1 = try self.in1.?.getConst().?.data.cloneAsync(stream);
            errdefer x1.deinitAsync(stream);

            var x2 = try self.in2.?.getConst().?.data.cloneAsync(stream);
            errdefer x2.deinitAsync(stream);

            var v1 = try Variable(Self.In1).create(allocator, x1.move(), stream);
            defer v1.release(allocator);

            var v2 = try Variable(Self.In2).create(allocator, x2.move(), stream);
            defer v2.release(allocator);

            var mutgy = gy;
            var gy1 = mutgy.clone();
            defer gy1.release(allocator);
            var gy2 = mutgy.move();
            defer gy2.release(allocator);

            var gx1 = try mul(T, allocator, gy1.move(), v2.move(), cuda_context, stream);
            defer gx1.release(allocator);

            var gx2 = try mul(T, allocator, gy2.move(), v1.move(), cuda_context, stream);
            defer gx2.release(allocator);

            return .{ gx1.move(), gx2.move() };
        }
    };
}

pub fn mul(
    comptime T: type,
    allocator: std.mem.Allocator,
    x1: PVariable(T),
    x2: PVariable(T),
    cuda_context: *const CudaContext,
    stream: *const Stream,
) !PVariable(T) {
    var mutx1 = x1;
    var mutx2 = x2;
    return try makefunc(Mul(T), allocator, mutx1.move(), mutx2.move(), cuda_context, stream);
}
