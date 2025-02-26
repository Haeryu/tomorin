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

pub fn FuncDecorator1in1out(comptime Self: type) type {
    return struct {
        pub fn create(allocator: std.mem.Allocator) !PFunction {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .in = null,
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
            if (self.in) |*in| {
                in.release(allocator);
                self.in = null;
            }
            if (self.out) |out| {
                var outmut = out;
                if (outmut.upgrade()) |strong| {
                    var strongmut = strong;
                    strongmut.release(allocator);
                } else {
                    outmut.release(allocator);
                }
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
            self.in = pxs_tagged[0].?.untagMove(Self.In);

            var y = try self.forward(&self.in.?.get().?.data, cuda_context, stream);
            errdefer y.deinitAsync(stream);

            var py = try Variable(Self.Out).create(allocator, y, stream);
            errdefer py.release(allocator);

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

            var gx = try self.backward(allocator, &pgy.get().?.grad.?.get().?.data, cuda_context, stream);
            errdefer gx.deinitAsync(stream);

            var pgx = try Variable(Self.In).create(allocator, gx.move(), stream);
            errdefer pgx.release(allocator);
            std.debug.assert(self.in.?.get().?.grad == null);
            self.in.?.get().?.grad = pgx.move();
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

// pub fn Exp(comptime T: type) type {
//     return struct {
//         in: ?PVariable(T) = null,
//         out: ?PVariableWeak(T) = null,
//         const In = T;
//         const Out = T;

//         pub usingnamespace FuncDecorator1in1out(Exp(T));

//         const Self = Exp(T);

//         pub fn forward(_: *Self, x: *const GPUTensor(T), _: *const CudaContext, stream: *const Stream) !GPUTensor(T) {
//             var y = try x.cloneAsync(stream);
//             errdefer y.deinitAsync(stream);

//             try y.exp(stream);

//             return y;
//         }

//         pub fn backward(self: *Self, _: std.mem.Allocator, gy: *const GPUTensor(T), _: *const CudaContext, stream: *const Stream) !GPUTensor(T) {
//             var gx = try self.in.?.get().?.data.cloneAsync(stream);
//             errdefer gx.deinitAsync(stream);

//             try gx.exp(stream);
//             try gx.product(gy, stream);

//             return gx;
//         }
//     };
// }

pub fn Neg(comptime T: type) type {
    return struct {
        in: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        const In = T;
        const Out = T;

        pub usingnamespace FuncDecorator1in1out(Neg(T));

        const Self = Neg(T);

        pub fn forward(_: *Self, x: *const GPUTensor(T), cuda_context: *const CudaContext, stream: *const Stream) !GPUTensor(T) {
            var y = try x.cloneAsync(stream);
            try y.scale(-1.0, cuda_context, stream);

            return y;
        }

        pub fn backward(
            _: *Self,
            allocator: std.mem.Allocator,
            gy: *const GPUTensor(T),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !GPUTensor(T) {
            return try neg(T, allocator, gy.move(), cuda_context, stream);
        }
    };
}

fn makefunc(
    comptime F: type,
    allocator: std.mem.Allocator,
    x: PVariable(F.In),
    cuda_context: *const CudaContext,
    stream: *const Stream,
) !PVariable(F.Out) {
    errdefer x.release(allocator);

    var self = try F.create(allocator);
    errdefer self.release(allocator);

    var tagged = [1]PVarTagged{PVarTagged.init(F.In1, x.move())};

    var y = (try self.forward(allocator, &tagged, cuda_context, stream))[0].?;

    return y.untagMove(F.Out);
}

pub fn neg(
    comptime T: type,
    allocator: std.mem.Allocator,
    x: PVariable(T),
    cuda_context: *const CudaContext,
    stream: *const Stream,
) !PVariable(T) {
    return try makefunc(Neg(T), allocator, x, cuda_context, stream);
}
