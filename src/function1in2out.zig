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

const add = @import("function2in1out.zig").add;

pub fn FuncDecorator1in2out(comptime Self: type) type {
    return struct {
        pub fn create(allocator: std.mem.Allocator) !PFunction {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .in = null,
                .out1 = null,
                .out2 = null,
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
            if (self.out1) |out1| {
                var outmut1 = out1;
                if (outmut1.upgrade()) |strong1| {
                    var strongmut1 = strong1;
                    strongmut1.release(allocator);
                } else {
                    outmut1.release(allocator);
                }
                self.out1 = null;
            }
            if (self.out2) |out2| {
                var outmut2 = out2;
                if (outmut2.upgrade()) |strong2| {
                    var strongmut2 = strong2;
                    strongmut2.release(allocator);
                } else {
                    outmut2.release(allocator);
                }
                self.out2 = null;
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

            var y1, var y2 = try self.forward(&self.in.?.get().?.data, cuda_context, stream);
            errdefer y1.deinitAsync(stream);
            errdefer y2.deinitAsync(stream);

            var py1 = try Variable(Self.In).create(allocator, y1.move(), stream);
            errdefer py1.release(allocator);

            var py2 = try Variable(Self.In).create(allocator, y2.move(), stream);
            errdefer py2.release(allocator);

            py1.get().?.setCreator(pcreator.clone());
            py2.get().?.setCreator(pcreator.move());

            self.out1 = py1.downgrade();
            self.out2 = py2.downgrade();

            return .{ PVarTagged.init(Self.Out1, py1.move()), PVarTagged.init(Self.Out2, py2.move()) } ++ .{null} ** (Function.max_args_out - 2);
        }

        pub fn backwardDecorated(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            var gy1 = self.out1.?.upgrade().?.getConst().?.grad.?.clone();
            defer gy1.release(allocator);

            var gy2 = self.out1.?.upgrade().?.getConst().?.grad.?.clone();
            defer gy2.release(allocator);

            var gx = try self.backward(allocator, gy1.move(), gy2.move(), cuda_context, stream);
            errdefer gx.deinitAsync(stream);

            std.debug.assert(self.in.?.get().?.grad == null);
            self.in.?.get().?.grad = gx.move();
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
    allocator: std.mem.Allocator,
    x: PVariable(F.In),
    cuda_context: *const CudaContext,
    stream: *const Stream,
) !std.meta.Tuple(&.{ PVariable(F.Out1), PVariable(F.Out2) }) {
    var mutx = x;
    errdefer mutx.release(allocator);

    var self = try F.create(allocator);
    errdefer self.release(allocator);

    var tagged = [1]?PVarTagged{PVarTagged.init(F.In, mutx.move())};

    var ys = try self.forward(allocator, &tagged, cuda_context, stream);
    var y1 = ys[0].?.move();
    var y2 = ys[1].?.move();

    return .{ y1.untagMove(F.Out1), y2.untagMove(F.Out2) };
}

pub fn Clone(comptime T: type) type {
    return struct {
        in: ?PVariable(T) = null,
        out1: ?PVariableWeak(T) = null,
        out2: ?PVariableWeak(T) = null,

        const In = T;
        const Out1 = T;
        const Out2 = T;

        pub usingnamespace FuncDecorator1in2out(Clone(T));

        const Self = Clone(T);

        pub fn forward(
            _: *Self,
            x: *const GPUTensor(T),
            _: *const CudaContext,
            stream: *const Stream,
        ) !std.meta.Tuple(&.{ GPUTensor(T), GPUTensor(T) }) {
            var y1 = try x.cloneAsync(stream);
            errdefer y1.deinitAsync(stream);

            var y2 = try x.cloneAsync(stream);
            errdefer y2.deinitAsync(stream);

            return .{ y1, y2 };
        }

        pub fn backward(
            _: *Self,
            allocator: std.mem.Allocator,
            gy1: PVariable(T),
            gy2: PVariable(T),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !PVariable(T) {
            var gy1mut = gy1;
            var gy2mut = gy2;

            return try add(T, allocator, gy1mut.move(), gy2mut.move(), cuda_context, stream);
        }
    };
}

pub fn clone(
    comptime T: type,
    allocator: std.mem.Allocator,
    x: PVariable(T),
    cuda_context: *const CudaContext,
    stream: *const Stream,
) !std.meta.Tuple(&.{ PVariable(T), PVariable(T) }) {
    var xmut = x;
    return makefunc(Clone(T), allocator, xmut.move(), cuda_context, stream);
}
