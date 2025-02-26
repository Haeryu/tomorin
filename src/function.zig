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

const sliceCast = @import("util.zig").sliceCast;

pub const PFunction = struct {
    pfn: Rc(Function, Function.Destructor),

    pub const Queue = std.PriorityQueue(*Function, void, struct {
        fn comp(_: void, a: *Function, b: *Function) std.math.Order {
            return std.math.order(b.generation, a.generation);
        }
    }.comp);

    pub fn forward(
        self: *PFunction,
        allocator: std.mem.Allocator,
        pxs_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![Function.max_args_out]?PVarTagged {
        return try self.pfn.get().?.forward(allocator, self.move(), pxs_tagged, cuda_context, stream);
    }

    pub fn backward(
        self: *PFunction,
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !void {
        try self.pfn.get().?.backward(allocator, cuda_context, stream);
    }

    pub fn clone(self: PFunction) PFunction {
        return .{ .pfn = self.pfn.clone() };
    }

    pub fn move(self: PFunction) PFunction {
        var mutself = self;
        return .{ .pfn = mutself.pfn.move() };
    }

    pub fn get(self: *PFunction) ?*Function {
        return self.pfn.get();
    }

    pub fn getConst(self: *const PFunction) ?*const Function {
        return self.pfn.getConst();
    }

    pub fn release(self: *PFunction, allocator: std.mem.Allocator) void {
        self.pfn.release(allocator);
    }

    pub fn addInputsCreators(
        self: *const PFunction,
        queue: *Queue,
        seen_set: *std.AutoHashMap(PFunction, void),
    ) !void {
        try self.pfn.getConst().?.addInputsCreators(queue, seen_set);
    }
};

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    generation: usize,
    const max_args_out: comptime_int = 2;

    const VTable = struct {
        destroy: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
        forward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            pcreator: PFunction,
            pxs_tagged: []?PVarTagged,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror![max_args_out]?PVarTagged,

        backward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror!void,

        add_inputs_creators: *const fn (
            ctx: *anyopaque,
            queue: *PFunction.Queue,
            seen_set: *std.AutoHashMap(*const Function, void),
        ) anyerror!void,
    };

    pub fn destroy(self: *Function, allocator: std.mem.Allocator) void {
        self.vtable.destroy(self.ptr, allocator);
    }

    pub fn forward(
        self: *Function,
        allocator: std.mem.Allocator,
        pcreator: PFunction,
        pxs_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![max_args_out]?PVarTagged {
        for (pxs_tagged) |px_tagged| {
            if (px_tagged) |px_tagged_nonnull| {
                if (px_tagged_nonnull.getGeneration() > self.generation) {
                    self.generation = px_tagged_nonnull.getGeneration();
                }
            } else {
                break;
            }
        }
        return try self.vtable.forward(self.ptr, allocator, pcreator, pxs_tagged, cuda_context, stream);
    }

    pub fn backward(
        self: *Function,
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !void {
        try self.vtable.backward(self.ptr, allocator, cuda_context, stream);
    }

    pub fn addInputsCreators(
        self: *const Function,
        queue: *PFunction.Queue,
        seen_set: *std.AutoHashMap(*const Function, void),
    ) !void {
        try self.vtable.add_inputs_creators(self.ptr, queue, seen_set);
    }

    pub const Destructor = struct {
        allocator: std.mem.Allocator,

        pub fn destroy(self: *Destructor, function: *Function) void {
            function.destroy(self.allocator);
        }
    };
};

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
            }
            self.in = null;
            if (self.out) |*out| {
                out.release(allocator);
            }
            self.out = null;
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
            self.in = pxs_tagged[0].?.untagMove(Self.Elem);

            var y = try self.forward(&self.in.?.get().?.data, cuda_context, stream);
            errdefer y.deinitAsync(stream);

            var py = try Variable(Self.Elem).create(allocator, y, stream);
            errdefer py.release(allocator);

            py.get().?.setCreator(pcreator.move());

            self.out = py.downgrade();

            return .{PVarTagged.init(Self.Elem, py.move())} ++ .{null} ** (Function.max_args_out - 1);
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

            var gx = try self.backward(&pgy.get().?.grad.?.get().?.data, cuda_context, stream);
            errdefer gx.deinitAsync(stream);

            var pgx = try Variable(Self.Elem).create(allocator, gx.move(), stream);
            errdefer pgx.release(allocator);
            std.debug.assert(self.in.?.get().?.grad == null);
            self.in.?.get().?.grad = pgx.move();
        }

        pub fn getInputs(ctx: *anyopaque) [Function.max_args_in]?PVarTagged {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return .{self.in} ++ .{null} ** (Function.max_args_in - 1);
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

pub fn Exp(comptime T: type) type {
    return struct {
        in: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        const Elem = T;

        pub usingnamespace FuncDecorator1in1out(Exp(T));

        const Self = Exp(T);

        pub fn forward(_: *Self, x: *const GPUTensor(T), _: *const CudaContext, stream: *const Stream) !GPUTensor(T) {
            var y = try x.cloneAsync(stream);
            errdefer y.deinitAsync(stream);

            try y.exp(stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *const GPUTensor(T), _: *const CudaContext, stream: *const Stream) !GPUTensor(T) {
            var gx = try self.in.?.get().?.data.cloneAsync(stream);
            errdefer gx.deinitAsync(stream);

            try gx.exp(stream);
            try gx.product(gy, stream);

            return gx;
        }
    };
}
