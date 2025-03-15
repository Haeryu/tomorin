const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;
const Chain = @import("chain.zig").Chain;

const Variable = @import("variable.zig").Variable;
const TaggedVar = @import("variable.zig").TaggedVar;

const sliceCast = @import("util.zig").sliceCast;
const add = @import("function2in1out.zig").add;

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    prev: ?*Function = null,
    next: ?*Function = null,
    chain: *Chain,

    pub const max_out = 3;

    pub const Queue = std.PriorityQueue(
        *Function,
        void,
        struct {
            fn comp(_: void, a: *Function, b: *Function) std.math.Order {
                return std.math.order(b.getGeneration(), a.getGeneration());
            }
        }.comp,
    );

    pub const SeenSet = std.AutoHashMap(*Function, void);

    const VTable = struct {
        destroy: *const fn (ctx: *anyopaque) void,

        forward: *const fn (ctx: *anyopaque, args: []*TaggedVar, out: []?*TaggedVar) anyerror!void,

        backward: *const fn (ctx: *anyopaque) anyerror!void,

        enqueue: *const fn (ctx: *anyopaque, queue: *Queue, seen_set: *SeenSet) anyerror!void,

        get_generation: *const fn (ctx: *anyopaque) usize,

        get_dot_alloc: *const fn (ctx: *anyopaque, var_seen_set: *TaggedVar.SeenSet) anyerror![]u8,
    };

    pub fn destroy(self: *Function) void {
        self.unchain();
        self.vtable.destroy(self.ptr);
    }

    pub fn forward(self: *Function, args: []*TaggedVar, out: []?*TaggedVar) !void {
        try self.vtable.forward(self.ptr, args, out);
    }

    pub fn backward(self: *Function) !void {
        return try self.vtable.backward(self.ptr);
    }

    pub fn enqueue(self: *Function, queue: *Queue, seen_set: *SeenSet) !void {
        try self.vtable.enqueue(self.ptr, queue, seen_set);
    }

    pub fn getGeneration(self: *Function) usize {
        return self.vtable.get_generation(self.ptr);
    }

    pub fn getDotAlloc(self: *const Function, var_seen_set: *TaggedVar.SeenSet) ![]u8 {
        return self.vtable.get_dot_alloc(self.ptr, var_seen_set);
    }

    pub fn unchain(self: *Function) void {
        if (self.prev) |prev| {
            prev.next = self.next;
        }
        if (self.next) |next| {
            next.prev = self.prev;
        }
        if (self.chain.func_chain == self) {
            self.chain.func_chain = self.next;
        }
        self.prev = null;
        self.next = null;
    }
};

pub const FunctionBase = struct {
    context: *Context,
    func_ptr: *Function,
    chain: *Chain,
    generation: usize = 0,
};

pub usingnamespace @import("function1in1out.zig");
pub usingnamespace @import("function1slice1in1out.zig");
pub usingnamespace @import("function2slice1in1out.zig");
// pub usingnamespace @import("function1in2out.zig");
pub usingnamespace @import("function2in1out.zig");
pub usingnamespace @import("function1scalar1in1out.zig");
pub usingnamespace @import("function2scalar1in1out.zig");
pub usingnamespace @import("function3in1out.zig");
pub usingnamespace @import("function1tensor1slice1in1out.zig");

// pub fn matyas(comptime T: type, x: VarKey, y: VarKey) !VarKey {
//     const x_y_sq = try @This().add(T, try @This().square(T, x), try @This().square(T, y));
//     const x_y_sq_sc = try @This().scale(T, x_y_sq, 0.26);

//     const xy = try @This().mul(T, x, y);
//     const xy_sc = try @This().scale(T, xy, 0.48);

//     const z = try @This().sub(T, x_y_sq_sc, xy_sc);

//     return z;
// }

pub fn matyas(comptime T: type, x: *TaggedVar, y: *TaggedVar) !*TaggedVar {
    const x_y_sq = try @This().add(T, try @This().square(T, x), try @This().square(T, y));
    const x_y_sq_sc = try @This().scale(T, x_y_sq, 0.26);

    const xy = try @This().mul(T, x, y);
    const xy_sc = try @This().scale(T, xy, 0.48);

    const z = try @This().sub(T, x_y_sq_sc, xy_sc);

    return z;
}

pub fn FuncDecorator1in1outBase(comptime Self: type) type {
    return struct {
        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.context;

            // if (self.in) |in| {
            //     // in.release();
            // }
            self.in = null;
            if (self.out) |out| {
                //out.release();
                out.resetCreator();
            }
            self.out = null;
            context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []*TaggedVar, out: []?*TaggedVar) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const context = self.base.context;

            self.in = args[0];

            var y = try self.forward(&self.in.?.asUntaggedConst(Self.In).data);
            errdefer y.deinitAsync(context.stream);

            self.out = try self.base.chain.createVariable(Self.Out, y.move(), null);

            self.base.generation = self.in.?.getGeneration();
            self.out.?.asUntagged(Self.Out).setCreator(
                self.base.func_ptr,
            );

            out[0] = self.out.?;

            // if (context.options.front_only) {
            //     self.in = null;
            //     self.out = null;
            // }
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const gx = try self.backward(self.out.?.asUntaggedConst(Self.Out).grad.?);

            if (self.in.?.asUntaggedConst(Self.Out).grad) |in_grad| {
                self.in.?.setGrad(try add(Self.Out, in_grad, gx));
            } else {
                self.in.?.setGrad(gx);
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.in.?.asUntaggedConst(Self.In).creator) |creator| {
                if (!seen_set.contains(creator)) {
                    try seen_set.put(creator, {});
                    try queue.add(creator);
                }
            }
        }
    };
}

pub fn makefunc1in1outBase(func_ptr: *Function, x: *TaggedVar) !*TaggedVar {
    var out: [Function.max_out]?*TaggedVar = .{null} ** Function.max_out;
    var in: [1]*TaggedVar = .{x};

    try func_ptr.forward(&in, out[0..1]);

    // if (x.getContextConst().options.front_only) {
    //     out[0].?.protect();
    //     defer out[0].?.unprotect();

    //     x.getContext().destroyFunctions();
    // }

    return out[0].?;
}

pub fn FuncDecorator2in1outBase(comptime Self: type) type {
    return struct {
        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.context;

            // if (self.in1) |in1| {
            //     // in1.release();
            // }
            self.in1 = null;
            // if (self.in2) |in2| {
            //     // in2.release();
            // }
            self.in2 = null;
            if (self.out) |out| {
                // out.release();
                out.resetCreator();
            }
            self.out = null;
            context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []*TaggedVar, out: []?*TaggedVar) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const context = self.base.context;

            self.in1 = args[0];

            self.in2 = args[1];

            var y = try self.forward(
                &self.in1.?.asUntaggedConst(Self.In1).data,
                &self.in2.?.asUntaggedConst(Self.In2).data,
            );
            errdefer y.deinitAsync(context.stream);

            self.out = try self.base.chain.createVariable(Self.Out, y.move(), null);

            self.base.generation = @max(self.in1.?.getGeneration(), self.in2.?.getGeneration());
            self.out.?.asUntagged(Self.Out).setCreator(
                self.base.func_ptr,
            );

            out[0] = self.out.?;

            // if (context.options.front_only) {
            //     // self.in = null;
            //     self.out = null;
            // }
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const gx1, const gx2 = try self.backward(self.out.?.asUntaggedConst(Self.Out).grad.?);

            if (self.in1.?.asUntaggedConst(Self.Out).grad) |in_grad1| {
                self.in1.?.setGrad(try add(Self.Out, in_grad1, gx1));
            } else {
                self.in1.?.setGrad(gx1);
            }
            if (self.in2.?.asUntaggedConst(Self.Out).grad) |in_grad2| {
                self.in2.?.setGrad(try add(Self.Out, in_grad2, gx2));
            } else {
                self.in2.?.setGrad(gx2);
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.in1.?.asUntaggedConst(Self.In1).creator) |creator1| {
                if (!seen_set.contains(creator1)) {
                    try seen_set.put(creator1, {});
                    try queue.add(creator1);
                }
            }

            if (self.in2.?.asUntaggedConst(Self.In2).creator) |creator2| {
                if (!seen_set.contains(creator2)) {
                    try seen_set.put(creator2, {});
                    try queue.add(creator2);
                }
            }
        }
    };
}

pub fn makefunc2in1outBase(func_ptr: *Function, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    std.debug.assert(x1.getContextConst() == x2.getContextConst());

    var out: [Function.max_out]?*TaggedVar = .{null} ** Function.max_out;

    var in: [2]*TaggedVar = .{ x1, x2 };

    try func_ptr.forward(&in, out[0..1]);

    return out[0].?;
}

pub fn FuncDecorator3in1outBase(comptime Self: type) type {
    return struct {
        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.context;

            // if (self.in1) |in1| {
            //     // in1.release();
            // }
            self.in1 = null;
            // if (self.in2) |in2| {
            //     // in2.release();
            // }
            self.in2 = null;
            self.in3 = null;
            if (self.out) |out| {
                // out.release();
                out.resetCreator();
            }
            self.out = null;
            context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []*TaggedVar, out: []?*TaggedVar) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const context = self.base.context;

            self.in1 = args[0];

            self.in2 = args[1];

            self.in3 = args[2];

            var y = try self.forward(
                &self.in1.?.asUntaggedConst(Self.In1).data,
                &self.in2.?.asUntaggedConst(Self.In2).data,
                &self.in3.?.asUntaggedConst(Self.In3).data,
            );
            errdefer y.deinitAsync(context.stream);

            self.out = try self.base.chain.createVariable(Self.Out, y.move(), null);

            self.base.generation = @max(self.in1.?.getGeneration(), self.in2.?.getGeneration(), self.in3.?.getGeneration());
            self.out.?.asUntagged(Self.Out).setCreator(
                self.base.func_ptr,
            );

            out[0] = self.out.?;

            // if (context.options.front_only) {
            //     // self.in = null;
            //     self.out = null;
            // }
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const gx1, const gx2, const gx3 = try self.backward(self.out.?.asUntaggedConst(Self.Out).grad.?);

            if (self.in1.?.asUntaggedConst(Self.Out).grad) |in_grad1| {
                self.in1.?.setGrad(try add(Self.Out, in_grad1, gx1));
            } else {
                self.in1.?.setGrad(gx1);
            }
            if (self.in2.?.asUntaggedConst(Self.Out).grad) |in_grad2| {
                self.in2.?.setGrad(try add(Self.Out, in_grad2, gx2));
            } else {
                self.in2.?.setGrad(gx2);
            }
            if (self.in3.?.asUntaggedConst(Self.Out).grad) |in_grad3| {
                self.in3.?.setGrad(try add(Self.Out, in_grad3, gx3));
            } else {
                self.in3.?.setGrad(gx3);
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.in1.?.asUntaggedConst(Self.In1).creator) |creator1| {
                if (!seen_set.contains(creator1)) {
                    try seen_set.put(creator1, {});
                    try queue.add(creator1);
                }
            }

            if (self.in2.?.asUntaggedConst(Self.In2).creator) |creator2| {
                if (!seen_set.contains(creator2)) {
                    try seen_set.put(creator2, {});
                    try queue.add(creator2);
                }
            }

            if (self.in3.?.asUntaggedConst(Self.In3).creator) |creator3| {
                if (!seen_set.contains(creator3)) {
                    try seen_set.put(creator3, {});
                    try queue.add(creator3);
                }
            }
        }
    };
}

pub fn makefunc3in1outBase(func_ptr: *Function, x1: *TaggedVar, x2: *TaggedVar, x3: *TaggedVar) !*TaggedVar {
    std.debug.assert(x1.getContextConst() == x2.getContextConst());
    std.debug.assert(x1.getContextConst() == x3.getContextConst());

    var out: [Function.max_out]?*TaggedVar = .{null} ** Function.max_out;

    var in: [3]*TaggedVar = .{ x1, x2, x3 };

    try func_ptr.forward(&in, out[0..1]);

    return out[0].?;
}

// test
pub fn testFunctions() !void {
    try @This().test1i1o();
    try @This().test1s1i1o();
    try @This().test1slice1i1o();
    try @This().test1tensor1slice1i1o();
    try @This().test2i1o();
    try @This().test2Scalar1i1o();
    try @This().test3i1o();
}
