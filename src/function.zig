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

const Variable = @import("variable.zig").Variable;

const sliceCast = @import("util.zig").sliceCast;

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

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

        forward: *const fn (ctx: *anyopaque, args: []const VarKey, out: []?VarKey) anyerror!void,

        backward: *const fn (ctx: *anyopaque) anyerror!void,

        enqueue: *const fn (ctx: *anyopaque, queue: *Queue, seen_set: *SeenSet) anyerror!void,

        get_generation: *const fn (ctx: *anyopaque) usize,

        get_dot_alloc: *const fn (ctx: *anyopaque) anyerror![]u8,
    };

    pub fn destroy(self: *Function) void {
        self.vtable.destroy(self.ptr);
    }

    pub fn forward(self: *Function, args: []const VarKey, out: []?VarKey) !void {
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

    pub fn getDotAlloc(self: *const Function) ![]u8 {
        return self.vtable.get_dot_alloc(self.ptr);
    }
};

pub const FunctionBase = struct {
    self_key: FuncKey,
    generation: usize = 0,
};

pub usingnamespace @import("function1in1out.zig");
// pub usingnamespace @import("function1in2out.zig");
pub usingnamespace @import("function2in1out.zig");
pub usingnamespace @import("function1scalar1in1out.zig");
pub usingnamespace @import("function2scalar1in1out.zig");

// pub fn matyas(comptime T: type, x: VarKey, y: VarKey) !VarKey {
//     const x_y_sq = try @This().add(T, try @This().square(T, x), try @This().square(T, y));
//     const x_y_sq_sc = try @This().scale(T, x_y_sq, 0.26);

//     const xy = try @This().mul(T, x, y);
//     const xy_sc = try @This().scale(T, xy, 0.48);

//     const z = try @This().sub(T, x_y_sq_sc, xy_sc);

//     return z;
// }

pub fn matyas(comptime T: type, x: VarKey, y: VarKey) !VarKey {
    const x_y_sq = try @This().add(T, try @This().square(T, x), try @This().square(T, y));
    const x_y_sq_sc = try @This().scale(T, x_y_sq, 0.26);

    const xy = try @This().mul(T, x, y);
    const xy_sc = try @This().scale(T, xy, 0.48);

    const z = try @This().sub(T, x_y_sq_sc, xy_sc);

    return z;
}
