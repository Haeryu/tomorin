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
const Variable = @import("variable.zig").Variable;

const Function = @import("function.zig").Function;
const FunctionBase = @import("function.zig").FunctionBase;
const FuncDecorator1in1outBase = @import("function.zig").FuncDecorator1in1outBase;
const Chain = @import("chain.zig").Chain;
const makefunc1in1outBase = @import("function.zig").makefunc1in1outBase;

const sumToEx = @import("function1in1out.zig").sumToEx;
const expEx = @import("function1in1out.zig").expEx;
const sumEx = @import("function1in1out.zig").sumEx;
const mulEx = @import("function2in1out.zig").mulEx;
const subEx = @import("function2in1out.zig").subEx;
const getItemGradEx = @import("function2slice1in1out.zig").getItemGradEx;

pub fn FuncDecorator1Slice1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, slice: anytype, chain: *Chain) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try chain.registerFunction(
                .{
                    .ptr = self,
                    .vtable = &.{
                        .forward = &Base.forwardDecorated,
                        .backward = &Base.backwardDecorated,
                        .destroy = &Base.destroy,
                        .get_generation = &Base.getGeneration,
                        .enqueue = &Base.enqueue,
                        .get_dot_alloc = &getDotAlloc,
                    },
                    .chain = chain,
                },
            );

            self.* = .{
                .in = null,
                .out = null,
                .slice = slice,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
                    .chain = chain,
                },
            };

            return func_ptr;
        }

        pub fn getDotAlloc(ctx: *anyopaque, var_seen_set: *TaggedVar.SeenSet) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.context.allocator;

            const in_contains = var_seen_set.contains(self.in.?);
            const in = if (!in_contains) try self.in.?.getDotAlloc() else "";
            defer if (!in_contains) allocator.free(in);

            try var_seen_set.put(self.in.?, {});

            const out_contains = var_seen_set.contains(self.out.?);
            const out = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out);

            try var_seen_set.put(self.out.?, {});

            return try std.fmt.allocPrint(allocator,
                \\{} [label="{s}", color=lightblue, style=filled, shape=box]
                \\{s}
                \\{s}
                \\{} -> {}
                \\{} -> {}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                in,
                out,
                @intFromPtr(self.in.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(comptime F: type, x: *TaggedVar, slice: anytype, chain: *Chain) !*TaggedVar {
    const funckey = try F.create(x.getContext(), slice, chain);

    return try makefunc1in1outBase(funckey, x);
}

pub fn Reshape(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        old_shape: []const usize = &.{},
        slice: []const usize, // shape
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = Reshape(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            self.old_shape = x.base.getShapeConst();

            var y = try GPUTensor(T).initAsync(self.slice, context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.writeAsync(x.ptr.?, x.calcLen(), 0, context.stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try reshapeEx(T, gy, self.old_shape, self.base.chain);
        }
    };
}

pub fn reshape(comptime T: type, x: *TaggedVar, shape: []const usize) !*TaggedVar {
    return try makefunc(Reshape(T), x, shape, x.getContext().current_chain.?);
}

pub fn reshapeEx(comptime T: type, x: *TaggedVar, shape: []const usize, chain: *Chain) !*TaggedVar {
    return try makefunc(Reshape(T), x, shape, chain);
}

pub fn BroadCastTo(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        x_shape: []const usize = &.{},
        slice: []const usize, // shape
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = BroadCastTo(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            self.x_shape = x.base.getShapeConst();

            var y = try x.broadcastTo(self.slice, context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try sumToEx(T, gy, self.x_shape, self.base.chain);
        }
    };
}

pub fn broadcastTo(comptime T: type, x: *TaggedVar, shape: []const usize) !*TaggedVar {
    return try broadcastToEx(T, x, shape, x.getContext().current_chain.?);
}

pub fn broadcastToEx(comptime T: type, x: *TaggedVar, shape: []const usize, chain: *Chain) !*TaggedVar {
    return try makefunc(BroadCastTo(T), x, shape, chain);
}

pub fn Softmax(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        slice: ?[]const isize, // axis
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = Softmax(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            var max = try x.max(context.allocator, self.slice, true, context.stream);
            defer max.deinitAsync(context.stream);

            var max_brod = try max.broadcastTo(x.base.getShapeConst(), context.stream);
            defer max_brod.deinitAsync(context.stream);

            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.sub(&max_brod, context.stream);
            try y.exp(context.stream);

            var y_sum = try y.sum(context.allocator, self.slice, true, context.stream);
            defer y_sum.deinitAsync(context.stream);

            var y_sum_brod = try y_sum.broadcastTo(y.base.getShapeConst(), context.stream);
            defer y_sum_brod.deinitAsync(context.stream);

            try y.divide(&y_sum_brod, context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const gx = try mulEx(T, self.in.?, gy, self.base.chain);

            const sum_dx = try sumEx(T, gx, self.slice, self.base.chain);

            const sum_dx_broad = try broadcastToEx(T, sum_dx, self.in.?.getShape(), self.base.chain);

            const y_sumdx = try mulEx(T, self.in.?, sum_dx_broad, self.base.chain);

            return try subEx(T, gx, y_sumdx, self.base.chain);
        }
    };
}

pub fn softmax(comptime T: type, x: *TaggedVar, axis: []const isize) !*TaggedVar {
    return try softmaxEx(T, x, axis, x.getContext().current_chain.?);
}

pub fn softmaxEx(comptime T: type, x: *TaggedVar, shape: []const isize, chain: *Chain) !*TaggedVar {
    return try makefunc(Softmax(T), x, shape, chain);
}

pub fn LogSoftmax(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        slice: ?[]const isize, // reduction axes
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = LogSoftmax(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            var max = try x.max(context.allocator, self.slice, true, context.stream);
            defer max.deinitAsync(context.stream);
            var max_brod = try max.broadcastTo(x.base.getShapeConst(), context.stream);
            defer max_brod.deinitAsync(context.stream);

            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            try y.sub(&max_brod, context.stream);

            var exp_val = try y.cloneAsync(context.stream);
            defer exp_val.deinitAsync(context.stream);
            try exp_val.exp(context.stream);

            var log_sum_exp = try exp_val.sum(context.allocator, self.slice, true, context.stream);
            defer log_sum_exp.deinitAsync(context.stream);
            try log_sum_exp.log(context.stream);

            var log_sum_exp_brod = try log_sum_exp.broadcastTo(x.base.getShapeConst(), context.stream);
            defer log_sum_exp_brod.deinitAsync(context.stream);

            try y.sub(&log_sum_exp_brod, context.stream);
            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const exp_y = try expEx(T, self.out.?, self.base.chain);
            const sum_gy = try sumEx(T, gy, self.slice, self.base.chain);
            const sum_gy_broad = try broadcastToEx(T, sum_gy, self.out.?.getShape(), self.base.chain);
            return try subEx(T, gy, try mulEx(T, exp_y, sum_gy_broad, self.base.chain), self.base.chain);
        }
    };
}

pub fn logSoftmax(comptime T: type, x: *TaggedVar, axis: []const isize) !*TaggedVar {
    return try logSoftmaxEx(T, x, axis, x.getContext().current_chain.?);
}

pub fn logSoftmaxEx(comptime T: type, x: *TaggedVar, shape: []const isize, chain: *Chain) !*TaggedVar {
    return try makefunc(LogSoftmax(T), x, shape, chain);
}

pub fn GetItem(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        slice: []const GPUTensor(T).Slice, // reduction axes
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = GetItem(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            return try x.getItem(context.allocator, self.slice, context.stream);
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try getItemGradEx(T, gy, self.in.?.getShape(), self.base.chain);
        }
    };
}

pub fn getItem(comptime T: type, x: *TaggedVar, slice: []const GPUTensor(T).Slice) !*TaggedVar {
    return try logSoftmaxEx(T, x, slice, x.getContext().current_chain.?);
}

pub fn getItemEx(comptime T: type, x: *TaggedVar, slice: []const GPUTensor(T).Slice, chain: *Chain) !*TaggedVar {
    return try makefunc(GetItem(T), x, slice, chain);
}
