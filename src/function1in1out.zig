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

const add = @import("function2in1out.zig").add;
const scale = @import("function1scalar1in1out.zig").scale;
const shift = @import("function1scalar1in1out.zig").shift;
const mul = @import("function2in1out.zig").mul;
const div = @import("function2in1out.zig").div;
const broadcastTo = @import("function1shape1in1out.zig").broadcastTo;

// TODO: 1in1outBase -> 1in1scalar, 1in2scalar ...

pub fn FuncDecorator1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, chain: *Chain) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try context.registerFunction(
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
                chain,
            );

            self.* = .{
                .in = null,
                .out = null,
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

fn makefunc(comptime F: type, x: *TaggedVar, chain: *Chain) !*TaggedVar {
    const funckey = try F.create(x.getContext(), chain);

    return try makefunc1in1outBase(funckey, x);
}

pub fn Neg(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = false;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Neg(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            try y.scale(-1.0, context.stream);
            return y.move();
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !*TaggedVar {
            return try neg(T, gy);
        }
    };
}

pub fn neg(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Neg(T), x, x.getContext().current_chain.?);
}

pub fn Square(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const ref_in_at_back = true;

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Square(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            try y.product(x, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try mul(T, try scale(T, self.in.?, 2.0), gy);
        }
    };
}

pub fn square(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Square(T), x);
}

pub fn Exp(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Exp(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            try y.exp(x, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try mul(T, self.out.?, gy);
        }
    };
}

pub fn exp(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Exp(T), x);
}

pub fn Sin(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Sin(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            try y.sin(context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try mul(T, try cos(T, self.in.?), gy);
        }
    };
}

pub fn sin(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Sin(T), x);
}

pub fn Cos(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Cos(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            try y.cos(context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try mul(T, try scale(T, try sin(T, self.in.?), -1.0), gy);
        }
    };
}

pub fn cos(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Cos(T), x);
}

pub fn Tan(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Tan(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            try y.tan(context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try div(T, gy, try square(T, try cos(T, self.in.?)));
        }
    };
}

pub fn tan(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Tan(T), x);
}

pub fn Tanh(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Tanh(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.tanh(context.stream);
            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try mul(T, gy, try shift(T, try neg(T, try square(T, self.out.?)), 1.0));
        }
    };
}

pub fn tanh(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Tanh(T), x);
}

pub fn Transpose(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Transpose(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            var y = try x.transpose(context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !*TaggedVar {
            return try transpose(T, gy);
        }
    };
}

pub fn transpose(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Transpose(T), x, x.getContext().current_chain.?);
}

pub fn FuncDecoratorSum(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, axis: []const isize, chain: *Chain) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try context.registerFunction(
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
                chain,
            );

            self.* = .{
                .in = null,
                .out = null,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
                    .chain = chain,
                },
                .axis = axis,
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

pub fn Sum(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        x_shape: []const usize = &.{},
        axis: []const isize,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecoratorSum(Self);

        const Self = Sum(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            self.x_shape = x.base.getShapeConst();

            var y = try x.sum(context.allocator, self.axis, true, context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try broadcastTo(T, gy, self.x_shape);
        }
    };
}

pub fn sum(comptime T: type, x: *TaggedVar, axis: []const isize) !*TaggedVar {
    const funckey = try Sum(T).create(x.getContext(), axis, x.getContext().current_chain.?);

    return try makefunc1in1outBase(funckey, x);
}

pub fn FuncDecoratorSumTo(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, shape: []const usize, chain: *Chain) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try context.registerFunction(
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
                chain,
            );

            self.* = .{
                .in = null,
                .out = null,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
                    .chain = chain,
                },
                .shape = shape,
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

pub fn SumTo(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        shape: []const usize,
        x_shape: []const usize = &.{},
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecoratorSumTo(Self);

        const Self = SumTo(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            self.x_shape = x.base.getShapeConst();

            var y = try x.sumTo(context.allocator, self.shape, context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try broadcastTo(T, gy, self.x_shape);
        }
    };
}

pub fn sumTo(comptime T: type, x: *TaggedVar, shape: []const usize) !*TaggedVar {
    const funckey = try SumTo(T).create(x.getContext(), shape, x.getContext().current_chain.?);

    return try makefunc1in1outBase(funckey, x);
}

pub fn Sigmoid(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Sigmoid(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            //        # y = 1 / (1 + xp.exp(-x))
            // y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation

            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.sigmoid(context.stream);
            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const y = self.out.?;
            const one_minus_y = try shift(T, try neg(T, y), 1.0);
            const y_times_one_minus_y = try mul(T, y, one_minus_y);
            return try mul(T, gy, y_times_one_minus_y);
        }
    };
}

pub fn sigmoid(comptime T: type, x: *TaggedVar) !*TaggedVar {
    return try makefunc(Sigmoid(T), x, x.getContext().current_chain.?);
}
