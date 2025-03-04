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
const FuncDecorator2in1outBase = @import("function.zig").FuncDecorator2in1outBase;
const makefunc1in2outBase = @import("function.zig").makefunc1in2outBase;

const neg = @import("function1in1out.zig").neg;
const pow = @import("function1scalar1in1out.zig").pow;
const square = @import("function1in1out.zig").square;

pub fn FuncDecorator2in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator2in1outBase(Self);

        pub fn create(context: *Context) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try context.registerFunction(.{
                .ptr = self,
                .vtable = &.{
                    .forward = &Base.forwardDecorated,
                    .backward = &Base.backwardDecorated,
                    .destroy = &Base.destroy,
                    .get_generation = &Base.getGeneration,
                    .enqueue = &Base.enqueue,
                    .get_dot_alloc = &getDotAlloc,
                },
            });

            self.* = .{
                .in1 = null,
                .in2 = null,
                .out = null,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
                },
            };

            return func_ptr;
        }

        pub fn getDotAlloc(ctx: *anyopaque, var_seen_set: *TaggedVar.SeenSet) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.context.allocator;

            const in1_contains = var_seen_set.contains(self.in1.?);
            const in1 = if (!in1_contains) try self.in1.?.getDotAlloc() else "";
            defer if (!in1_contains) allocator.free(in1);

            try var_seen_set.put(self.in1.?, {});

            const in2_contains = var_seen_set.contains(self.in2.?);
            const in2 = if (!in2_contains) try self.in2.?.getDotAlloc() else "";
            defer if (!in2_contains) allocator.free(in2);

            try var_seen_set.put(self.in2.?, {});

            const out_contains = var_seen_set.contains(self.out.?);
            const out = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out);

            try var_seen_set.put(self.out.?, {});

            return try std.fmt.allocPrint(self.base.context.allocator,
                \\{} [label="{s}", color=lightblue, style=filled, shape=box]
                \\{s}
                \\{s}
                \\{s}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                in1,
                in2,
                out,
                @intFromPtr(self.in1.?),
                @intFromPtr(ctx),
                @intFromPtr(self.in2.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(
    comptime F: type,
    x1: *TaggedVar,
    x2: *TaggedVar,
) !*TaggedVar {
    const funckey = try F.create(x1.getContext());

    return try makefunc1in2outBase(funckey, x1, x2);
}

pub fn Add(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub const ref_in1_at_back = false;
        pub const ref_in2_at_back = false;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Add(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            if (x1 == x2) {
                var new_x1 = try x1.cloneAsync(context.stream);
                defer new_x1.deinitAsync(context.stream);

                const y = try new_x1.add(x2, context.cuda_context, context.stream);

                return y;
            } else {
                const y = try x1.add(x2, context.cuda_context, context.stream);

                return y;
            }
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            return .{ gy, gy };
        }
    };
}

pub fn add(
    comptime T: type,
    x1: *TaggedVar,
    x2: *TaggedVar,
) !*TaggedVar {
    return try makefunc(Add(T), x1, x2);
}

pub fn Sub(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub const ref_in1_at_back = false;
        pub const ref_in2_at_back = false;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Sub(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            if (x1 == x2) {
                var new_x1 = try x1.cloneAsync(context.stream);
                defer new_x1.deinitAsync(context.stream);

                const y = try new_x1.sub(x2, context.cuda_context, context.stream);

                return y;
            } else {
                const y = try x1.sub(x2, context.cuda_context, context.stream);

                return y;
            }
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            return .{ gy, try neg(Self.In2, gy) };
        }
    };
}

pub fn sub(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try makefunc(Sub(T), x1, x2);
}

pub fn Mul(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub const ref_in1_at_back = true;
        pub const ref_in2_at_back = true;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Mul(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x1.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.product(x2, context.stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            return .{ try mul(In1, gy, self.in2.?), try mul(In2, gy, self.in1.?) };
        }
    };
}

pub fn mul(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try makefunc(Mul(T), x1, x2);
}

pub fn Div(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Out = T;

        pub const ref_in1_at_back = true;
        pub const ref_in2_at_back = true;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Div(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x1.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.divide(x2, context.stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const gx0 = try div(T, gy, self.in2.?);

            const x2_sq = try square(T, self.in2.?);
            const minus_x1 = try neg(T, self.in1.?);
            const denom = try div(T, minus_x1, x2_sq);
            const gx1 = try mul(T, gy, denom);

            return .{ gx0, gx1 };
        }
    };
}

pub fn div(comptime T: type, x1: *TaggedVar, x2: *TaggedVar) !*TaggedVar {
    return try makefunc(Div(T), x1, x2);
}
