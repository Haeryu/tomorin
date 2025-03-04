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
const makefunc1in1outBase = @import("function.zig").makefunc1in1outBase;

const add = @import("function2in1out.zig").add;
const mul = @import("function2in1out.zig").mul;

pub fn FuncDecorator1Scalar1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, scalar: Self.Scalar) !*Function {
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
                .in = null,
                .out = null,
                .scalar = scalar,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
                },
            };

            return func_ptr;
        }

        pub fn getDotAlloc(ctx: *anyopaque) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.context.allocator;
            const in = if (Self.ref_in_at_back) try self.in.?.getDotAlloc() else "";
            defer if (Self.ref_in_at_back) allocator.free(in);

            const scalar = try std.fmt.allocPrint(allocator, "{} [label=\"{s}\", color=aquamarine, style=filled, shape=circle]", .{
                @intFromPtr(&self.scalar),
                @typeName(Self.Scalar),
            });
            defer allocator.free(scalar);

            return try std.fmt.allocPrint(allocator,
                \\{} [label="{s}", color=lightblue, style=filled, shape=box]
                \\{s}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\{s}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                scalar,
                @intFromPtr(&self.scalar),
                @intFromPtr(ctx),
                @intFromPtr(self.in.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
                in,
            });
        }
    };
}

fn makefunc(comptime F: type, x: *TaggedVar, scalar: F.Scalar) !*TaggedVar {
    const funckey = try F.create(x.getContext(), scalar);

    return try makefunc1in1outBase(funckey, x);
}

pub fn Shift(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: T,
        base: FunctionBase,

        pub const Scalar = T;
        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = false;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Shift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.shift(self.scalar, context.stream);
            return y;
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !*TaggedVar {
            return gy;
        }
    };
}

pub fn shift(comptime T: type, x: *TaggedVar, scalar: T) !*TaggedVar {
    return try makefunc(Shift(T), x, scalar);
}

pub fn Scale(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: T,
        base: FunctionBase,

        pub const Scalar = T;
        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = false;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Scale(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            return try x.scale(self.scalar, context.cuda_context, context.stream);
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try scale(T, gy, self.scalar);
        }
    };
}

pub fn scale(comptime T: type, x: *TaggedVar, scalar: T) !*TaggedVar {
    return try makefunc(Scale(T), x, scalar);
}

pub fn Powf(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: T,
        base: FunctionBase,

        pub const Scalar = T;
        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Powf(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            return try y.powf(self.scalar, context.cuda_context, context.stream);
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const x_cmin1 = try powf(T, self.in.?, self.scalar - 1.0);
            const c_x_cmin1 = try scale(T, x_cmin1, self.scalar);
            return try mul(T, c_x_cmin1, gy);
        }
    };
}

pub fn powf(comptime T: type, x: *TaggedVar, scalar: T) !*TaggedVar {
    return try makefunc(Powf(T), x, scalar);
}

pub fn Pow(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: i32,
        base: FunctionBase,

        pub const Scalar = i32;
        pub const In = T;
        pub const Out = T;

        pub const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Pow(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            try y.pow(self.scalar, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const x_cmin1 = try pow(T, self.in.?, self.scalar - 1);
            const c_x_cmin1 = try scale(
                T,
                x_cmin1,
                if (T == BF16) BF16.fromF32(@floatFromInt(self.scalar)) else @floatFromInt(self.scalar),
            );
            return try mul(T, c_x_cmin1, gy);
        }
    };
}

pub fn pow(comptime T: type, x: *TaggedVar, scalar: i32) !*TaggedVar {
    return try makefunc(Pow(T), x, scalar);
}
