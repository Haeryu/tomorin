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

const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;

const Function = @import("function.zig").Function;
const FunctionBase = @import("function.zig").FunctionBase;

const add = @import("function2in1out.zig").add;
const mul = @import("function2in1out.zig").mul;

pub fn FuncDecorator1Scalar1in1out(comptime Self: type) type {
    return struct {
        pub fn create(context: *Context, scalar: Self.Scalar) !FuncKey {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const self_key = try context.registerFunction(.{
                .ptr = self,
                .vtable = &.{
                    .forward = &forwardDecorated,
                    .backward = &backwardDecorated,
                    .destroy = &destroy,
                    .get_generation = &getGeneration,
                    .enqueue = &enqueue,
                    .get_dot_alloc = &getDotAlloc,
                },
            });

            self.* = .{
                .in = null,
                .out = null,
                .scalar = scalar,
                .base = .{
                    .self_key = self_key,
                },
            };

            return self_key;
        }

        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;
            if (Self.owns_in) {
                context.releaseVariable(self.in.?);
            }
            if (Self.owns_out) {
                context.releaseVariable(self.out.?);
            }
            context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []const VarKey, out: []?VarKey) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;
            self.in = args[0];

            var in = context.acquireVariable(args[0]).asUntaggedConst(Self.In);
            defer {
                if (!Self.owns_in) {
                    context.releaseVariable(self.in.?);
                }
            }

            var y = try self.forward(&in.data);
            errdefer y.deinitAsync(context.stream);

            const var_y = try context.createVariable(Self.Out, y, null);
            defer context.releaseVariable(var_y);
            self.out = var_y;

            in = context.refVariable(args[0]).asUntaggedConst(Self.In);
            self.base.generation = in.generation;
            context.acquireVariable(var_y).asUntagged(Self.Out).setCreator(
                self.base.self_key,
                self.base.generation,
            );

            defer {
                if (!Self.owns_out) {
                    context.releaseVariable(self.out.?);
                }
            }

            out[0] = var_y;
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;

            const out = context.refVariable(self.out.?).asUntaggedConst(Self.Out);

            const gx = try self.backward(out.grad.?);

            const in = context.refVariable(self.in.?).asUntagged(Self.In);

            if (in.grad) |in_grad| {
                in.grad = try add(Self.Out, in_grad, gx);
            } else {
                in.grad = gx;
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;
            const in = context.refVariable(self.in.?).asUntagged(Self.In);

            if (in.creator) |creator| {
                const in_creator = context.refFunction(creator);
                if (!seen_set.contains(in_creator)) {
                    try seen_set.put(in_creator, {});
                    try queue.add(in_creator);
                }
            }
        }

        pub fn getDotAlloc(ctx: *anyopaque) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.self_key.context.allocator;
            const in = if (Self.owns_in) try self.in.?.ref().getDotAlloc() else "";
            defer if (Self.owns_in) allocator.free(in) else {};
            const out = if (Self.owns_out) try self.in.?.ref().getDotAlloc() else "";
            defer if (Self.owns_in) allocator.free(out) else {};

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
                \\{s}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                scalar,
                @intFromPtr(&self.scalar),
                @intFromPtr(ctx),
                @intFromPtr(self.in.?.refConst()),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?.refConst()),
                in,
                out,
            });
        }
    };
}

fn makefunc(comptime F: type, x: VarKey, scalar: F.Scalar) !VarKey {
    const funckey = try F.create(x.context, scalar);

    var out: [Function.max_out]?VarKey = .{null} ** Function.max_out;
    try x.context.refFunction(funckey).forward(&.{x}, out[0..1]);

    return out[0].?;
}

pub fn Shift(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar: T,
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        const owns_in = false;
        const owns_out = false;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Shift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.shift(self.scalar, context.stream);
            return y;
        }

        pub fn backward(_: *Self, gy: VarKey) !VarKey {
            return gy;
        }
    };
}

pub fn shift(comptime T: type, x: VarKey, scalar: T) !VarKey {
    return try makefunc(Shift(T), x, scalar);
}

pub fn Scale(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar: T,
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        const owns_in = false;
        const owns_out = false;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Scale(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            return try x.scale(self.scalar, context.cuda_context, context.stream);
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            return try scale(T, gy, self.scalar);
        }
    };
}

pub fn scale(comptime T: type, x: VarKey, scalar: T) !VarKey {
    return try makefunc(Scale(T), x, scalar);
}

pub fn Powf(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar: T,
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        const owns_in = true;
        const owns_out = false;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Powf(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            return try y.powf(self.scalar, context.cuda_context, context.stream);
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            const x_cmin1 = try powf(T, self.in.?, self.scalar - 1.0);
            const c_x_cmin1 = try scale(T, x_cmin1, self.scalar);
            return try mul(T, c_x_cmin1, gy);
        }
    };
}

pub fn powf(comptime T: type, x: VarKey, scalar: T) !VarKey {
    return try makefunc(Powf(T), x, scalar);
}

pub fn Pow(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar: i32,
        base: FunctionBase,

        const Scalar = i32;
        const In = T;
        const Out = T;

        const owns_in = true;
        const owns_out = false;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Pow(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            try y.pow(self.scalar, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            const x_cmin1 = try pow(T, self.in.?, self.scalar - 1);
            const c_x_cmin1 = try scale(T, x_cmin1, @floatFromInt(self.scalar));
            return try mul(T, c_x_cmin1, gy);
        }
    };
}

pub fn pow(comptime T: type, x: VarKey, scalar: i32) !VarKey {
    return try makefunc(Pow(T), x, scalar);
}
