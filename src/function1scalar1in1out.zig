const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;
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

            self.in.?.release();
            self.in = null;
            self.out.?.release();
            self.out = null;
            context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []*TaggedVar, out: []?*TaggedVar) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            const context = self.base.self_key.context;
            self.in = args[0];

            var y = try self.forward(&self.in.?.asUntaggedConst(Self.In).data);
            errdefer y.deinitAsync(context.stream);

            self.out = try context.createVariable(Self.Out, y.move(), null);

            self.base.generation = self.in.?.getGeneration();
            self.out.?.asUntagged(Self.Out).setCreator(
                self.base.self_key,
                self.base.generation,
            );

            out[0] = self.out.?;
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

        pub fn getDotAlloc(ctx: *anyopaque) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.self_key.context.allocator;
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

    var out: [Function.max_out]?*TaggedVar = .{null} ** Function.max_out;
    var in: [1]*TaggedVar = .{x};

    try x.getContext().refFunction(funckey).forward(&in, out[0..1]);

    return out[0].?;
}

pub fn Shift(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: T,
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        const ref_in_at_back = false;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Shift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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

        const Scalar = T;
        const In = T;
        const Out = T;

        const ref_in_at_back = false;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Scale(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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

        const Scalar = T;
        const In = T;
        const Out = T;

        const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Powf(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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

        const Scalar = i32;
        const In = T;
        const Out = T;

        const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Pow(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            try y.pow(self.scalar, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const x_cmin1 = try pow(T, self.in.?, self.scalar - 1);
            const c_x_cmin1 = try scale(T, x_cmin1, @floatFromInt(self.scalar));
            return try mul(T, c_x_cmin1, gy);
        }
    };
}

pub fn pow(comptime T: type, x: *TaggedVar, scalar: i32) !*TaggedVar {
    return try makefunc(Pow(T), x, scalar);
}
