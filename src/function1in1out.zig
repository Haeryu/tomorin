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
const scale = @import("function1scalar1in1out.zig").scale;
const mul = @import("function2in1out.zig").mul;

// TODO: 1in1outBase -> 1in1scalar, 1in2scalar ...
pub fn FuncDecorator1in1out(comptime Self: type) type {
    return struct {
        pub fn create(context: *Context) !FuncKey {
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

            return try std.fmt.allocPrint(allocator,
                \\{} [label="{s}", color=lightblue, style=filled, shape=box]
                \\{} -> {}
                \\{} -> {}
                \\{s}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                @intFromPtr(self.in.?.refConst()),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?.refConst()),
                in,
            });
        }
    };
}

fn makefunc(comptime F: type, x: *const TaggedVar) !*TaggedVar {
    const funckey = try F.create(x.getContext());
    var out: [Function.max_out]?*TaggedVar = .{null} ** Function.max_out;
    var in: [1]*TaggedVar = .{x};

    try x.getContext().refFunction(funckey).forward(&in, out[0..1]);
    return out[0].?;
}

pub fn Neg(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        const In = T;
        const Out = T;

        const ref_in_at_back = false;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Neg(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);

            return try y.scale(-1.0, context.cuda_context, context.stream);
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !*TaggedVar {
            return try neg(T, gy);
        }
    };
}

pub fn neg(
    comptime T: type,
    x: *TaggedVar,
) !*TaggedVar {
    return try makefunc(Neg(T), x);
}

pub fn Square(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        const ref_in_at_back = true;

        const In = T;
        const Out = T;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Square(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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

        const In = T;
        const Out = T;

        const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Exp(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            try y.exp(x, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try mul(T, self.in.?, gy);
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

        const In = T;
        const Out = T;

        const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Sin(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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

        const In = T;
        const Out = T;

        const ref_in_at_back = true;

        pub usingnamespace FuncDecorator1in1out(Self);

        const Self = Cos(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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
