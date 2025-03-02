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

const neg = @import("function1in1out.zig").neg;
const pow = @import("function1scalar1in1out.zig").pow;
const square = @import("function1in1out.zig").square;

pub fn FuncDecorator2in1out(comptime Self: type) type {
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
                .in1 = null,
                .in2 = null,
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

            self.in1.?.release();
            self.in1 = null;
            self.in2.?.release();
            self.in2 = null;
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

            self.in1 = args[0];

            self.in2 = args[1];

            var y = try self.forward(
                &self.in1.?.asUntaggedConst(Self.In1).data,
                &self.in2.?.asUntaggedConst(Self.In2).data,
            );
            errdefer y.deinitAsync(context.stream);

            self.out = try context.createVariable(Self.Out, y.move(), null);

            self.base.generation = @max(self.in1.?.getGeneration(), self.in2.?.getGeneration());
            self.out.?.asUntagged(Self.Out).setCreator(
                self.base.self_key,
                self.base.generation,
            );

            self.out.?.setBefore(self.in2);
            self.in2.?.setBefore(self.in1);

            out[0] = self.out.?;
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

            if (self.in1) |in1| {
                if (in1.asUntaggedConst(Self.In1).creator) |creator1| {
                    if (!seen_set.contains(creator1)) {
                        try seen_set.put(creator1, {});
                        try queue.add(creator1);
                    }
                }
            }

            if (self.in2) |in2| {
                if (in2.asUntaggedConst(Self.In2).creator) |creator2| {
                    if (!seen_set.contains(creator2)) {
                        try seen_set.put(creator2, {});
                        try queue.add(creator2);
                    }
                }
            }
        }

        pub fn getDotAlloc(ctx: *anyopaque) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.self_key.context.allocator;
            const in1 = if (Self.ref_in1_at_back) try self.in1.?.getDotAlloc() else "";
            defer if (Self.ref_in1_at_back) allocator.free(in1);
            const in2 = if (Self.ref_in2_at_back) try self.in2.?.getDotAlloc() else "";
            defer if (Self.ref_in2_at_back) allocator.free(in2);

            return try std.fmt.allocPrint(self.base.self_key.context.allocator,
                \\{} [label="{s}", color=lightblue, style=filled, shape=box]
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\{s}
                \\{s}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                @intFromPtr(self.in1.?),
                @intFromPtr(ctx),
                @intFromPtr(self.in2.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
                in1,
                in2,
            });
        }
    };
}

fn makefunc(
    comptime F: type,
    x1: *TaggedVar,
    x2: *TaggedVar,
) !*TaggedVar {
    std.debug.assert(x1.getContextConst() == x2.getContextConst());

    const funckey = try F.create(x1.getContext());

    var out: [Function.max_out]?*TaggedVar = .{null} ** Function.max_out;

    var in: [2]*TaggedVar = .{ x1, x2 };

    try x1.getContext().refFunction(funckey).forward(&in, out[0..1]);

    return out[0].?;
}

pub fn Add(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        const In1 = T;
        const In2 = T;
        const Out = T;

        const ref_in1_at_back = false;
        const ref_in2_at_back = false;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Add(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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

        const In1 = T;
        const In2 = T;
        const Out = T;

        const ref_in1_at_back = false;
        const ref_in2_at_back = false;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Sub(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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

        const In1 = T;
        const In2 = T;
        const Out = T;

        const ref_in1_at_back = true;
        const ref_in2_at_back = true;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Mul(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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

        const In1 = T;
        const In2 = T;
        const Out = T;

        const ref_in1_at_back = true;
        const ref_in2_at_back = true;

        pub usingnamespace FuncDecorator2in1out(Self);

        const Self = Div(T);

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
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
