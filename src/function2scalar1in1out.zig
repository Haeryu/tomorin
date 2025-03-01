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

const FuncDecorator1in1outBase = @import("function1in1out.zig").FuncDecorator1in1outBase;
const add = @import("function2in1out.zig").add;
const mul = @import("function2in1out.zig").mul;
const scale = @import("function1scalar1in1out.zig").scale;

pub fn FuncDecorator2Scalar1in1out(comptime Self: type) type {
    return struct {
        pub fn create(context: *Context, scalar1: Self.Scalar1, scalar2: Self.Scalar2) !FuncKey {
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
                .scalar1 = scalar1,
                .scalar2 = scalar2,
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
                in.grad = try add(Self.Out, in_grad, gx, context);
            } else {
                in.grad = gx;
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.self_key.context;
            const in = self.base.context.refVariable(self.in.?).asUntagged(Self.In);

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

            const scalar1 = try std.fmt.allocPrint(allocator, "{} [label=\"{s}\", color=aquamarine, style=filled, shape=circle]", .{
                @intFromPtr(&self.scalar1),
                @typeName(Self.Scalar1),
            });
            defer allocator.free(scalar1);

            const scalar2 = try std.fmt.allocPrint(allocator, "{} [label=\"{s}\", color=aquamarine, style=filled, shape=circle]", .{
                @intFromPtr(&self.scalar2),
                @typeName(Self.Scalar2),
            });
            defer allocator.free(scalar2);

            return try std.fmt.allocPrint(allocator,
                \\{} [label="{s}", color=lightblue, style=filled, shape=box]
                \\{s}
                \\{s}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\{s}
                \\{s}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                scalar1,
                scalar2,
                @intFromPtr(&self.scalar1),
                @intFromPtr(ctx),
                @intFromPtr(&self.scalar2),
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

fn makefunc(
    comptime F: type,
    x: VarKey,
    scalar1: F.Scalar1,
    scalar2: F.Scalar2,
) !VarKey {
    const funckey = try F.create(x.context, scalar1, scalar2);
    var out: [Function.max_out]?VarKey = .{null} ** Function.max_out;
    try x.context.refFunction(funckey).forward(&.{x}, out[0..1]);

    return out[0];
}

pub fn ScaleShift(comptime T: type) type {
    return struct {
        in: ?VarKey,
        out: ?VarKey,
        scalar1: T, // scale
        scalar2: T, // shift
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        const owns_in = false;
        const owns_out = false;

        pub usingnamespace FuncDecorator2Scalar1in1out(Self);

        const Self = ScaleShift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.scaleShift(self.scalar1, self.scalar2, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: VarKey) !VarKey {
            return try scale(T, gy, self.scalar1);
        }
    };
}

pub fn scaleShift(comptime T: type, x: VarKey, scal: T, shif: T) !VarKey {
    return try makefunc(ScaleShift(T), x, scal, shif);
}
