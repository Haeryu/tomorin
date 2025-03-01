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

            if (self.in) |in| {
                if (in.asUntaggedConst(Self.Out).grad) |in_grad| {
                    in.setGrad(try add(Self.Out, in_grad, gx));
                } else {
                    in.setGrad(gx);
                }
            }
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.in) |in| {
                if (in.asUntaggedConst(Self.In).creator) |creator| {
                    if (!seen_set.contains(creator)) {
                        try seen_set.put(creator, {});
                        try queue.add(creator);
                    }
                }
            }
        }

        pub fn getDotAlloc(ctx: *anyopaque) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.self_key.context.allocator;
            const in = try self.in.?.getDotAlloc();
            defer allocator.free(in);
            const out = try self.out.?.ref().getDotAlloc();
            defer allocator.free(out);

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
    x: *TaggedVar,
    scalar1: F.Scalar1,
    scalar2: F.Scalar2,
) !*TaggedVar {
    const funckey = try F.create(x.getContext(), scalar1, scalar2);
    var out: [Function.max_out]?*TaggedVar = .{null} ** Function.max_out;
    var in: [1]*TaggedVar = .{x};

    try x.getContext().refFunction(funckey).forward(&in, out[0..1]);

    return out[0];
}

pub fn ScaleShift(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar1: T, // scale
        scalar2: T, // shift
        base: FunctionBase,

        const Scalar = T;
        const In = T;
        const Out = T;

        pub usingnamespace FuncDecorator2Scalar1in1out(Self);

        const Self = ScaleShift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.self_key.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.scaleShift(self.scalar1, self.scalar2, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try scale(T, gy, self.scalar1);
        }
    };
}

pub fn scaleShift(comptime T: type, x: *TaggedVar, scal: T, shif: T) !*TaggedVar {
    return try makefunc(ScaleShift(T), x, scal, shif);
}
