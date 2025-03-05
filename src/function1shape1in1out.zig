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

pub fn FuncDecorator1Shape1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, shape: []const usize) !*Function {
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
                .shape = shape,
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

fn makefunc(comptime F: type, x: *TaggedVar, shape: []const usize) !*TaggedVar {
    const funckey = try F.create(x.getContext(), shape);

    return try makefunc1in1outBase(funckey, x);
}

pub fn Reshape(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        old_shape: []const usize = &.{},
        shape: []const usize,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Shape1in1out(Self);

        const Self = Reshape(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            self.old_shape = x.base.getShapeConst();

            var y = try GPUTensor(T).initAsync(self.shape, context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.writeAsync(x.ptr.?, x.calcLen(), 0, context.stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try reshape(T, gy, self.old_shape);
        }
    };
}

pub fn reshape(
    comptime T: type,
    x: *TaggedVar,
    shape: []const usize,
) !*TaggedVar {
    return try makefunc(Reshape(T), x, shape);
}

pub fn BroadCastTo(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        x_shape: []const usize = &.{},
        shape: []const usize,
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Shape1in1out(Self);

        const Self = BroadCastTo(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            self.x_shape = x.base.getShapeConst();

            var y = try x.broadcastTo(self.shape, context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try reshape(T, gy, self.x_shape);
        }
    };
}

pub fn broadcastTo(
    comptime T: type,
    x: *TaggedVar,
    shape: []const usize,
) !*TaggedVar {
    return try makefunc(BroadCastTo(T), x, shape);
}
