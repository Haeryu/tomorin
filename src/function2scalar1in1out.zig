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

const addEx = @import("function2in1out.zig").addEx;
const mulEx = @import("function2in1out.zig").mulEx;
const scaleEx = @import("function1scalar1in1out.zig").scaleEx;

pub fn FuncDecorator2Scalar1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, scalar1: Self.Scalar1, scalar2: Self.Scalar2, chain: *Chain) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try chain.registerFunction(
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
                context.current_chain,
                chain,
            );

            self.* = .{
                .in = null,
                .out = null,
                .scalar1 = scalar1,
                .scalar2 = scalar2,
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
                \\{s}
                \\{s}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                scalar1,
                scalar2,
                in,
                out,
                @intFromPtr(&self.scalar1),
                @intFromPtr(ctx),
                @intFromPtr(&self.scalar2),
                @intFromPtr(ctx),
                @intFromPtr(self.in.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(
    comptime F: type,
    x: *TaggedVar,
    scalar1: F.Scalar1,
    scalar2: F.Scalar2,
    chain: *Chain,
) !*TaggedVar {
    const funckey = try F.create(x.getContext(), scalar1, scalar2, chain);
    return try makefunc1in1outBase(funckey, x);
}

pub fn ScaleShift(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar1: T, // scale
        scalar2: T, // shift
        base: FunctionBase,

        pub const Scalar1 = T;
        pub const Scalar2 = T;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2Scalar1in1out(Self);

        const Self = ScaleShift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.scaleShift(self.scalar1, self.scalar2, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try scaleEx(T, gy, self.scalar1, self.base.chain);
        }
    };
}

pub fn scaleShift(comptime T: type, x: *TaggedVar, scal: T, shif: T) !*TaggedVar {
    return try scaleShiftEx(T, x, scal, shif, x.getContext().current_chain.?);
}

pub fn scaleShiftEx(comptime T: type, x: *TaggedVar, scal: T, shif: T, chain: *Chain) !*TaggedVar {
    return try makefunc(ScaleShift(T), x, scal, shif, chain);
}
