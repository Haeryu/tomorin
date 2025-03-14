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
const expEx = @import("function1in1out.zig").expEx;
const scaleEx = @import("function1scalar1in1out.zig").scaleEx;
const shiftEx = @import("function1scalar1in1out.zig").shiftEx;
const mulEx = @import("function2in1out.zig").mulEx;
const divEx = @import("function2in1out.zig").divEx;
const subEx = @import("function2in1out.zig").subEx;
const sumEx = @import("function1in1out.zig").sumEx;
const broadcastToEx = @import("function1slice1in1out.zig").broadcastToEx;
const logSoftmaxEx = @import("function1slice1in1out.zig").logSoftmaxEx;
const softmaxEx = @import("function1slice1in1out.zig").softmaxEx;

const dbg = @import("util.zig").debugPrintGpuTensor;

// TODO: 1in1outBase -> 1in1scalar, 1in2scalar ...

pub fn FuncDecorator1tensor1slice1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, t: *TaggedVar, slice: anytype, chain: *Chain) !*Function {
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
            );

            self.* = .{
                .in = null,
                .out = null,
                .t = t,
                .slice = slice,
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

            const t_contains = var_seen_set.contains(self.t);
            const t = if (!t_contains) try self.t.getDotAlloc() else "";
            defer if (!t_contains) allocator.free(t);

            try var_seen_set.put(self.out.?, {});

            return try std.fmt.allocPrint(allocator,
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
                in,
                out,
                t,
                @intFromPtr(self.in.?),
                @intFromPtr(ctx),
                @intFromPtr(self.t),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(comptime F: type, x: *TaggedVar, t: *TaggedVar, slice: anytype, chain: *Chain) !*TaggedVar {
    const funckey = try F.create(x.getContext(), t, slice, chain);

    return try makefunc1in1outBase(funckey, x);
}

pub fn SoftmaxCrossEntropy(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        t: *TaggedVar,
        slice: ?[]const isize, // softmax axis
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1tensor1slice1in1out(Self);

        const Self = SoftmaxCrossEntropy(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            const batch_size: T = @floatFromInt(x.base.getShapeConst()[0]);

            var x_clone = try x.cloneAsync(context.stream);
            defer x_clone.deinitAsync(context.stream);

            const x_var = try self.base.chain.createVariable(T, x_clone.move(), null);
            defer x_var.destroy();

            var logsm = try logSoftmaxEx(T, x_var, self.slice.?, self.base.chain);
            defer logsm.destroy();

            var prod = try mulEx(T, logsm, self.t, self.base.chain);
            defer prod.destroy();

            //try dbg(T, &logsm.asUntagged(T).data, context);
            //try dbg(T, &prod.asUntagged(T).data, context);

            var loss = try sumEx(T, prod, null, self.base.chain);
            defer loss.destroy();

            var scale = try scaleEx(T, loss, -1.0 / batch_size, self.base.chain);
            // var scale = try scaleEx(T, loss, 1.0 / batch_size, self.base.chain);
            defer scale.destroy();

            return scale.asUntagged(T).data.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const batch_size = self.in.?.getShape()[0];

            const y = try softmaxEx(T, self.in.?, self.slice.?, self.base.chain);

            const y_min_onehot = try subEx(T, y, self.t, self.base.chain);

            const gy_scaled = try scaleEx(
                T,
                gy,
                1.0 / @as(T, @floatFromInt(batch_size)),
                self.base.chain,
            );

            return try mulEx(
                T,
                try broadcastToEx(T, gy_scaled, y_min_onehot.getShape(), self.base.chain),
                y_min_onehot,
                self.base.chain,
            );
        }
    };
}

pub fn softmaxCrossEntropy(comptime T: type, logits: *TaggedVar, target: *TaggedVar, axis: []const isize) !*TaggedVar {
    return try softmaxCrossEntropyEx(T, logits, target, axis, logits.getContext().current_chain.?);
}

pub fn softmaxCrossEntropyEx(comptime T: type, logits: *TaggedVar, target: *TaggedVar, axis: []const isize, chain: *Chain) !*TaggedVar {
    return try makefunc(SoftmaxCrossEntropy(T), logits, target, axis, chain);
}
