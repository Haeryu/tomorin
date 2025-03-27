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

pub fn test1tensor1slice1i1o() !void {
    // var gpa: std.heap.DebugAllocator(.{}) = .init;
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();
    // try testsoftmaxCrossEntropyforward(allocator);
    // try testsoftmaxCrossEntropyBackward(allocator);

    std.debug.print("All test1tensor1slice1i1o tests passed.\n", .{});
}
