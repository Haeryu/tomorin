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

const sumToEx = @import("function1in1out.zig").sumToEx;
const getItemEx = @import("function1slice1in1out.zig").getItemEx;
const expEx = @import("function1in1out.zig").expEx;
const sumEx = @import("function1in1out.zig").sumEx;
const mulEx = @import("function2in1out.zig").mulEx;
const subEx = @import("function2in1out.zig").subEx;

pub fn FuncDecorator2Slice1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, slice1: anytype, slice2: anytype, chain: *Chain) !*Function {
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
                .slice1 = slice1,
                .slice2 = slice2,
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

fn makefunc(comptime F: type, x: *TaggedVar, slice1: anytype, slice2: anytype, chain: *Chain) !*TaggedVar {
    const funckey = try F.create(x.getContext(), slice1, slice2, chain);

    return try makefunc1in1outBase(funckey, x);
}

pub fn GetItemGrad(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        slice1: []const usize, // old shape
        slice2: []const GPUTensor(T).Slice, // slice
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2Slice1in1out(Self);

        const Self = GetItemGrad(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            return try self.in.?.asUntagged(T).data.getItemGrad(context.allocator, self.slice2, x, context.stream);
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try getItemEx(T, gy, self.slice2, self.base.chain);
        }
    };
}

pub fn getItemGrad(comptime T: type, x: *TaggedVar, slice: []const GPUTensor(T).Slice, old_shape: []const usize) !*TaggedVar {
    return try getItemGradEx(T, x, slice, old_shape, x.getContext().current_chain.?);
}

pub fn getItemGradEx(comptime T: type, x: *TaggedVar, slice: []const GPUTensor(T).Slice, old_shape: []const usize, chain: *Chain) !*TaggedVar {
    return try makefunc(GetItemGrad(T), x, slice, old_shape, chain);
}
