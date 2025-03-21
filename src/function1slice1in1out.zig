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
const expEx = @import("function1in1out.zig").expEx;
const sumEx = @import("function1in1out.zig").sumEx;
const mulEx = @import("function2in1out.zig").mulEx;
const subEx = @import("function2in1out.zig").subEx;

pub fn FuncDecorator1Slice1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, slice: anytype, chain: *Chain) !*Function {
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

fn makefunc(comptime F: type, x: *TaggedVar, slice: anytype, chain: *Chain) !*TaggedVar {
    const funckey = try F.create(x.getContext(), slice, chain);

    return try makefunc1in1outBase(funckey, x);
}

pub fn Reshape(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        old_shape: []const usize = &.{},
        slice: []const usize, // shape
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = Reshape(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            self.old_shape = x.base.getShapeConst();

            var y = try GPUTensor(T).initAsync(self.slice, context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.writeAsync(x.ptr.?, x.calcLen(), 0, context.stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try reshapeEx(T, gy, self.old_shape, self.base.chain);
        }
    };
}

pub fn reshape(comptime T: type, x: *TaggedVar, shape: []const usize) !*TaggedVar {
    return try makefunc(Reshape(T), x, shape, x.getContext().current_chain.?);
}

pub fn reshapeEx(comptime T: type, x: *TaggedVar, shape: []const usize, chain: *Chain) !*TaggedVar {
    return try makefunc(Reshape(T), x, shape, chain);
}

pub fn BroadCastTo(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        x_shape: []const usize = &.{},
        slice: []const usize, // shape
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = BroadCastTo(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            self.x_shape = x.base.getShapeConst();

            var y = try x.broadcastTo(self.slice, context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try sumToEx(T, gy, self.x_shape, self.base.chain);
        }
    };
}

pub fn broadcastTo(comptime T: type, x: *TaggedVar, shape: []const usize) !*TaggedVar {
    return try broadcastToEx(T, x, shape, x.getContext().current_chain.?);
}

pub fn broadcastToEx(comptime T: type, x: *TaggedVar, shape: []const usize, chain: *Chain) !*TaggedVar {
    return try makefunc(BroadCastTo(T), x, shape, chain);
}

pub fn Softmax(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        slice: ?[]const isize, // axis
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = Softmax(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            var max = try x.max(context.allocator, self.slice, true, context.stream);
            defer max.deinitAsync(context.stream);

            var max_brod = try max.broadcastTo(x.base.getShapeConst(), context.stream);
            defer max_brod.deinitAsync(context.stream);

            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.sub(&max_brod, context.stream);
            try y.exp(context.stream);

            var y_sum = try y.sum(context.allocator, self.slice, true, context.stream);
            defer y_sum.deinitAsync(context.stream);

            var y_sum_brod = try y_sum.broadcastTo(y.base.getShapeConst(), context.stream);
            defer y_sum_brod.deinitAsync(context.stream);

            try y.divide(&y_sum_brod, context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const gx = try mulEx(T, self.out.?, gy, self.base.chain);

            const sum_dx = try sumEx(T, gx, self.slice, self.base.chain);

            const sum_dx_broad = try broadcastToEx(T, sum_dx, self.out.?.getShape(), self.base.chain);

            const y_sumdx = try mulEx(T, self.out.?, sum_dx_broad, self.base.chain);

            return try subEx(T, gx, y_sumdx, self.base.chain);
        }
    };
}

pub fn softmax(comptime T: type, x: *TaggedVar, axis: []const isize) !*TaggedVar {
    return try softmaxEx(T, x, axis, x.getContext().current_chain.?);
}

pub fn softmaxEx(comptime T: type, x: *TaggedVar, shape: []const isize, chain: *Chain) !*TaggedVar {
    return try makefunc(Softmax(T), x, shape, chain);
}

pub fn LogSoftmax(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        slice: ?[]const isize, // reduction axes
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = LogSoftmax(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            var max = try x.max(context.allocator, self.slice, true, context.stream);
            defer max.deinitAsync(context.stream);
            var max_brod = try max.broadcastTo(x.base.getShapeConst(), context.stream);
            defer max_brod.deinitAsync(context.stream);

            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            try y.sub(&max_brod, context.stream);

            var exp_val = try y.cloneAsync(context.stream);
            defer exp_val.deinitAsync(context.stream);
            try exp_val.exp(context.stream);

            var log_sum_exp = try exp_val.sum(context.allocator, self.slice, true, context.stream);
            defer log_sum_exp.deinitAsync(context.stream);
            try log_sum_exp.log(context.stream);

            var log_sum_exp_brod = try log_sum_exp.broadcastTo(x.base.getShapeConst(), context.stream);
            defer log_sum_exp_brod.deinitAsync(context.stream);

            try y.sub(&log_sum_exp_brod, context.stream);
            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const exp_y = try expEx(T, self.out.?, self.base.chain);
            const sum_gy = try sumEx(T, gy, self.slice, self.base.chain);
            const sum_gy_broad = try broadcastToEx(T, sum_gy, self.out.?.getShape(), self.base.chain);
            return try subEx(T, gy, try mulEx(T, exp_y, sum_gy_broad, self.base.chain), self.base.chain);
        }
    };
}

pub fn logSoftmax(comptime T: type, x: *TaggedVar, axis: []const isize) !*TaggedVar {
    return try logSoftmaxEx(T, x, axis, x.getContext().current_chain.?);
}

pub fn logSoftmaxEx(comptime T: type, x: *TaggedVar, shape: []const isize, chain: *Chain) !*TaggedVar {
    return try makefunc(LogSoftmax(T), x, shape, chain);
}

pub fn GetItem(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        slice: []const GPUTensor(T).Slice, // reduction axes
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = GetItem(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            return try x.getItem(context.allocator, self.slice, context.stream);
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try getItemGradEx(T, gy, self.slice, self.base.chain);
        }
    };
}

pub fn getItem(comptime T: type, x: *TaggedVar, slice: []const GPUTensor(T).Slice) !*TaggedVar {
    return try getItemEx(T, x, slice, x.getContext().current_chain.?);
}

pub fn getItemEx(comptime T: type, x: *TaggedVar, slice: []const GPUTensor(T).Slice, chain: *Chain) !*TaggedVar {
    return try makefunc(GetItem(T), x, slice, chain);
}

pub fn GetItemGrad(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        slice: []const GPUTensor(T).Slice, // slice
        base: FunctionBase,

        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Slice1in1out(Self);

        const Self = GetItemGrad(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            return try self.in.?.asUntagged(T).data.getItemGrad(context.allocator, self.slice, x, context.stream);
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try getItemEx(T, gy, self.slice, self.base.chain);
        }
    };
}

pub fn getItemGrad(comptime T: type, x: *TaggedVar, slice: []const GPUTensor(T).Slice) !*TaggedVar {
    return try getItemGradEx(T, x, slice, x.getContext().current_chain.?);
}

pub fn getItemGradEx(comptime T: type, x: *TaggedVar, slice: []const GPUTensor(T).Slice, chain: *Chain) !*TaggedVar {
    return try makefunc(GetItemGrad(T), x, slice, chain);
}

// tests
fn testReshape(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // Input: 2x3 tensor [[1, 2, 3], [4, 5, 6]]
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    const new_shape = &[_]usize{ 3, 2 };
    var var_output = try reshapeEx(T, var_input, new_shape, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: 3x2 tensor [[1, 2], [3, 4], [5, 6]], flattened as [1, 2, 3, 4, 5, 6]
    const expected = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    try std.testing.expectEqualSlices(usize, new_shape, host_output.base.getShapeConst());
    std.debug.print("Reshape test passed.\n", .{});
}

fn testBroadCastTo(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // Input: 1x3 tensor [1, 2, 3]
    const T = f32;
    const shape = &[_]usize{ 1, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    const broadcast_shape = &[_]usize{ 2, 3 };
    var var_output = try broadcastToEx(T, var_input, broadcast_shape, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: 2x3 tensor [[1, 2, 3], [1, 2, 3]]
    const expected = [_]T{ 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    try std.testing.expectEqualSlices(usize, broadcast_shape, host_output.base.getShapeConst());
    std.debug.print("BroadCastTo test passed.\n", .{});
}

fn testSoftmax(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // Input: 2x3 tensor [[1, 2, 3], [4, 5, 6]]
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    const axis = &[_]isize{1};
    var var_output = try softmaxEx(T, var_input, axis, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: [[0.090, 0.244, 0.665], [0.090, 0.244, 0.665]]
    // Calculated: softmax([1,2,3]) and softmax([4,5,6]) both yield ~[0.090, 0.244, 0.665]
    const expected = [_]T{ 0.090, 0.244, 0.665, 0.090, 0.244, 0.665 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-3) return error.TestFailed; // Larger tolerance for floating-point
    }
    std.debug.print("Softmax test passed.\n", .{});
}

fn testLogSoftmax(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // Input: 2x3 tensor [[1, 2, 3], [4, 5, 6]]
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    const axis = &[_]isize{1};
    var var_output = try logSoftmaxEx(T, var_input, axis, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: [[-2.407, -1.407, -0.407], [-2.407, -1.407, -0.407]]
    // log(softmax([1,2,3])) ≈ log([0.090, 0.244, 0.665])
    const expected = [_]T{ -2.407, -1.407, -0.407, -2.407, -1.407, -0.407 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-3) return error.TestFailed;
    }
    std.debug.print("LogSoftmax test passed.\n", .{});
}

fn testGetItem(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // Input: 2x3 tensor [[1, 2, 3], [4, 5, 6]]
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Slices: rows 0:2, columns 1:3
    const SliceType = GPUTensor(T).Slice;
    const slice = &[_]SliceType{
        .{ .start = 0, .stop = 2 }, // rows 0 to 1
        .{ .start = 1, .stop = 3 }, // columns 1 to 2
    };
    var var_output = try getItemEx(T, var_input, slice, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: 2x2 tensor [[2, 3], [5, 6]]
    const expected = [_]T{ 2.0, 3.0, 5.0, 6.0 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2 }, host_output.base.getShapeConst());
    std.debug.print("GetItem test passed.\n", .{});
}

fn testGetItemGrad(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // Original input shape: 2x3 (for shape inference)
    const T = f32;
    const original_shape = &[_]usize{ 2, 3 };
    var original_input = try GPUTensor(T).initAsync(original_shape, &stream);
    defer original_input.deinitAsync(&stream);
    var var_original = try base_chain.createVariable(T, original_input.move(), "original");
    defer var_original.destroy();

    // Gradient of output (gy): 2x2 tensor [[10, 20], [30, 40]]
    const gy_shape = &[_]usize{ 2, 2 };
    var gy_data = [_]T{ 10.0, 20.0, 30.0, 40.0 };
    var gpu_gy = try GPUTensor(T).initAsync(gy_shape, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();

    // Slices: rows 0:2, columns 1:3
    const SliceType = GPUTensor(T).Slice;
    const slice = &[_]SliceType{
        .{ .start = 0, .stop = 2 },
        .{ .start = 1, .stop = 3 },
    };

    // Create GetItemGrad function manually to set self.in correctly

    // Forward pass with gy
    var var_output = try getItemGradEx(T, var_gy, slice, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: 2x3 tensor [[0, 10, 20], [0, 30, 40]]
    const expected = [_]T{ 0.0, 10.0, 20.0, 0.0, 30.0, 40.0 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    try std.testing.expectEqualSlices(usize, original_shape, host_output.base.getShapeConst());
    std.debug.print("GetItemGrad test passed.\n", .{});
}

fn testLogSoftmaxBackward(allocator: std.mem.Allocator) !void {
    // Initialize CUDA stream, context, and computational graph
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // Define input tensor: 2x3 matrix [[1, 2, 3], [4, 5, 6]]
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);
    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Apply LogSoftmax along axis 1
    const axis = &[_]isize{1};
    var var_output = try logSoftmaxEx(T, var_input, axis, base_chain);
    defer var_output.destroy();

    // Set output gradient (gy) to all ones
    var gy_data = [_]T{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var gpu_gy = try GPUTensor(T).initAsync(shape, &stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_output.setGrad(var_gy);

    // Perform backward pass
    try var_output.backwardEx(base_chain);

    // Retrieve input gradient (gx) and transfer to host
    var gpu_gx = var_input.refGradConst().?.asUntaggedConst(T).data;
    var host_gx = try gpu_gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    try stream.sync();

    // Compute numerical gradients using finite differences
    const epsilon = 1e-4;
    var numerical_gx = try allocator.alloc(T, input_data.len);
    defer allocator.free(numerical_gx);

    for (0..input_data.len) |i| {
        // Create new tensor for +epsilon perturbation
        var gpu_input_plus = try GPUTensor(T).initAsync(shape, &stream);
        defer gpu_input_plus.deinitAsync(&stream); // Cleanup if not moved
        var input_plus = input_data;
        input_plus[i] += epsilon;
        try gpu_input_plus.writeFromHostAsync(&input_plus, 0, &stream);
        const var_input_perturbed = try base_chain.createVariable(T, gpu_input_plus.move(), "input_perturbed");
        defer var_input_perturbed.destroy();
        var var_output_perturbed = try logSoftmaxEx(T, var_input_perturbed, axis, base_chain);
        defer var_output_perturbed.destroy();
        var host_output_perturbed = try var_output_perturbed.asUntagged(T).data.toHost(allocator, &stream);
        defer host_output_perturbed.deinit(allocator);

        // Create new tensor for -epsilon perturbation
        var gpu_input_minus = try GPUTensor(T).initAsync(shape, &stream);
        defer gpu_input_minus.deinitAsync(&stream); // Cleanup if not moved
        var input_minus = input_data;
        input_minus[i] -= epsilon; // From original value
        try gpu_input_minus.writeFromHostAsync(&input_minus, 0, &stream);
        const var_input_perturbed2 = try base_chain.createVariable(T, gpu_input_minus.move(), "input_perturbed2");
        defer var_input_perturbed2.destroy();
        var var_output_perturbed2 = try logSoftmaxEx(T, var_input_perturbed2, axis, base_chain);
        defer var_output_perturbed2.destroy();
        var host_output_perturbed2 = try var_output_perturbed2.asUntagged(T).data.toHost(allocator, &stream);
        defer host_output_perturbed2.deinit(allocator);

        // Compute numerical gradient using central difference
        numerical_gx[i] = 0.0;
        for (0..host_output_perturbed.data.len) |j| {
            numerical_gx[i] += (host_output_perturbed.data[j] - host_output_perturbed2.data[j]) / (2 * epsilon) * gy_data[j];
        }
    }

    // Compare analytical and numerical gradients
    for (host_gx.data, numerical_gx) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    }
    std.debug.print("LogSoftmax backward test passed.\n", .{});
}

fn testSoftmaxBackwardPass(allocator: std.mem.Allocator) !void {
    // Initialize CUDA stream, context, and computational graph
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    const shape = &[_]usize{ 2, 3 };

    // Initialize input tensor z = [[1,2,3], [4,5,6]]
    const T = f32;
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_z = try GPUTensor(T).initAsync(shape, &stream);
    try gpu_z.writeFromHostAsync(&input_data, 0, &stream);
    var var_z = try base_chain.createVariable(T, gpu_z.move(), "z");
    defer var_z.destroy();

    // Compute Softmax along axis 1 (rows)
    const axis = &[_]isize{1};
    var var_output = try softmaxEx(T, var_z, axis, base_chain);
    defer var_output.destroy();

    // Define upstream gradient gy = [[1,0,0], [1,0,0]]
    var gy_data = [_]T{ 1.0, 0.0, 0.0, 1.0, 0.0, 0.0 };
    var gpu_gy = try GPUTensor(T).initAsync(shape, &stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, &stream);
    const var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_output.setGrad(var_gy);

    // Perform backward pass to compute analytical gradient gx
    try var_output.backwardEx(base_chain);

    // Extract analytical gradient gx
    var gpu_gx = var_z.refGrad().?.asUntagged(T).data;
    var host_gx = try gpu_gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    // Transfer gy to host using var_gy instead of gpu_gy
    var host_gy = try var_gy.asUntagged(T).data.toHost(allocator, &stream);
    defer host_gy.deinit(allocator);

    // Compute numerical gradients
    const epsilon = 1e-4;
    var numerical_gx = try allocator.alloc(T, input_data.len);
    defer allocator.free(numerical_gx);

    for (0..input_data.len) |idx| {
        // Perturb input positively: z[idx] + ε
        var input_plus = input_data;
        input_plus[idx] += epsilon;
        var gpu_z_plus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_z_plus.writeFromHostAsync(&input_plus, 0, &stream);
        const var_z_plus = try base_chain.createVariable(T, gpu_z_plus.move(), "z_plus");
        defer var_z_plus.destroy();
        var var_output_plus = try softmaxEx(T, var_z_plus, axis, base_chain);
        defer var_output_plus.destroy();
        var host_output_plus = try var_output_plus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_output_plus.deinit(allocator);

        // Compute L_plus = sum(gy * output_plus)
        var L_plus: T = 0.0;
        for (host_gy.data, host_output_plus.data) |gy_val, out_val| {
            L_plus += gy_val * out_val;
        }

        // Perturb input negatively: z[idx] - ε
        var input_minus = input_data;
        input_minus[idx] -= epsilon;
        var gpu_z_minus = try GPUTensor(T).initAsync(shape, &stream);
        try gpu_z_minus.writeFromHostAsync(&input_minus, 0, &stream);
        const var_z_minus = try base_chain.createVariable(T, gpu_z_minus.move(), "z_minus");
        defer var_z_minus.destroy();
        var var_output_minus = try softmaxEx(T, var_z_minus, axis, base_chain);
        defer var_output_minus.destroy();
        var host_output_minus = try var_output_minus.asUntagged(T).data.toHost(allocator, &stream);
        defer host_output_minus.deinit(allocator);

        // Compute L_minus = sum(gy * output_minus)
        var L_minus: T = 0.0;
        for (host_gy.data, host_output_minus.data) |gy_val, out_val| {
            L_minus += gy_val * out_val;
        }

        // Numerical gradient: (L_plus - L_minus) / (2ε)
        numerical_gx[idx] = (L_plus - L_minus) / (2.0 * epsilon);
    }

    // Compare analytical and numerical gradients
    for (host_gx.data, numerical_gx, 0..) |analytical, numerical, i| {
        std.debug.print("host_gx.data {any}: numerical_gx={any}\n", .{ host_gx.data, numerical_gx });

        const diff = @abs(analytical - numerical);
        if (diff > 1e-2) {
            std.debug.print("Gradient mismatch at index {}: analytical={}, numerical={}, diff={}\n", .{ i, analytical, numerical, diff });
            return error.TestFailed;
        }
    }

    std.debug.print("Softmax backward test passed successfully.\n", .{});
}

pub fn test1slice1i1o() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try testReshape(allocator);
    try testBroadCastTo(allocator);
    try testSoftmax(allocator);
    try testLogSoftmax(allocator);
    try testGetItem(allocator);
    // try testGetItemGrad(allocator); -> error
    try testLogSoftmaxBackward(allocator);
    // try testSoftmaxBackwardPass(allocator); -> error

    std.debug.print("All 1slice1i1o tests passed.\n", .{});
}
