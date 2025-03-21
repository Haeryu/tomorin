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

pub fn FuncDecorator1Scalar1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, scalar: Self.Scalar, chain: *Chain) !*Function {
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
                .scalar = scalar,
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

            const scalar = try std.fmt.allocPrint(allocator, "{} [label=\"{s}\", color=aquamarine, style=filled, shape=circle]", .{
                @intFromPtr(&self.scalar),
                @typeName(Self.Scalar),
            });
            defer allocator.free(scalar);

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
                scalar,
                in,
                out,
                @intFromPtr(&self.scalar),
                @intFromPtr(ctx),
                @intFromPtr(self.in.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(comptime F: type, x: *TaggedVar, scalar: F.Scalar, chain: *Chain) !*TaggedVar {
    const funckey = try F.create(x.getContext(), scalar, chain);

    return try makefunc1in1outBase(funckey, x);
}

pub fn Shift(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: T,
        base: FunctionBase,

        pub const Scalar = T;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Shift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.shift(self.scalar, context.stream);
            return y;
        }

        pub fn backward(_: *Self, gy: *TaggedVar) !*TaggedVar {
            return gy;
        }
    };
}

pub fn shift(comptime T: type, x: *TaggedVar, scalar: T) !*TaggedVar {
    return try shiftEx(T, x, scalar, x.getContext().current_chain.?);
}

pub fn shiftEx(comptime T: type, x: *TaggedVar, scalar: T, chain: *Chain) !*TaggedVar {
    return try makefunc(Shift(T), x, scalar, chain);
}

pub fn Scale(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: T,
        base: FunctionBase,

        pub const Scalar = T;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Scale(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            try y.scale(self.scalar, context.stream);
            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try scaleEx(T, gy, self.scalar, self.base.chain);
        }
    };
}

pub fn scale(comptime T: type, x: *TaggedVar, scalar: T) !*TaggedVar {
    return try scaleEx(T, x, scalar, x.getContext().current_chain.?);
}

pub fn scaleEx(comptime T: type, x: *TaggedVar, scalar: T, chain: *Chain) !*TaggedVar {
    return try makefunc(Scale(T), x, scalar, chain);
}

pub fn Powf(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: T,
        base: FunctionBase,

        pub const Scalar = T;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Powf(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            try y.powf(self.scalar, context.stream);
            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const x_cmin1 = try powfEx(T, self.in.?, self.scalar - 1.0, self.base.chain);
            const c_x_cmin1 = try scaleEx(T, x_cmin1, self.scalar, self.base.chain);
            return try mulEx(T, c_x_cmin1, gy, self.base.chain);
        }
    };
}

pub fn powf(comptime T: type, x: *TaggedVar, scalar: T) !*TaggedVar {
    return try powfEx(T, x, scalar, x.getContext().current_chain.?);
}

pub fn powfEx(comptime T: type, x: *TaggedVar, scalar: T, chain: *Chain) !*TaggedVar {
    return try makefunc(Powf(T), x, scalar, chain);
}

pub fn Pow(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: i32,
        base: FunctionBase,

        pub const Scalar = i32;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = Pow(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);
            try y.pow(self.scalar, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const x_cmin1 = try powEx(T, self.in.?, self.scalar - 1, self.base.chain);
            const c_x_cmin1 = try scaleEx(
                T,
                x_cmin1,
                if (T == BF16) BF16.fromF32(@floatFromInt(self.scalar)) else @floatFromInt(self.scalar),
                self.base.chain,
            );
            return try mulEx(T, c_x_cmin1, gy, self.base.chain);
        }
    };
}

pub fn pow(comptime T: type, x: *TaggedVar, scalar: i32) !*TaggedVar {
    return try powEx(T, x, scalar, x.getContext().current_chain.?);
}

pub fn powEx(comptime T: type, x: *TaggedVar, scalar: i32, chain: *Chain) !*TaggedVar {
    return try makefunc(Pow(T), x, scalar, chain);
}

pub fn MaxPooling(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: Scalar,
        base: FunctionBase,

        pub const Scalar = struct {
            kernel_size: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
        };
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = MaxPooling(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            const stream = context.stream;
            const kernel_size = self.scalar.kernel_size;
            const stride = self.scalar.stride;
            const padding = self.scalar.padding;
            return try x.maxPool2dForward(kernel_size, stride, padding, stream);
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const context = self.base.context;

            var back = try self.in.?.asUntagged(T).data.maxPool2dBackward(
                &gy.asUntagged(T).data,
                self.scalar.kernel_size,
                self.scalar.stride,
                self.scalar.padding,
                context.stream,
            );
            errdefer back.deinitAsync(context.stream);

            return try self.base.chain.createVariable(T, back.move(), null);
        }
    };
}

// Wrapper functions for convenience
pub fn maxPooling(comptime T: type, x: *TaggedVar, scalar: MaxPooling(T).Scalar) !*TaggedVar {
    return try maxPoolingEx(T, x, scalar, x.getContext().current_chain.?);
}

pub fn maxPoolingEx(comptime T: type, x: *TaggedVar, scalar: MaxPooling(T).Scalar, chain: *Chain) !*TaggedVar {
    return try makefunc(MaxPooling(T), x, scalar, chain);
}

pub fn AveragePooling(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: Scalar,
        base: FunctionBase,

        pub const Scalar = struct {
            kernel_size: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
        };
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Scalar1in1out(Self);

        const Self = AveragePooling(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            const stream = context.stream;
            const kernel_size = self.scalar.kernel_size;
            const stride = self.scalar.stride;
            const padding = self.scalar.padding;
            return try x.avgPool2dForward(kernel_size, stride, padding, stream);
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const x_shape = self.in.?.getShape();
            const context = self.base.context;

            var back = try gy.asUntagged(T).data.avgPool2dBackward(
                x_shape,
                self.scalar.kernel_size,
                self.scalar.stride,
                self.scalar.padding,
                context.stream,
            );
            errdefer back.deinitAsync(context.stream);

            return try self.base.chain.createVariable(T, back.move(), null);
        }
    };
}

// Wrapper functions for convenience
pub fn averagePooling(comptime T: type, x: *TaggedVar, scalar: AveragePooling(T).Scalar) !*TaggedVar {
    return try averagePoolingEx(T, x, scalar, x.getContext().current_chain.?);
}

pub fn averagePoolingEx(comptime T: type, x: *TaggedVar, scalar: AveragePooling(T).Scalar, chain: *Chain) !*TaggedVar {
    return try makefunc(AveragePooling(T), x, scalar, chain);
}

// tests
fn testShift(allocator: std.mem.Allocator) !void {
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

    // Input: 2x2 tensor [1.0, 2.0, 3.0, 4.0]
    const T = f32;
    const shape = &[_]usize{ 2, 2 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    const scalar: T = 2.0;
    var var_output = try shiftEx(T, var_input, scalar, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: [1.0 + 2.0, 2.0 + 2.0, 3.0 + 2.0, 4.0 + 2.0] = [3.0, 4.0, 5.0, 6.0]
    const expected = [_]T{ 3.0, 4.0, 5.0, 6.0 };
    for (host_output.data, expected) |got, expe| {
        if (@abs(got - expe) > 1e-6) return error.TestFailed;
    }
    std.debug.print("Shift test passed.\n", .{});
}

fn testScale(allocator: std.mem.Allocator) !void {
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

    // Input: 2x2 tensor [1.0, 2.0, 3.0, 4.0]
    const T = f32;
    const shape = &[_]usize{ 2, 2 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    const scalar: T = 3.0;
    var var_output = try scaleEx(T, var_input, scalar, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: [1.0 * 3.0, 2.0 * 3.0, 3.0 * 3.0, 4.0 * 3.0] = [3.0, 6.0, 9.0, 12.0]
    const expected = [_]T{ 3.0, 6.0, 9.0, 12.0 };
    for (host_output.data, expected) |got, expe| {
        if (@abs(got - expe) > 1e-6) return error.TestFailed;
    }
    std.debug.print("Scale test passed.\n", .{});
}

fn testPowf(allocator: std.mem.Allocator) !void {
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

    // Input: 2x2 tensor [1.0, 2.0, 3.0, 4.0]
    const T = f32;
    const shape = &[_]usize{ 2, 2 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    const scalar: T = 2.0;
    var var_output = try powfEx(T, var_input, scalar, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: [1.0^2, 2.0^2, 3.0^2, 4.0^2] = [1.0, 4.0, 9.0, 16.0]
    const expected = [_]T{ 1.0, 4.0, 9.0, 16.0 };
    for (host_output.data, expected) |got, expe| {
        if (@abs(got - expe) > 1e-6) return error.TestFailed;
    }
    std.debug.print("Powf test passed.\n", .{});
}

fn testPow(allocator: std.mem.Allocator) !void {
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

    // Input: 2x2 tensor [1.0, 2.0, 3.0, 4.0]
    const T = f32;
    const shape = &[_]usize{ 2, 2 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    const scalar: i32 = 3;
    var var_output = try powEx(T, var_input, scalar, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: [1.0^3, 2.0^3, 3.0^3, 4.0^3] = [1.0, 8.0, 27.0, 64.0]
    const expected = [_]T{ 1.0, 8.0, 27.0, 64.0 };
    for (host_output.data, expected) |got, expe| {
        if (@abs(got - expe) > 1e-6) return error.TestFailed;
    }
    std.debug.print("Pow test passed.\n", .{});
}

fn testMaxPooling(allocator: std.mem.Allocator) !void {
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

    // Input: [1, 1, 4, 4] tensor
    const T = f32;
    const shape = &[_]usize{ 1, 1, 4, 4 };
    var input_data = [_]T{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Pooling parameters: 2x2 kernel, stride 2, no padding
    const kernel_size = [2]usize{ 2, 2 };
    const stride = [2]usize{ 2, 2 };
    const padding = [2]usize{ 0, 0 };
    const scalar = MaxPooling(T).Scalar{
        .kernel_size = kernel_size,
        .stride = stride,
        .padding = padding,
    };

    // Forward pass
    var var_output = try maxPoolingEx(T, var_input, scalar, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected output: max values of each 2x2 window
    // Top-left: max(1, 2, 5, 6) = 6.0
    // Top-right: max(3, 4, 7, 8) = 8.0
    // Bottom-left: max(9, 10, 13, 14) = 14.0
    // Bottom-right: max(11, 12, 15, 16) = 16.0
    const expected_output = [_]T{ 6.0, 8.0, 14.0, 16.0 };
    for (host_output.data, expected_output) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    // Backward pass: Set upstream gradient (gy) to all ones
    var gy_data = [_]T{ 1.0, 1.0, 1.0, 1.0 };
    var gpu_gy = try GPUTensor(T).initAsync(&[_]usize{ 1, 1, 2, 2 }, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, &stream);

    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();

    try var_output.backwardEx(base_chain);
    var gx = var_input.refGrad().?;
    defer gx.destroy();

    var gpu_gx = gx.asUntagged(T).data;
    var host_gx = try gpu_gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    try stream.sync();

    // Expected gx: gradients routed to max positions
    // 6.0 from (5, 6), 8.0 from (7, 8), 14.0 from (13, 14), 16.0 from (15, 16)
    const expected_gx = [_]T{
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
    };
    for (host_gx.data, expected_gx) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    std.debug.print("MaxPooling test passed.\n", .{});
}

fn testAveragePooling(allocator: std.mem.Allocator) !void {
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

    // Input: [1, 1, 4, 4] tensor
    const T = f32;
    const shape = &[_]usize{ 1, 1, 4, 4 };
    var input_data = [_]T{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Pooling parameters: 2x2 kernel, stride 2, no padding
    const kernel_size = [2]usize{ 2, 2 };
    const stride = [2]usize{ 2, 2 };
    const padding = [2]usize{ 0, 0 };
    const scalar = AveragePooling(T).Scalar{
        .kernel_size = kernel_size,
        .stride = stride,
        .padding = padding,
    };

    // Forward pass
    var var_output = try averagePoolingEx(T, var_input, scalar, base_chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, &stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected output: averages of each 2x2 window
    // Top-left: (1 + 2 + 5 + 6) / 4 = 3.5
    // Top-right: (3 + 4 + 7 + 8) / 4 = 5.5
    // Bottom-left: (9 + 10 + 13 + 14) / 4 = 11.5
    // Bottom-right: (11 + 12 + 15 + 16) / 4 = 13.5
    const expected_output = [_]T{ 3.5, 5.5, 11.5, 13.5 };
    for (host_output.data, expected_output) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    // Backward pass: Set upstream gradient (gy) to all ones
    var gy_data = [_]T{ 1.0, 1.0, 1.0, 1.0 };
    var gpu_gy = try GPUTensor(T).initAsync(&[_]usize{ 1, 1, 2, 2 }, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, &stream);

    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();

    try var_output.backwardEx(base_chain);
    var gx = var_input.refGrad().?;
    defer gx.destroy();

    var gpu_gx = gx.asUntagged(T).data;
    var host_gx = try gpu_gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    try stream.sync();

    // Expected gx: each position in a 2x2 window gets 1.0 / 4 = 0.25
    const expected_gx = [_]T{
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
    };
    for (host_gx.data, expected_gx) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    std.debug.print("AveragePooling test passed.\n", .{});
}

pub fn test1s1i1o() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    try testShift(allocator);
    try testScale(allocator);
    try testPowf(allocator);
    try testPow(allocator);
    try testMaxPooling(allocator);
    try testAveragePooling(allocator);

    std.debug.print("All scalar tests passed.\n", .{});
}
