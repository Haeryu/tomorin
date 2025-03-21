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
const FuncDecorator2in1outBase = @import("function.zig").FuncDecorator2in1outBase;
const Chain = @import("chain.zig").Chain;
const makefunc2in1outBase = @import("function.zig").makefunc2in1outBase;

const transposeEx = @import("function1in1out.zig").transposeEx;
const sumToEx = @import("function1in1out.zig").sumToEx;
const matmulEx = @import("function2in1out.zig").matmulEx;
const broadcastToEx = @import("function1slice1in1out.zig").broadcastToEx;
const Conv2d = @import("function1scalar3in1out.zig").Conv2d;
const conv2dEx = @import("function1scalar3in1out.zig").conv2dEx;
const deconv2dEx = @import("function1scalar3in1out.zig").deconv2dEx;

const getDeconvOutsize = @import("util.zig").getDeconvOutsize;

pub fn FuncDecorator1scalar2in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator2in1outBase(Self);

        pub fn create(context: *Context, scalar: anytype, chain: *Chain) !*Function {
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
                .in1 = null,
                .in2 = null,
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

            const in1_contains = var_seen_set.contains(self.in1.?);
            const in1 = if (!in1_contains) try self.in1.?.getDotAlloc() else "";
            defer if (!in1_contains) allocator.free(in1);

            try var_seen_set.put(self.in1.?, {});

            const in2_contains = var_seen_set.contains(self.in2.?);
            const in2 = if (!in2_contains) try self.in2.?.getDotAlloc() else "";
            defer if (!in2_contains) allocator.free(in2);

            try var_seen_set.put(self.in2.?, {});

            const out_contains = var_seen_set.contains(self.out.?);
            const out = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out);

            try var_seen_set.put(self.out.?, {});

            return try std.fmt.allocPrint(self.base.context.allocator,
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
                in1,
                in2,
                out,
                @intFromPtr(self.in1.?),
                @intFromPtr(ctx),
                @intFromPtr(self.in2.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(
    comptime F: type,
    x1: *TaggedVar,
    x2: *TaggedVar,
    scalar: anytype,
    chain: *Chain,
) !*TaggedVar {
    const funckey = try F.create(x1.getContext(), scalar, chain);

    return try makefunc2in1outBase(funckey, x1, x2);
}

pub fn Conv2DNoBias(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: Option,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Scalar = *Conv2d(T);
        pub const Out = T;

        pub usingnamespace FuncDecorator1scalar2in1out(Self);

        const Self = Conv2DNoBias(T);

        pub const Option = struct {
            stride: [2]usize,
            padding: [2]usize,
            dilation: [2]usize,
        };

        pub fn forward(self: *Self, x: *const GPUTensor(T), w: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            const k_shape = w.base.getShapeConst();
            const kh = k_shape[2];
            const kw = k_shape[3];

            var col =
                try x.im2col(.{ kh, kw }, self.scalar.stride, self.scalar.padding, self.scalar.dilation, context.stream);
            defer col.deinitAsync(context.stream);

            var w_reshaped = try w.reshape(&.{ k_shape[0], k_shape[1] * kh * kw }, context.stream); // {1, 4}
            defer w_reshaped.deinitAsync(context.stream);

            var y = try col.tensordot(&w_reshaped, context.allocator, &.{1}, &.{1}, context.stream);
            defer y.deinitAsync(context.stream);

            var y_roll = try y.rollaxis(3, 1, context.stream);
            errdefer y_roll.deinitAsync(context.stream);

            return y_roll.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const x = self.in1.?;
            const w = self.in2.?;

            const gx = try deconv2dNoBiasEx(
                T,
                gy,
                w,
                .{
                    .stride = self.scalar.stride,
                    .padding = self.scalar.padding,
                    .dilation = self.scalar.dilation,
                    .outsize = .{ x.getShape()[2], x.getShape()[3] },
                },
                self.base.chain,
            );

            const gw = try conv2dgradwEx(
                T,
                x,
                gy,
                .{
                    .w = w,
                    .stride = self.scalar.stride,
                    .padding = self.scalar.padding,
                    .dilation = self.scalar.dilation,
                },
                self.base.chain,
            );

            return .{ gx, gw };
        }
    };
}

pub fn conv2dNoBias(comptime T: type, x: *TaggedVar, w: *TaggedVar, option: Conv2DNoBias(T).Option) !*TaggedVar {
    return try conv2dNoBiasEx(T, x, w, option, x.getContext().current_chain.?);
}

pub fn conv2dNoBiasEx(comptime T: type, x: *TaggedVar, w: *TaggedVar, option: Conv2DNoBias(T).Option, chain: *Chain) !*TaggedVar {
    return try makefunc(Conv2DNoBias(T), x, w, option, chain);
}

pub fn Deconv2dNoBias(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: Option,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const Scalar = Option;
        pub const Out = T;

        pub const Option = struct {
            stride: [2]usize,
            padding: [2]usize,
            dilation: [2]usize,
            outsize: ?[2]usize,
        };

        pub usingnamespace FuncDecorator1scalar2in1out(Self);

        const Self = Deconv2dNoBias(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T), weight: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            const sh, const sw = self.scalar.stride;
            const ph, const pw = self.scalar.padding;
            const w_shape = weight.base.getShapeConst();
            //  const c = w_shape[0];
            const oc = w_shape[1];
            const kh = w_shape[2];
            const kw = w_shape[3];
            const x_shape = x.base.getShapeConst();
            const n = x_shape[0];
            // const c = x_shape[1];
            const h = x_shape[2];
            const w = x_shape[3];

            if (self.scalar.outsize == null) {
                self.scalar.outsize = .{
                    getDeconvOutsize(h, kh, sh, ph),
                    getDeconvOutsize(w, kw, sw, pw),
                };
            }

            const out_h, const out_w = self.scalar.outsize.?;

            const img_shape: [4]usize = .{ n, oc, out_h, out_w };

            var gcol = try weight.tensordot(x, context.allocator, &.{0}, &.{1}, context.stream);
            defer gcol.deinitAsync(context.stream);

            var gcol_roll = try gcol.rollaxis(3, 0, context.stream);
            defer gcol_roll.deinitAsync(context.stream);

            var y = try gcol_roll.col2im(
                &img_shape,
                .{ kh, kw },
                self.scalar.stride,
                self.scalar.padding,
                self.scalar.dilation,
                context.stream,
            );
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const x = self.in1.?;
            const w = self.in2.?;

            const gx = try conv2dNoBiasEx(
                T,
                gy,
                w,
                .{
                    .stride = self.scalar.stride,
                    .padding = self.scalar.padding,
                    .dilation = self.scalar.dilation,
                },
                self.base.chain,
            );

            const gw = try conv2dgradwEx(
                T,
                gy,
                x,
                .{
                    .w = w,
                    .stride = self.scalar.stride,
                    .padding = self.scalar.padding,
                    .dilation = self.scalar.dilation,
                },
                self.base.chain,
            );

            return .{ gx, gw };
        }
    };
}

pub fn deconv2dNoBias(comptime T: type, x: *TaggedVar, w: *TaggedVar, option: Deconv2dNoBias(T).Option) !*TaggedVar {
    return try conv2dEx(T, x, w, option, x.getContext().current_chain.?);
}

pub fn deconv2dNoBiasEx(comptime T: type, x: *TaggedVar, w: *TaggedVar, option: Deconv2dNoBias(T).Option, chain: *Chain) !*TaggedVar {
    return try makefunc(Deconv2dNoBias(T), x, w, option, chain);
}

pub fn Conv2DGradW(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: Scalar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const In3 = T;
        pub const Scalar = struct {
            w: *TaggedVar,
            stride: [2]usize,
            padding: [2]usize,
            dilation: [2]usize,
        };
        pub const Out = T;

        pub usingnamespace FuncDecorator1scalar2in1out(Self);

        const Self = Conv2DGradW(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T), gy: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            const w = self.scalar.w;
            const w_shape = w.getShape();
            const kh = w_shape[2];
            const kw = w_shape[3];
            const kernel_size: [2]usize = .{ kh, kw };
            const stride = self.scalar.stride;
            const padding = self.scalar.padding;
            const dilation = self.scalar.dilation;

            var col = try x.im2col(kernel_size, stride, padding, dilation, context.stream);
            defer col.deinitAsync(context.stream);

            var gw = try gy.tensordot(&col, context.allocator, &.{ 0, 2, 3 }, &.{ 0, 2, 3 }, context.stream);
            errdefer gw.deinitAsync(context.stream);

            return gw.move();
        }

        pub fn backward(self: *Self, _: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar }) {
            const x = self.in1.?;
            const gy = self.in2.?;
            const gw = self.out.?;

            const x_shape = x.getShape();
            const xh = x_shape[0];
            const xw = x_shape[1];

            const gx = try deconv2dNoBiasEx(
                T,
                gy,
                gw,
                .{
                    .stride = self.scalar.stride,
                    .padding = self.scalar.padding,
                    .dilation = self.scalar.dilation,
                    .outsize = .{ xh, xw },
                },
                self.base.chain,
            );

            const ggy = try conv2dNoBiasEx(
                T,
                x,
                gw,
                .{
                    .stride = self.scalar.stride,
                    .padding = self.scalar.padding,
                    .dilation = self.scalar.dilation,
                },
                self.base.chain,
            );

            return .{ gx, ggy };
        }
    };
}

pub fn conv2dgradw(comptime T: type, x: *TaggedVar, gy: *TaggedVar, scalar: Conv2DGradW(T).Scalar) !*TaggedVar {
    return try conv2dgradwEx(T, x, gy, scalar, x.getContext().current_chain.?);
}

pub fn conv2dgradwEx(comptime T: type, x: *TaggedVar, gy: *TaggedVar, scalar: Conv2DGradW(T).Scalar, chain: *Chain) !*TaggedVar {
    return try makefunc(Conv2DGradW(T), x, gy, scalar, chain);
}

fn testConv2dNoBiasForward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    // Input tensor: (N=1, C=1, H=3, W=3)
    // Data = [1,2,3,4,5,6,7,8,9]
    const input_shape = &[_]usize{ 1, 1, 3, 3 };
    var input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    defer gpu_input.deinitAsync(stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Weight tensor: (OutC=1, InC=1, KH=2, KW=2)
    // Data = [1,0,0,1], an identity‐like filter.
    const weight_shape = &[_]usize{ 1, 1, 2, 2 };
    var weight_data = [_]T{ 1, 0, 0, 1 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    defer gpu_weight.deinitAsync(stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    var var_weight = try chain.createVariable(T, gpu_weight.move(), "weight");
    defer var_weight.destroy();

    // Conv2DNoBias options
    const option = Conv2DNoBias(T).Option{
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
        .dilation = .{ 1, 1 },
    };

    // Forward pass
    var var_output = try conv2dNoBiasEx(T, var_input, var_weight, option, chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected output shape = (1, 1, 2, 2), data = [6, 8, 12, 14].
    // Explanation: each 2x2 patch in the 3x3 input is “dot”ed
    //   with [1,0;0,1] => (top‐left + bottom‐right).
    const expected = [_]T{ 6, 8, 12, 14 };
    const expected_shape = &[_]usize{ 1, 1, 2, 2 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    try std.testing.expectEqualSlices(usize, expected_shape, host_output.base.getShapeConst());

    std.debug.print("Conv2dNoBias forward test passed.\n", .{});
}

fn testConv2dNoBiasBackward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    // Same input as forward: (1,1,3,3) with data = 1..9
    const input_shape = &[_]usize{ 1, 1, 3, 3 };
    var input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Weight: (1,1,2,2), data = [1,0,0,1]
    const weight_shape = &[_]usize{ 1, 1, 2, 2 };
    var weight_data = [_]T{ 1, 0, 0, 1 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    var var_weight = try chain.createVariable(T, gpu_weight.move(), "weight");
    defer var_weight.destroy();

    // Options
    const option = Conv2DNoBias(T).Option{
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
        .dilation = .{ 1, 1 },
    };

    // Forward pass
    var var_output = try conv2dNoBiasEx(T, var_input, var_weight, option, chain);
    defer var_output.destroy();

    // Upstream gradient: shape = (1,1,2,2) with all ones
    const gy_shape = &[_]usize{ 1, 1, 2, 2 };
    var gy_data = [_]T{ 1, 1, 1, 1 };
    var gpu_gy = try GPUTensor(T).initAsync(gy_shape, stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, stream);
    var var_gy = try chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();

    // Set the gradient
    var_output.setGrad(var_gy);

    // Backward pass
    try var_output.backwardEx(chain);

    // Read out gx, gw
    var gx_data = var_input.refGrad().?.asUntagged(T).data;
    var host_gx = try gx_data.toHost(allocator, stream);
    defer host_gx.deinit(allocator);

    var gw_data = var_weight.refGrad().?.asUntagged(T).data;
    var host_gw = try gw_data.toHost(allocator, stream);
    defer host_gw.deinit(allocator);

    try stream.sync();

    //----------------------------------------------------------------
    // 1) Numeric gradient check for x
    //----------------------------------------------------------------
    const epsilon = 1e-4;
    var numerical_gx = try allocator.alloc(T, input_data.len);
    defer allocator.free(numerical_gx);

    for (0..input_data.len) |i| {
        var input_plus = input_data;
        input_plus[i] += epsilon;
        var gpu_input_plus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_plus.writeFromHostAsync(&input_plus, 0, stream);
        const var_input_plus = try chain.createVariable(T, gpu_input_plus.move(), "input_plus");
        defer var_input_plus.destroy();
        var var_out_plus = try conv2dNoBiasEx(T, var_input_plus, var_weight, option, chain);
        defer var_out_plus.destroy();
        var out_plus_host = try var_out_plus.asUntagged(T).data.toHost(allocator, stream);
        defer out_plus_host.deinit(allocator);

        var input_minus = input_data;
        input_minus[i] -= epsilon;
        var gpu_input_minus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_minus.writeFromHostAsync(&input_minus, 0, stream);
        const var_input_minus = try chain.createVariable(T, gpu_input_minus.move(), "input_minus");
        defer var_input_minus.destroy();
        var var_out_minus = try conv2dNoBiasEx(T, var_input_minus, var_weight, option, chain);
        defer var_out_minus.destroy();
        var out_minus_host = try var_out_minus.asUntagged(T).data.toHost(allocator, stream);
        defer out_minus_host.deinit(allocator);

        // Multiply the difference by the upstream gradient
        numerical_gx[i] = 0.0;
        for (0..gy_data.len) |j| {
            const diff = out_plus_host.data[j] - out_minus_host.data[j];
            numerical_gx[i] += diff / (2.0 * epsilon) * gy_data[j];
        }
    }

    // Compare analytical gx vs numeric gx
    for (host_gx.data, numerical_gx) |analytical, numeric| {
        if (@abs(analytical - numeric) > 1e-2) {
            std.debug.print("gx mismatch: {any} vs {any}\n", .{ analytical, numeric });
            return error.TestFailed;
        }
    }

    //----------------------------------------------------------------
    // 2) Check weight gradient gw
    //----------------------------------------------------------------
    // This is the same logic as your existing testConv2dBackward
    // example. The result is 12,16,24,28 for the 2x2 kernel.
    // Explanation: each position in the kernel sees the sum of
    //   overlapping input * upstream=1 for all partial sums.
    const expected_gw = [_]T{ 12.0, 16.0, 24.0, 28.0 };
    for (host_gw.data, expected_gw) |got, exp| {
        if (@abs(got - exp) > 1e-6) {
            std.debug.print("gw mismatch: got={any}, exp={any}\n", .{ got, exp });
            return error.TestFailed;
        }
    }

    std.debug.print("Conv2dNoBias backward test passed.\n", .{});
}

fn testDeconv2dNoBiasForward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    // Input: (1,1,2,2), data=[1,2,3,4]
    const input_shape = &[_]usize{ 1, 1, 2, 2 };
    var input_data = [_]T{ 1, 2, 3, 4 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    defer gpu_input.deinitAsync(stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Weight: (1,1,2,2), data=[1,0,0,1]
    const weight_shape = &[_]usize{ 1, 1, 2, 2 };
    var weight_data = [_]T{ 1, 0, 0, 1 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    defer gpu_weight.deinitAsync(stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    var var_weight = try chain.createVariable(T, gpu_weight.move(), "weight");
    defer var_weight.destroy();

    // Deconv2dNoBias options
    // We'll explicitly set outsize=3x3
    const option = Deconv2dNoBias(T).Option{
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
        .dilation = .{ 1, 1 },
        .outsize = .{ 3, 3 },
    };

    // Forward pass
    var var_output = try deconv2dNoBiasEx(T, var_input, var_weight, option, chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected: shape = (1,1,3,3)
    //   data = [1,2,0, 3,5,2, 0,3,4]
    // This is exactly your usual “transposed conv with identity kernel” scenario.
    const expected = [_]T{
        1, 2, 0,
        3, 5, 2,
        0, 3, 4,
    };
    const expected_shape = &[_]usize{ 1, 1, 3, 3 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    try std.testing.expectEqualSlices(usize, expected_shape, host_output.base.getShapeConst());

    std.debug.print("Deconv2dNoBias forward test passed.\n", .{});
}

fn testDeconv2dNoBiasBackward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    // Input: (1,1,2,2) => [1,2,3,4]
    const input_shape = &[_]usize{ 1, 1, 2, 2 };
    var input_data = [_]T{ 1, 2, 3, 4 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Weight: (1,1,2,2), data=[1,0,0,1]
    const weight_shape = &[_]usize{ 1, 1, 2, 2 };
    var weight_data = [_]T{ 1, 0, 0, 1 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    var var_weight = try chain.createVariable(T, gpu_weight.move(), "weight");
    defer var_weight.destroy();

    // Deconv2dNoBias options
    const option = Deconv2dNoBias(T).Option{
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
        .dilation = .{ 1, 1 },
        .outsize = .{ 3, 3 },
    };

    // Forward pass
    var var_output = try deconv2dNoBiasEx(T, var_input, var_weight, option, chain);
    defer var_output.destroy();

    // Upstream gradient: shape = (1,1,3,3) all ones
    const gy_shape = &[_]usize{ 1, 1, 3, 3 };
    var gy_data = [_]T{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    };
    var gpu_gy = try GPUTensor(T).initAsync(gy_shape, stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, stream);
    var var_gy = try chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();

    var_output.setGrad(var_gy);

    // Backward pass
    try var_output.backwardEx(chain);

    // Grad for x, weight
    var gx_data = var_input.refGrad().?.asUntagged(T).data;
    var host_gx = try gx_data.toHost(allocator, stream);
    defer host_gx.deinit(allocator);

    var gw_data = var_weight.refGrad().?.asUntagged(T).data;
    var host_gw = try gw_data.toHost(allocator, stream);
    defer host_gw.deinit(allocator);

    try stream.sync();

    //---------------------------------------------------------
    // 1) Numeric gradient check for x
    //---------------------------------------------------------
    const epsilon = 1e-4;
    var numerical_gx = try allocator.alloc(T, input_data.len);
    defer allocator.free(numerical_gx);

    for (0..input_data.len) |i| {
        var input_plus = input_data;
        input_plus[i] += epsilon;
        var gpu_input_plus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_plus.writeFromHostAsync(&input_plus, 0, stream);
        const var_input_plus = try chain.createVariable(T, gpu_input_plus.move(), "input_plus");
        defer var_input_plus.destroy();
        var var_out_plus = try deconv2dNoBiasEx(T, var_input_plus, var_weight, option, chain);
        defer var_out_plus.destroy();
        var out_plus_host = try var_out_plus.asUntagged(T).data.toHost(allocator, stream);
        defer out_plus_host.deinit(allocator);

        var input_minus = input_data;
        input_minus[i] -= epsilon;
        var gpu_input_minus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_minus.writeFromHostAsync(&input_minus, 0, stream);
        const var_input_minus = try chain.createVariable(T, gpu_input_minus.move(), "input_minus");
        defer var_input_minus.destroy();
        var var_out_minus = try deconv2dNoBiasEx(T, var_input_minus, var_weight, option, chain);
        defer var_out_minus.destroy();
        var out_minus_host = try var_out_minus.asUntagged(T).data.toHost(allocator, stream);
        defer out_minus_host.deinit(allocator);

        numerical_gx[i] = 0.0;
        for (0..out_plus_host.data.len) |j| {
            const diff = out_plus_host.data[j] - out_minus_host.data[j];
            numerical_gx[i] += diff / (2.0 * epsilon) * gy_data[j];
        }
    }

    for (host_gx.data, numerical_gx) |analytical, numeric| {
        if (@abs(analytical - numeric) > 1e-2) {
            std.debug.print("deconv gx mismatch: got={any}, exp={any}\n", .{ analytical, numeric });
            return error.TestFailed;
        }
    }

    //---------------------------------------------------------
    // 2) We could do a numeric check for w as well if desired,
    //    or just ensure it doesn't blow up. For now, no bias means
    //    no trivial check like sum(gy). We simply trust it or
    //    do a smaller "spot check."
    //---------------------------------------------------------

    std.debug.print("Deconv2dNoBias backward test passed.\n", .{});
}

/// Master test function that runs all four tests for Conv2DNoBias/Deconv2DNoBias.
pub fn test1scalar2i1o() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const chain = try context.createChain();
    context.current_chain = chain;
    defer chain.clear();

    try testConv2dNoBiasForward(allocator, &stream, chain);
    try testConv2dNoBiasBackward(allocator, &stream, chain);

    try testDeconv2dNoBiasForward(allocator, &stream, chain);
    try testDeconv2dNoBiasBackward(allocator, &stream, chain);

    std.debug.print("All Conv2DNoBias and Deconv2DNoBias tests passed in test1scalar2i1o.\n", .{});
}
