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
const FuncDecorator3in1outBase = @import("function.zig").FuncDecorator3in1outBase;
const Chain = @import("chain.zig").Chain;
const makefunc3in1outBase = @import("function.zig").makefunc3in1outBase;

const transposeEx = @import("function1in1out.zig").transposeEx;
const sumToEx = @import("function1in1out.zig").sumToEx;
const matmulEx = @import("function2in1out.zig").matmulEx;
const broadcastToEx = @import("function1slice1in1out.zig").broadcastToEx;
const conv2DNoBiasEx = @import("function1scalar2in1out.zig").conv2DNoBiasEx;
const conv1DNoBiasEx = @import("function1scalar2in1out.zig").conv1DNoBiasEx;
const conv2DgradwEx = @import("function1scalar2in1out.zig").conv2DgradwEx;
const conv1DgradwEx = @import("function1scalar2in1out.zig").conv1DgradwEx;
const deconv2DNoBiasEx = @import("function1scalar2in1out.zig").deconv2DNoBiasEx;
const deconv1DNoBiasEx = @import("function1scalar2in1out.zig").deconv1DNoBiasEx;

const getDeconvOutsize = @import("util.zig").getDeconvOutsize;

const dbg = @import("util.zig").debugPrintGpuTensor;

pub fn FuncDecorator1scalar3in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator3in1outBase(Self);

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
                .in3 = null,
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

            const in3_contains = var_seen_set.contains(self.in3.?);
            const in3 = if (!in3_contains) try self.in3.?.getDotAlloc() else "";
            defer if (!in3_contains) allocator.free(in3);

            try var_seen_set.put(self.in3.?, {});

            const out_contains = var_seen_set.contains(self.out.?);
            const out = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out);

            try var_seen_set.put(self.out.?, {});

            return try std.fmt.allocPrint(self.base.context.allocator,
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
                in1,
                in2,
                in3,
                out,
                @intFromPtr(self.in1.?),
                @intFromPtr(ctx),
                @intFromPtr(self.in2.?),
                @intFromPtr(ctx),
                @intFromPtr(self.in3.?),
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
    x3: *TaggedVar,
    scalar: anytype,
    chain: *Chain,
) !*TaggedVar {
    const funckey = try F.create(x1.getContext(), scalar, chain);

    return try makefunc3in1outBase(funckey, x1, x2, x3);
}

pub fn Conv1D(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        in3: ?*TaggedVar, // bias
        out: ?*TaggedVar,
        scalar: Option,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const In3 = T; // bias
        pub const Scalar = Option;
        pub const Out = T;

        pub const Option = struct {
            stride: usize,
            padding: usize,
            dilation: usize,
        };

        pub usingnamespace FuncDecorator1scalar3in1out(Self);

        const Self = Conv1D(T);

        /// Forward pass
        /// x: shape [N, C, L]
        /// w: shape [OutC, C, K]
        /// b: shape [OutC]
        pub fn forward(self: *Self, x: *const GPUTensor(T), w: *const GPUTensor(T), b: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            const w_shape = w.base.getShapeConst(); // [OutC, InC, K]
            const outc = w_shape[0];
            const c = w_shape[1];
            const k = w_shape[2];

            // im2col for 1D
            var col = try x.im2col1d(
                k,
                self.scalar.stride,
                self.scalar.padding,
                self.scalar.dilation,
                context.stream,
            );
            defer col.deinitAsync(context.stream);
            // Now col is typically [N, C*k, out_len], or something similar.

            // Reshape w => [OutC, C*K] to dot with col => [N, C*K, out_len]
            var w_reshaped = try w.reshape(&.{ outc, c * k }, context.stream);
            defer w_reshaped.deinitAsync(context.stream);

            // tensordot: we want (C*K) dimension to match
            // => y shape [N, OutC, out_len]
            var y = try col.tensordot(&w_reshaped, context.allocator, &.{1}, &.{1}, context.stream);
            defer y.deinitAsync(context.stream);

            var y_roll = try y.rollaxis(2, 1, context.stream);
            defer y_roll.deinitAsync(context.stream);

            // Add bias
            var b_reshaped = try b.reshape(&.{ 1, b.calcLen(), 1 }, context.stream);
            defer b_reshaped.deinitAsync(context.stream);

            // broadcast [1,OutC,1] => [N,OutC,outLen]
            var b_brod = try b_reshaped.broadcastTo(y_roll.base.getShapeConst(), context.stream);
            defer b_brod.deinitAsync(context.stream);

            try y_roll.add(&b_brod, context.stream);

            // Final shape is [N, OutC, out_len] – no further rollaxis needed for 1D.

            return y_roll.move();
        }

        /// Backward pass
        /// gy: shape matches output => [N, OutC, out_len]
        /// Returns (gx, gw, gb).
        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar, *TaggedVar }) {
            const x = self.in1.?; // input
            const w = self.in2.?; // weight

            // (1) Gradient wrt x => deconv1dNoBias(gy, w)
            const gx = try deconv1DNoBiasEx(
                T,
                gy,
                w,
                .{
                    .stride = self.scalar.stride,
                    .padding = self.scalar.padding,
                    .dilation = self.scalar.dilation,
                    // outsize must be x.L
                    .outsize = x.getShape()[2],
                },
                self.base.chain,
            );

            // (2) Gradient wrt w => conv1dgradw( x, gy )
            const gw = try conv1DgradwEx(
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

            // (3) Gradient wrt bias => sum(gy) across N,L
            var gb_data = try gy.asUntagged(T).data.sum(
                self.base.context.allocator,
                &.{ 0, 2 }, // sum over N and out_len
                true,
                self.base.context.stream,
            );
            errdefer gb_data.deinitAsync(self.base.context.stream);

            const gb_var = try self.base.chain.createVariable(T, gb_data.move(), null);

            return .{ gx, gw, gb_var };
        }
    };
}

/// Helper to chain them up:
pub fn conv1D(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar, option: Conv1D(T).Option) !*TaggedVar {
    return try conv1DEx(T, x, w, b, option, x.getContext().current_chain.?);
}

pub fn conv1DEx(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar, option: Conv1D(T).Option, chain: *Chain) !*TaggedVar {
    return try makefunc(Conv1D(T), x, w, b, option, chain);
}

pub fn Deconv1D(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        in3: ?*TaggedVar, // bias
        out: ?*TaggedVar,
        scalar: Option,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const In3 = T;
        pub const Scalar = Option;
        pub const Out = T;

        pub const Option = struct {
            stride: usize,
            padding: usize,
            dilation: usize,
            outsize: ?usize,
        };

        pub usingnamespace FuncDecorator1scalar3in1out(Self);

        const Self = Deconv1D(T);

        /// Forward pass
        /// x: [N, C, L]
        /// w: [OutC, C, K]
        /// b: [OutC]
        pub fn forward(self: *Self, x: *const GPUTensor(T), weight: *const GPUTensor(T), b: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;

            const s = self.scalar.stride;
            const p = self.scalar.padding;
            const d = self.scalar.dilation;

            const w_shape = weight.base.getShapeConst(); // [OutC, InC, K]
            const outc = w_shape[0];
            const k = w_shape[2];

            const x_shape = x.base.getShapeConst(); // [N, C, L]
            const n = x_shape[0];
            const l = x_shape[2];

            // If no outsize is specified, compute from transposed-conv formula
            if (self.scalar.outsize == null) {
                // same logic as 2D but for 1D:
                // outsize = getDeconvOutsize(L, kernel=K, stride=S, padding=P)
                self.scalar.outsize = getDeconvOutsize(l, k, s, p);
            }

            const out_l = self.scalar.outsize.?;

            const out_shape: [3]usize = .{ n, outc, out_l };

            // We'll do the "tensordot" approach just like Deconv2d
            // weight:  [OutC, InC, K]
            // x:       [N, InC, L]
            // We want to dot the "InC" dimension so reorder or use tensordot
            var gcol = try weight.tensordot(
                x,
                context.allocator,
                &.{0}, // weight OutC dimension not matched
                &.{1}, // x InC dimension
                context.stream,
            );
            defer gcol.deinitAsync(context.stream);
            // shape of gcol is something like [N, OutC, L, K] (depending on how tensordot’s code arranges dims)

            // We next need to "col2im1d" across L,K into out_l
            // Possibly we roll dimension if needed:
            var gcol_roll = try gcol.rollaxis(3, 0, context.stream);
            defer gcol_roll.deinitAsync(context.stream);
            // Now shape might be [K, N, OutC, L]

            var y = try gcol_roll.col2im1d(
                &out_shape,
                k,
                s,
                p,
                d,
                context.stream,
            );
            errdefer y.deinitAsync(context.stream);

            // Add bias
            var b_reshaped = try b.reshape(&.{ 1, b.calcLen(), 1 }, context.stream);
            defer b_reshaped.deinitAsync(context.stream);

            var b_brod = try b_reshaped.broadcastTo(y.base.getShapeConst(), context.stream);
            defer b_brod.deinitAsync(context.stream);

            try y.add(&b_brod, context.stream);

            return y.move();
        }

        /// backward => (gx, gw, gb)
        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar, *TaggedVar }) {
            const x = self.in1.?; // [N, C, L]
            const w = self.in2.?; // [OutC, C, K]

            // (1) gx => conv1dNoBias(gy, w)  [the standard “gradient = conv” for transposed]
            const gx = try conv1DNoBiasEx(
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

            // (2) gw => conv1dgradw(gy, x)
            const gw = try conv1DgradwEx(
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

            // (3) gb => sum(gy) across N, L
            var gb_data = try gy.asUntagged(T).data.sum(
                self.base.context.allocator,
                &.{ 0, 2 },
                true,
                self.base.context.stream,
            );
            errdefer gb_data.deinitAsync(self.base.context.stream);

            const gb_var = try self.base.chain.createVariable(T, gb_data.move(), null);

            return .{ gx, gw, gb_var };
        }
    };
}

/// Helper:
pub fn deconv1D(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar, option: Deconv1D(T).Option) !*TaggedVar {
    return try deconv1DEx(T, x, w, b, option, x.getContext().current_chain.?);
}

pub fn deconv1DEx(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar, option: Deconv1D(T).Option, chain: *Chain) !*TaggedVar {
    return try makefunc(Deconv1D(T), x, w, b, option, chain);
}

pub fn Conv2D(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        in3: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: Option,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const In3 = T;
        pub const Scalar = Option;
        pub const Out = T;

        pub const Option = struct {
            stride: [2]usize,
            padding: [2]usize,
            dilation: [2]usize,
        };

        pub usingnamespace FuncDecorator1scalar3in1out(Self);

        const Self = Conv2D(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T), w: *const GPUTensor(T), b: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            const k_shape = w.base.getShapeConst();
            const kh = k_shape[2];
            const kw = k_shape[3];

            var col =
                try x.im2col(.{ kh, kw }, self.scalar.stride, self.scalar.padding, self.scalar.dilation, context.stream);
            defer col.deinitAsync(context.stream);

            // std.debug.print("col shape: {any}\n", .{col.base.getShapeConst()});
            // std.debug.print("w shape: {any}\n", .{w.base.getShapeConst()});
            var w_reshaped = try w.reshape(&.{ k_shape[0], k_shape[1] * kh * kw }, context.stream); // {1, 4}
            defer w_reshaped.deinitAsync(context.stream);

            var y = try col.tensordot(&w_reshaped, context.allocator, &.{1}, &.{1}, context.stream);
            defer y.deinitAsync(context.stream);

            var b_broad = try b.broadcastTo(y.base.getShapeConst(), context.stream);
            defer b_broad.deinitAsync(context.stream);

            try y.add(&b_broad, context.stream);

            var y_roll = try y.rollaxis(3, 1, context.stream);
            errdefer y_roll.deinitAsync(context.stream);

            return y_roll.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar, *TaggedVar }) {
            const x = self.in1.?;
            const w = self.in2.?;

            const gx = try deconv2DNoBiasEx(
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

            const gw = try conv2DgradwEx(
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

            var gb = try gy.asUntagged(T).data.sum(self.base.context.allocator, &.{ 0, 2, 3 }, true, self.base.context.stream);
            errdefer gb.deinitAsync(self.base.context.stream);

            const gb_v = try self.base.chain.createVariable(T, gb.move(), null);

            return .{ gx, gw, gb_v };
        }
    };
}

pub fn conv2D(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar, option: Conv2D(T).Option) !*TaggedVar {
    return try conv2DEx(T, x, w, b, option, x.getContext().current_chain.?);
}

pub fn conv2DEx(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar, option: Conv2D(T).Option, chain: *Chain) !*TaggedVar {
    return try makefunc(Conv2D(T), x, w, b, option, chain);
}

pub fn Deconv2D(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        in3: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar: Option,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const In3 = T;
        pub const Scalar = Option;
        pub const Out = T;

        pub const Option = struct {
            stride: [2]usize,
            padding: [2]usize,
            dilation: [2]usize,
            outsize: ?[2]usize,
        };

        pub usingnamespace FuncDecorator1scalar3in1out(Self);

        const Self = Deconv2D(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T), weight: *const GPUTensor(T), b: *const GPUTensor(T)) !GPUTensor(T) {
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

            var b_reshaped = try b.reshape(&.{ 1, b.calcLen(), 1, 1 }, context.stream);
            defer b_reshaped.deinitAsync(context.stream);

            var b_brod = try b.broadcastTo(y.base.getShapeConst(), context.stream);
            defer b_brod.deinitAsync(context.stream);

            try y.add(&b_brod, context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar, *TaggedVar }) {
            const x = self.in1.?;
            const w = self.in2.?;

            const gx = try conv2DNoBiasEx(
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

            const gw = try conv2DgradwEx(
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

            var gb = try gy.asUntagged(T).data.sum(self.base.context.allocator, &.{ 0, 2, 3 }, true, self.base.context.stream);
            errdefer gb.deinitAsync(self.base.context.stream);

            const gb_v = try self.base.chain.createVariable(T, gb.move(), null);

            return .{ gx, gw, gb_v };
        }
    };
}

pub fn deconv2D(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar, option: Deconv2D(T).Option) !*TaggedVar {
    return try conv2DEx(T, x, w, b, option, x.getContext().current_chain.?);
}

pub fn deconv2DEx(comptime T: type, x: *TaggedVar, w: *TaggedVar, b: *TaggedVar, option: Deconv2D(T).Option, chain: *Chain) !*TaggedVar {
    return try makefunc(Deconv2D(T), x, w, b, option, chain);
}

pub fn BatchNorm(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar, // x
        in2: ?*TaggedVar, // gamma
        in3: ?*TaggedVar, // beta
        out: ?*TaggedVar,
        scalar: Option,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const In3 = T;
        pub const Scalar = Option;
        pub const Out = T;

        pub const Option = struct {
            running_mean: GPUTensor(T),
            running_variance: GPUTensor(T),
            inv_std: ?GPUTensor(T),
            context: *Context,

            decay: T,
            eps: T,
            train: bool,

            pub fn init(decay: T, eps: T, train: bool, x_shape: []const usize, context: *Context) !Option {
                const new_shape, const new_shape_keepdims = try GPUTensor(T).computeOutShape(context.allocator, x_shape, &.{0}, true);
                defer context.allocator.free(new_shape);
                defer context.allocator.free(new_shape_keepdims);

                var running_mean: GPUTensor(T) = try .initAsync(new_shape_keepdims, context.stream);
                errdefer running_mean.deinitAsync(context.stream);
                try running_mean.fill(0.0, context.stream);

                var running_variance: GPUTensor(T) = try .initAsync(new_shape_keepdims, context.stream);
                errdefer running_variance.deinitAsync(context.stream);
                try running_variance.fill(0.0, context.stream);

                return .{
                    .running_mean = running_mean.move(),
                    .running_variance = running_variance.move(),
                    .decay = decay,
                    .eps = eps,
                    .train = train,
                    .context = context,
                    .inv_std = null,
                };
            }

            pub fn move(self: *Option) Option {
                return .{
                    .running_mean = self.running_mean.move(),
                    .running_variance = self.running_variance.move(),
                    .decay = self.decay,
                    .eps = self.eps,
                    .train = self.train,
                    .context = self.context,
                    .inv_std = if (self.inv_std) |*is| is.move() else null,
                };
            }

            pub fn deinit(self: *Option) void {
                self.running_mean.deinitAsync(self.context.stream);
                self.running_variance.deinitAsync(self.context.stream);
                if (self.inv_std) |*iv| {
                    iv.deinitAsync(self.context.stream);
                }
            }
        };

        pub usingnamespace FuncDecorator1scalar3in1out(Self);

        const Self = BatchNorm(T);

        /// Forward pass: Normalizes input x using batch statistics (training) or running averages (inference)
        pub fn forward(self: *Self, x: *const GPUTensor(T), gamma: *const GPUTensor(T), beta: *const GPUTensor(T)) !GPUTensor(T) {
            const allocator = self.base.context.allocator;
            const stream = self.base.context.stream;

            // Handle 2D or 4D input
            var x_proc: GPUTensor(T) = blk: {
                if (x.base.getShapeConst().len == 4) {
                    const shape = x.base.getShapeConst();
                    const N = shape[0];
                    const C = shape[1];
                    const H = shape[2];
                    const W = shape[3];
                    var x_trans = try x.transposeEx(self.base.context.allocator, &.{ 0, 2, 3, 1 }, stream);
                    defer x_trans.deinitAsync(stream);
                    var x_reshaped = try x_trans.reshape(&.{ N * H * W, C }, stream);
                    errdefer x_reshaped.deinitAsync(stream);

                    break :blk x_reshaped.move();
                } else {
                    break :blk try x.cloneAsync(stream);
                }
            };
            defer x_proc.deinitAsync(stream);

            // // **Running Mean Setup**
            // // Running mean shape: (3,), initialized to [0, 0, 0], requires_grad=false
            // var running_mean_tensor = try GPUTensor(T).initAsync(&[_]usize{ 1, 3 }, stream);
            // defer running_mean_tensor.deinitAsync(stream);
            // try running_mean_tensor.fill(0.0, stream);
            // var running_mean = try chain.createVariable(T, running_mean_tensor.move(), "running_mean");
            // defer running_mean.destroy();

            // // **Running Variance Setup**
            // // Running variance shape: (3,), initialized to [0, 0, 0], requires_grad=false
            // var running_variance_tensor = try GPUTensor(T).initAsync(&[_]usize{ 1, 3 }, stream);
            // defer running_variance_tensor.deinitAsync(stream);
            // try running_variance_tensor.fill(0.0, stream);
            // var running_variance = try chain.createVariable(T, running_variance_tensor.move(), "running_variance");
            // defer running_variance.destroy();

            // **Running Mean Setup**
            // Running mean shape: (3,), initialized to [0, 0, 0], requires_grad=false

            // **Running Variance Setup**
            // Running variance shape: (3,), initialized to [0, 0, 0], requires_grad=false

            // Compute normalized output
            var y: GPUTensor(T) = blk: {
                var xc: GPUTensor(T) = undefined;
                if (self.scalar.train) {
                    // Training mode: Compute batch statistics and update running averages
                    var mean = try x_proc.mean(allocator, &.{0}, true, stream);
                    defer mean.deinitAsync(stream);

                    var variance_biased = try x_proc.varianceBiased(allocator, &.{0}, true, stream);
                    defer variance_biased.deinitAsync(stream);

                    // Compute unbiased variance adjustment
                    const m = @as(T, @floatFromInt(x_proc.calcLen() / gamma.calcLen()));
                    const s = if (m > 1) m - 1 else 1;
                    const adjust = m / s;

                    // Update running_mean
                    try self.scalar.running_mean.scale(self.scalar.decay, stream);
                    var temp_mean = try mean.cloneAsync(stream);
                    defer temp_mean.deinitAsync(stream);
                    try temp_mean.scale(1 - self.scalar.decay, stream);
                    try self.scalar.running_mean.add(&temp_mean, stream);

                    // Update running_variance with unbiased variance
                    try self.scalar.running_variance.scale(self.scalar.decay, stream);
                    var temp_variance = try variance_biased.cloneAsync(stream);
                    defer temp_variance.deinitAsync(stream);
                    try temp_variance.scale(adjust * (1 - self.scalar.decay), stream);
                    try self.scalar.running_variance.add(&temp_variance, stream);

                    // Compute inv_std for normalization and store for backward
                    var inv_std = try variance_biased.cloneAsync(stream);
                    defer inv_std.deinitAsync(stream); // Original inv_std is temporary
                    try inv_std.shift(self.scalar.eps, stream);
                    try inv_std.sqrt(stream);
                    try inv_std.inv(stream);
                    if (self.scalar.inv_std != null) {
                        std.mem.swap(GPUTensor(T), &self.scalar.inv_std.?, &inv_std);
                    } else {
                        self.scalar.inv_std = inv_std.move();
                    }

                    // Normalize: xc = (x - mean) * inv_std

                    var mean_broad = try mean.broadcastTo(x_proc.base.getShapeConst(), stream);
                    defer mean_broad.deinitAsync(stream);

                    var inv_std_broad = try self.scalar.inv_std.?.broadcastTo(x_proc.base.getShapeConst(), stream);
                    defer inv_std_broad.deinitAsync(stream);

                    try x_proc.sub(&mean_broad, stream);
                    try x_proc.product(&inv_std_broad, stream);
                    xc = x_proc.move();
                } else {
                    // Inference mode: Use running averages
                    var inv_std = try self.scalar.running_variance.cloneAsync(stream);
                    defer inv_std.deinitAsync(stream);
                    try inv_std.shift(self.scalar.eps, stream);
                    try inv_std.sqrt(stream);
                    try inv_std.inv(stream);

                    try x_proc.sub(&self.scalar.running_mean, stream);
                    try x_proc.product(&inv_std, stream);
                    xc = x_proc.move();
                }
                defer xc.deinitAsync(stream);

                // Apply gamma and beta with broadcasting
                var gamma_broad = try gamma.broadcastTo(xc.base.getShapeConst(), stream);
                defer gamma_broad.deinitAsync(stream);
                var beta_broad = try beta.broadcastTo(xc.base.getShapeConst(), stream);
                defer beta_broad.deinitAsync(stream);

                var y_temp = try xc.cloneAsync(stream);
                defer y_temp.deinitAsync(stream);
                try y_temp.product(&gamma_broad, stream);
                try y_temp.add(&beta_broad, stream);

                // Reshape back if input was 4D
                if (x.base.getShapeConst().len == 4) {
                    const shape = x.base.getShapeConst();
                    const N = shape[0];
                    const C = shape[1];
                    const H = shape[2];
                    const W = shape[3];
                    var y_reshaped = try y_temp.reshape(&.{ N, H, W, C }, stream);
                    defer y_reshaped.deinitAsync(stream);
                    var y_trans = try y_reshaped.transposeEx(self.base.context.allocator, &.{ 0, 3, 1, 2 }, stream);
                    errdefer y_trans.deinitAsync(stream);

                    break :blk y_trans.move();
                } else {
                    break :blk y_temp.move();
                }
            };

            return y.move();
        }

        /// Backward pass: Computes gradients for x, gamma, and beta
        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar, *TaggedVar }) {
            const allocator = self.base.context.allocator;
            const stream = self.base.context.stream;
            var gy_tensor = gy.asUntagged(T).data;

            // Handle 4D gradient input
            var gy_proc: GPUTensor(T) = blk: {
                if (gy_tensor.base.getShapeConst().len == 4) {
                    const shape = gy_tensor.base.getShapeConst();
                    const N = shape[0];
                    const C = shape[1];
                    const H = shape[2];
                    const W = shape[3];
                    var gy_trans = try gy_tensor.transposeEx(self.base.context.allocator, &.{ 0, 2, 3, 1 }, stream);
                    defer gy_trans.deinitAsync(stream);
                    var gy_reshaped = try gy_trans.reshape(&.{ N * H * W, C }, stream);
                    errdefer gy_reshaped.deinitAsync(stream);
                    break :blk gy_reshaped.move();
                } else {
                    break :blk try gy_tensor.cloneAsync(stream);
                }
            };
            defer gy_proc.deinitAsync(stream);

            var x_tensor = &self.in1.?.asUntagged(T).data;
            var gamma = &self.in2.?.asUntagged(T).data;

            var x_proc: GPUTensor(T) = blk: {
                if (x_tensor.base.getShapeConst().len == 4) {
                    const shape = x_tensor.base.getShapeConst();
                    const N = shape[0];
                    const C = shape[1];
                    const H = shape[2];
                    const W = shape[3];
                    var x_trans = try x_tensor.transposeEx(self.base.context.allocator, &.{ 0, 2, 3, 1 }, stream);
                    defer x_trans.deinitAsync(stream);
                    var x_reshaped = try x_trans.reshape(&.{ N * H * W, C }, stream);
                    errdefer x_reshaped.deinitAsync(stream);
                    break :blk x_reshaped.move();
                } else {
                    break :blk try x_tensor.cloneAsync(stream);
                }
            };
            defer x_proc.deinitAsync(stream);

            const batch_size = @as(T, @floatFromInt(gy_proc.base.getShapeConst()[0]));

            // Compute gradients
            var mean = try x_proc.mean(allocator, &.{0}, true, stream);
            defer mean.deinitAsync(stream);

            var mean_broad = try mean.broadcastTo(x_proc.base.getShapeConst(), stream);
            defer mean_broad.deinitAsync(stream);

            var inv_std_broad = try self.scalar.inv_std.?.broadcastTo(x_proc.base.getShapeConst(), stream);
            defer inv_std_broad.deinitAsync(stream);

            var xc = try x_proc.cloneAsync(stream);
            defer xc.deinitAsync(stream);
            try xc.sub(&mean_broad, stream);
            try xc.product(&inv_std_broad, stream);

            var gbeta = try gy_proc.sum(allocator, &.{0}, true, stream);
            defer gbeta.deinitAsync(stream);

            var xc_gy = try xc.cloneAsync(stream);
            defer xc_gy.deinitAsync(stream);
            try xc_gy.product(&gy_proc, stream);
            var ggamma = try xc_gy.sum(allocator, &.{0}, true, stream);
            defer ggamma.deinitAsync(stream);

            var gx = try gy_proc.cloneAsync(stream);
            defer gx.deinitAsync(stream);
            var gbeta_term = try gbeta.cloneAsync(stream);
            defer gbeta_term.deinitAsync(stream);
            try gbeta_term.scale(1 / batch_size, stream);

            var gbeta_term_broad = try gbeta_term.broadcastTo(x_proc.base.getShapeConst(), stream);
            defer gbeta_term_broad.deinitAsync(stream);

            try gx.sub(&gbeta_term_broad, stream);

            var ggamma_term = try ggamma.cloneAsync(stream);
            defer ggamma_term.deinitAsync(stream);
            try ggamma_term.scale(1 / batch_size, stream);
            var xc_ggamma = try xc.cloneAsync(stream);
            defer xc_ggamma.deinitAsync(stream);

            var ggamma_term_broad = try ggamma_term.broadcastTo(x_proc.base.getShapeConst(), stream);
            defer ggamma_term_broad.deinitAsync(stream);

            try xc_ggamma.product(&ggamma_term_broad, stream);
            try gx.sub(&xc_ggamma, stream);

            var gamma_inv_std = try gamma.cloneAsync(stream);
            defer gamma_inv_std.deinitAsync(stream);

            var inv_std_broad_gamma = try self.scalar.inv_std.?.broadcastTo(gamma_inv_std.base.getShapeConst(), stream);
            defer inv_std_broad_gamma.deinitAsync(stream);

            try gamma_inv_std.product(&inv_std_broad_gamma, stream);

            var gamma_inv_std_broad = try gamma_inv_std.broadcastTo(gx.base.getShapeConst(), stream);
            defer gamma_inv_std_broad.deinitAsync(stream);

            try gx.product(&gamma_inv_std_broad, stream);

            // Reshape gx back if input was 4D
            var gx_final: GPUTensor(T) = blk: {
                if (x_tensor.base.getShapeConst().len == 4) {
                    const shape = x_tensor.base.getShapeConst();
                    const N = shape[0];
                    const C = shape[1];
                    const H = shape[2];
                    const W = shape[3];
                    var gx_reshaped = try gx.reshape(&.{ N, H, W, C }, stream);
                    defer gx_reshaped.deinitAsync(stream);
                    var gx_trans = try gx_reshaped.transposeEx(self.base.context.allocator, &.{ 0, 3, 1, 2 }, stream);
                    errdefer gx_trans.deinitAsync(stream);
                    break :blk gx_trans.move();
                } else {
                    break :blk gx.move();
                }
            };
            defer gx_final.deinitAsync(stream);

            // Create TaggedVar for gradients
            const gx_v = try self.base.chain.createVariable(T, gx_final.move(), null);
            const ggamma_v = try self.base.chain.createVariable(T, ggamma.move(), null);
            const gbeta_v = try self.base.chain.createVariable(T, gbeta.move(), null);

            return .{ gx_v, ggamma_v, gbeta_v };
        }

        /// Cleanup function to deallocate inv_std
        pub fn predestroy(self: *Self) void {
            self.scalar.deinit();
        }
    };
}

pub fn batchNorm(
    comptime T: type,
    x: *TaggedVar,
    gamma: *TaggedVar,
    beta: *TaggedVar,
    option: BatchNorm(T).Option,
) !*TaggedVar {
    return try batchNormEx(T, x, gamma, beta, option, x.getContext().current_chain.?);
}

pub fn batchNormEx(
    comptime T: type,
    x: *TaggedVar,
    gamma: *TaggedVar,
    beta: *TaggedVar,
    option: BatchNorm(T).Option,
    chain: *Chain,
) !*TaggedVar {
    return try makefunc(BatchNorm(T), x, gamma, beta, option, chain);
}

// tests
fn testConv1dForward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    //
    // Example:
    // Input shape: [N=1, C=1, L=4] => [1,2,3,4]
    // Weight shape: [OutC=1, InC=1, K=2] => [1,1]
    // Bias shape: [1] => [0]
    // Option: stride=1, padding=0, dilation=1
    // => Output shape: [1,1,3]
    //
    // The output is a standard 1D convolution with kernel length=2, no padding:
    //   output[0] = (1*1 + 2*1) = 3
    //   output[1] = (2*1 + 3*1) = 5
    //   output[2] = (3*1 + 4*1) = 7
    // So the expected output is [3,5,7].
    //

    const input_shape = &[_]usize{ 1, 1, 4 };
    var input_data = [_]T{ 1, 2, 3, 4 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    defer gpu_input.deinitAsync(stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    const var_input = try chain.createVariable(T, gpu_input.move(), "conv1d_input");
    defer var_input.destroy();

    const weight_shape = &[_]usize{ 1, 1, 2 };
    var weight_data = [_]T{ 1, 1 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    defer gpu_weight.deinitAsync(stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    const var_weight = try chain.createVariable(T, gpu_weight.move(), "conv1d_weight");
    defer var_weight.destroy();

    const bias_shape = &[_]usize{1};
    var bias_data = [_]T{0};
    var gpu_bias = try GPUTensor(T).initAsync(bias_shape, stream);
    defer gpu_bias.deinitAsync(stream);
    try gpu_bias.writeFromHostAsync(&bias_data, 0, stream);
    const var_bias = try chain.createVariable(T, gpu_bias.move(), "conv1d_bias");
    defer var_bias.destroy();

    const option = Conv1D(T).Option{
        .stride = 1,
        .padding = 0,
        .dilation = 1,
    };

    var var_output = try conv1DEx(T, var_input, var_weight, var_bias, option, chain);
    defer var_output.destroy();

    // Check results on host
    var host_output = try var_output.asUntagged(T).data.toHost(allocator, stream);
    defer host_output.deinit(allocator);
    try stream.sync();

    const expected_data = [_]T{ 3, 5, 7 };
    const expected_shape = &[_]usize{ 1, 1, 3 };
    for (host_output.data, expected_data) |got, exp| {
        if (@abs(got - exp) > 1e-6) {
            std.debug.print("Conv1d forward mismatch: got={any}, exp={any}\n", .{ got, exp });
            return error.TestFailed;
        }
    }
    try std.testing.expectEqualSlices(usize, expected_shape, host_output.base.getShapeConst());

    std.debug.print("Conv1d forward test passed.\n", .{});
}

fn testConv1dBackward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    //
    // Same setup as testConv1dForward, but now we do a backward pass:
    //   input=[1,2,3,4]
    //   weight=[1,1], bias=0
    // Then we feed an upstream gradient gy=[1,1,1] (shape [1,1,3]) and
    // check the gradients w.r.t. x, w, and b (including a numeric check for x).
    //

    // Prepare input
    const input_shape = &[_]usize{ 1, 1, 4 };
    var input_data = [_]T{ 1, 2, 3, 4 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    const var_input = try chain.createVariable(T, gpu_input.move(), "conv1d_input");
    defer var_input.destroy();

    // weight
    const weight_shape = &[_]usize{ 1, 1, 2 };
    var weight_data = [_]T{ 1, 1 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    const var_weight = try chain.createVariable(T, gpu_weight.move(), "conv1d_weight");
    defer var_weight.destroy();

    // bias
    const bias_shape = &[_]usize{1};
    var bias_data = [_]T{0};
    var gpu_bias = try GPUTensor(T).initAsync(bias_shape, stream);
    try gpu_bias.writeFromHostAsync(&bias_data, 0, stream);
    const var_bias = try chain.createVariable(T, gpu_bias.move(), "conv1d_bias");
    defer var_bias.destroy();

    // forward option
    const option = Conv1D(T).Option{
        .stride = 1,
        .padding = 0,
        .dilation = 1,
    };

    // forward pass
    var var_output = try conv1DEx(T, var_input, var_weight, var_bias, option, chain);
    defer var_output.destroy();

    // Upstream gradient: shape [1,1,3], all ones => [1,1,1]
    const gy_shape = &[_]usize{ 1, 1, 3 };
    var gy_data = [_]T{ 1, 1, 1 };
    var gpu_gy = try GPUTensor(T).initAsync(gy_shape, stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, stream);
    const var_gy = try chain.createVariable(T, gpu_gy.move(), "conv1d_gy");
    defer var_gy.destroy();
    var_output.setGrad(var_gy);

    // backward pass
    try var_output.backwardEx(chain);

    // read out gradients
    const gx_data = var_input.refGrad().?.asUntagged(T).data;
    var host_gx = try gx_data.toHost(allocator, stream);
    defer host_gx.deinit(allocator);

    const gw_data = var_weight.refGrad().?.asUntagged(T).data;
    var host_gw = try gw_data.toHost(allocator, stream);
    defer host_gw.deinit(allocator);

    const gb_data = var_bias.refGrad().?.asUntagged(T).data;
    var host_gb = try gb_data.toHost(allocator, stream);
    defer host_gb.deinit(allocator);

    try stream.sync();

    //------------------------------------------------------------
    // 1) Numeric gradient check for x
    //------------------------------------------------------------
    const epsilon = 1e-4;
    var numerical_gx = try allocator.alloc(T, input_data.len);
    defer allocator.free(numerical_gx);

    for (0..input_data.len) |i| {
        // input_plus
        var input_plus = input_data;
        input_plus[i] += epsilon;
        var gpu_input_plus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_plus.writeFromHostAsync(&input_plus, 0, stream);
        const var_input_plus = try chain.createVariable(T, gpu_input_plus.move(), "input_plus");
        defer var_input_plus.destroy();

        var var_out_plus = try conv1DEx(T, var_input_plus, var_weight, var_bias, option, chain);
        defer var_out_plus.destroy();

        var out_plus_host = try var_out_plus.asUntagged(T).data.toHost(allocator, stream);
        defer out_plus_host.deinit(allocator);

        // input_minus
        var input_minus = input_data;
        input_minus[i] -= epsilon;
        var gpu_input_minus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_minus.writeFromHostAsync(&input_minus, 0, stream);
        const var_input_minus = try chain.createVariable(T, gpu_input_minus.move(), "input_minus");
        defer var_input_minus.destroy();

        var var_out_minus = try conv1DEx(T, var_input_minus, var_weight, var_bias, option, chain);
        defer var_out_minus.destroy();

        var out_minus_host = try var_out_minus.asUntagged(T).data.toHost(allocator, stream);
        defer out_minus_host.deinit(allocator);

        // numeric partial derivative
        numerical_gx[i] = 0;
        for (0..gy_data.len) |j| {
            const diff = out_plus_host.data[j] - out_minus_host.data[j];
            numerical_gx[i] += diff / (2.0 * epsilon) * gy_data[j];
        }
    }

    // compare numeric_gx vs. backprop gx
    for (host_gx.data, numerical_gx) |analytical, numeric| {
        if (@abs(analytical - numeric) > 1e-2) {
            std.debug.print("Conv1d gx mismatch:  got={any}, expect~={any}\n", .{ analytical, numeric });
            return error.TestFailed;
        }
    }

    //------------------------------------------------------------
    // 2) Check weight gradient
    //------------------------------------------------------------
    // With stride=1, kernel_size=2, input=[1,2,3,4], upstream= [1,1,1]:
    //   windows => [ (1,2), (2,3), (3,4) ]
    //   so gw[0] = (1 + 2 + 3) = 6
    //   gw[1] = (2 + 3 + 4) = 9
    const expected_gw = [_]T{ 6, 9 };
    for (host_gw.data, expected_gw) |got, exp| {
        if (@abs(got - exp) > 1e-6) {
            std.debug.print("Conv1d weight grad mismatch: got={any}, exp={any}\n", .{ got, exp });
            return error.TestFailed;
        }
    }

    //------------------------------------------------------------
    // 3) Check bias gradient => sum of gy => 3
    //------------------------------------------------------------
    const expected_gb = [_]T{3.0};
    for (host_gb.data, expected_gb) |got, exp| {
        if (@abs(got - exp) > 1e-6) {
            std.debug.print("Conv1d bias grad mismatch: got={any}, exp={any}\n", .{ got, exp });
            return error.TestFailed;
        }
    }

    std.debug.print("Conv1d backward test passed.\n", .{});
}

fn testDeconv1dForward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    //
    // Example:
    // Input shape: (1,1,2) => [1,2]
    // Weight shape: (1,1,2) => [1,1]
    // Bias shape: (1) => [0]
    // stride=1, padding=0 => outsize = (2-1)*1 + 2 = 3
    // so output shape => (1,1,3).
    //
    // The output for a standard "transposed conv" with kernel=[1,1], input=[1,2]:
    //   input(0)=1 => place [1,1] at output(0..1) => output(0)+=1, output(1)+=1
    //   input(1)=2 => place [2,2] at output(1..2) => output(1)+=2, output(2)+=2
    // => final output => [1, (1+2)=3, 2].
    //

    const input_shape = &[_]usize{ 1, 1, 2 };
    var input_data = [_]T{ 1, 2 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    defer gpu_input.deinitAsync(stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    const var_input = try chain.createVariable(T, gpu_input.move(), "deconv1d_input");
    defer var_input.destroy();

    const weight_shape = &[_]usize{ 1, 1, 2 };
    var weight_data = [_]T{ 1, 1 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    defer gpu_weight.deinitAsync(stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    const var_weight = try chain.createVariable(T, gpu_weight.move(), "deconv1d_weight");
    defer var_weight.destroy();

    const bias_shape = &[_]usize{1};
    var bias_data = [_]T{0};
    var gpu_bias = try GPUTensor(T).initAsync(bias_shape, stream);
    defer gpu_bias.deinitAsync(stream);
    try gpu_bias.writeFromHostAsync(&bias_data, 0, stream);
    const var_bias = try chain.createVariable(T, gpu_bias.move(), "deconv1d_bias");
    defer var_bias.destroy();

    const option = Deconv1D(T).Option{
        .stride = 1,
        .padding = 0,
        .dilation = 1,
        .outsize = 3, // we specify 3 explicitly
    };

    var var_output = try deconv1DEx(T, var_input, var_weight, var_bias, option, chain);
    defer var_output.destroy();

    var host_output = try var_output.asUntagged(T).data.toHost(allocator, stream);
    defer host_output.deinit(allocator);
    try stream.sync();

    const expected_data = [_]T{ 1, 3, 2 };
    const expected_shape = &[_]usize{ 1, 1, 3 };
    for (host_output.data, expected_data) |got, exp| {
        if (@abs(got - exp) > 1e-6) {
            std.debug.print("Deconv1d forward mismatch: got={any}, exp={any}\n", .{ got, exp });
            return error.TestFailed;
        }
    }
    try std.testing.expectEqualSlices(usize, expected_shape, host_output.base.getShapeConst());

    std.debug.print("Deconv1d forward test passed.\n", .{});
}

fn testDeconv1dBackward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    //
    // Same input as testDeconv1dForward => shape(1,1,2) => [1,2],
    // weight(1,1,2) => [1,1], bias=[0], outsize=3.
    // We'll do a backward pass with upstream gradient shape(1,1,3) => [1,1,1],
    // and do a numeric gradient check for x, plus check that bias grad = sum(gy).
    //

    const input_shape = &[_]usize{ 1, 1, 2 };
    var input_data = [_]T{ 1, 2 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    const var_input = try chain.createVariable(T, gpu_input.move(), "deconv1d_input");
    defer var_input.destroy();

    const weight_shape = &[_]usize{ 1, 1, 2 };
    var weight_data = [_]T{ 1, 1 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    const var_weight = try chain.createVariable(T, gpu_weight.move(), "deconv1d_weight");
    defer var_weight.destroy();

    const bias_shape = &[_]usize{1};
    var bias_data = [_]T{0};
    var gpu_bias = try GPUTensor(T).initAsync(bias_shape, stream);
    try gpu_bias.writeFromHostAsync(&bias_data, 0, stream);
    const var_bias = try chain.createVariable(T, gpu_bias.move(), "deconv1d_bias");
    defer var_bias.destroy();

    const option = Deconv1D(T).Option{
        .stride = 1,
        .padding = 0,
        .dilation = 1,
        .outsize = 3,
    };

    // forward
    var var_output = try deconv1DEx(T, var_input, var_weight, var_bias, option, chain);
    defer var_output.destroy();

    // upstream gradient: shape (1,1,3) => [1,1,1]
    const gy_shape = &[_]usize{ 1, 1, 3 };
    var gy_data = [_]T{ 1, 1, 1 };
    var gpu_gy = try GPUTensor(T).initAsync(gy_shape, stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, stream);
    const var_gy = try chain.createVariable(T, gpu_gy.move(), "deconv1d_gy");
    defer var_gy.destroy();

    var_output.setGrad(var_gy);
    try var_output.backwardEx(chain);

    // retrieve gradients
    const gx_data = var_input.refGrad().?.asUntagged(T).data;
    var host_gx = try gx_data.toHost(allocator, stream);
    defer host_gx.deinit(allocator);

    const gw_data = var_weight.refGrad().?.asUntagged(T).data;
    var host_gw = try gw_data.toHost(allocator, stream);
    defer host_gw.deinit(allocator);

    const gb_data = var_bias.refGrad().?.asUntagged(T).data;
    var host_gb = try gb_data.toHost(allocator, stream);
    defer host_gb.deinit(allocator);

    try stream.sync();

    //--------------------------------------------------------
    // Numeric gradient check for x
    //--------------------------------------------------------
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

        var var_out_plus = try deconv1DEx(T, var_input_plus, var_weight, var_bias, option, chain);
        defer var_out_plus.destroy();
        var out_plus_host = try var_out_plus.asUntagged(T).data.toHost(allocator, stream);
        defer out_plus_host.deinit(allocator);

        var input_minus = input_data;
        input_minus[i] -= epsilon;
        var gpu_input_minus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_minus.writeFromHostAsync(&input_minus, 0, stream);
        const var_input_minus = try chain.createVariable(T, gpu_input_minus.move(), "input_minus");
        defer var_input_minus.destroy();

        var var_out_minus = try deconv1DEx(T, var_input_minus, var_weight, var_bias, option, chain);
        defer var_out_minus.destroy();
        var out_minus_host = try var_out_minus.asUntagged(T).data.toHost(allocator, stream);
        defer out_minus_host.deinit(allocator);

        numerical_gx[i] = 0;
        for (0..gy_data.len) |j| {
            const diff = out_plus_host.data[j] - out_minus_host.data[j];
            numerical_gx[i] += diff / (2.0 * epsilon) * gy_data[j];
        }
    }

    for (host_gx.data, numerical_gx) |analytical, numeric| {
        if (@abs(analytical - numeric) > 1e-2) {
            std.debug.print("Deconv1d gx mismatch: got={any}, approx={any}\n", .{ analytical, numeric });
            return error.TestFailed;
        }
    }

    //--------------------------------------------------------
    // Simple check: bias grad is sum of gy => 3
    //--------------------------------------------------------
    const expected_gb = [_]T{3.0};
    for (host_gb.data, expected_gb) |got, exp| {
        if (@abs(got - exp) > 1e-6) {
            std.debug.print("Deconv1d bias grad mismatch: got={any}, exp={any}\n", .{ got, exp });
            return error.TestFailed;
        }
    }

    std.debug.print("Deconv1d backward test passed.\n", .{});
}

fn testConv2dForward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    // Input tensor: (1, 1, 3, 3) [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    const input_shape = &[_]usize{ 1, 1, 3, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    defer gpu_input.deinitAsync(stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Weight tensor: (1, 1, 2, 2) [[[[1, 0], [0, 1]]]]
    const weight_shape = &[_]usize{ 1, 1, 2, 2 };
    var weight_data = [_]T{ 1.0, 0.0, 0.0, 1.0 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    defer gpu_weight.deinitAsync(stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    var var_weight = try chain.createVariable(T, gpu_weight.move(), "weight");
    defer var_weight.destroy();

    // Bias tensor: (1,) [0]
    const bias_shape = &[_]usize{1};
    var bias_data = [_]T{0.0};
    var gpu_bias = try GPUTensor(T).initAsync(bias_shape, stream);
    defer gpu_bias.deinitAsync(stream);
    try gpu_bias.writeFromHostAsync(&bias_data, 0, stream);
    var var_bias = try chain.createVariable(T, gpu_bias.move(), "bias");
    defer var_bias.destroy();

    // Conv2d options
    const option = Conv2D(T).Option{
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
        .dilation = .{ 1, 1 },
    };

    // Perform Conv2d
    var var_output = try conv2DEx(T, var_input, var_weight, var_bias, option, chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected output: (1, 1, 2, 2) [[[[6, 8], [12, 14]]]]
    // Computed manually:
    // [1 0] * [1 2] + [0 1] * [4 5] = 1 + 5 = 6, etc.
    const expected = [_]T{ 6.0, 8.0, 12.0, 14.0 };
    const expected_shape = &[_]usize{ 1, 1, 2, 2 };
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    try std.testing.expectEqualSlices(usize, expected_shape, host_output.base.getShapeConst());

    std.debug.print("Conv2d forward test passed.\n", .{});
}

fn testConv2dBackward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    // Input tensor: (1, 1, 3, 3)
    const input_shape = &[_]usize{ 1, 1, 3, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Weight tensor: (1, 1, 2, 2)
    const weight_shape = &[_]usize{ 1, 1, 2, 2 };
    var weight_data = [_]T{ 1.0, 0.0, 0.0, 1.0 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    var var_weight = try chain.createVariable(T, gpu_weight.move(), "weight");
    defer var_weight.destroy();

    // Bias tensor: (1,)
    const bias_shape = &[_]usize{1};
    var bias_data = [_]T{0.0};
    var gpu_bias = try GPUTensor(T).initAsync(bias_shape, stream);
    try gpu_bias.writeFromHostAsync(&bias_data, 0, stream);
    var var_bias = try chain.createVariable(T, gpu_bias.move(), "bias");
    defer var_bias.destroy();

    // Conv2d options
    const option = Conv2D(T).Option{
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
        .dilation = .{ 1, 1 },
    };

    // Forward pass
    var var_output = try conv2DEx(T, var_input, var_weight, var_bias, option, chain);
    defer var_output.destroy();

    // Upstream gradient (gy): (1, 1, 2, 2) all ones
    const gy_shape = &[_]usize{ 1, 1, 2, 2 };
    var gy_data = [_]T{ 1.0, 1.0, 1.0, 1.0 };
    var gpu_gy = try GPUTensor(T).initAsync(gy_shape, stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, stream);
    var var_gy = try chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_output.setGrad(var_gy);

    // Backward pass
    try var_output.backwardEx(chain);

    // Retrieve gradients
    var gpu_gx = var_input.refGrad().?.asUntagged(T).data;
    var host_gx = try gpu_gx.toHost(allocator, stream);
    defer host_gx.deinit(allocator);

    var gpu_gw = var_weight.refGrad().?.asUntagged(T).data;
    var host_gw = try gpu_gw.toHost(allocator, stream);
    defer host_gw.deinit(allocator);

    var gpu_gb = var_bias.refGrad().?.asUntagged(T).data;
    var host_gb = try gpu_gb.toHost(allocator, stream);
    defer host_gb.deinit(allocator);

    try stream.sync();

    // Numerical gradient computation for x
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
        var var_out_plus = try conv2DEx(T, var_input_plus, var_weight, var_bias, option, chain);
        defer var_out_plus.destroy();
        var host_out_plus = try var_out_plus.asUntagged(T).data.toHost(allocator, stream);
        defer host_out_plus.deinit(allocator);

        var input_minus = input_data;
        input_minus[i] -= epsilon;
        var gpu_input_minus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_minus.writeFromHostAsync(&input_minus, 0, stream);
        const var_input_minus = try chain.createVariable(T, gpu_input_minus.move(), "input_minus");
        defer var_input_minus.destroy();
        var var_out_minus = try conv2DEx(T, var_input_minus, var_weight, var_bias, option, chain);
        defer var_out_minus.destroy();
        var host_out_minus = try var_out_minus.asUntagged(T).data.toHost(allocator, stream);
        defer host_out_minus.deinit(allocator);

        numerical_gx[i] = 0.0;
        for (0..host_out_plus.data.len) |j| {
            numerical_gx[i] += (host_out_plus.data[j] - host_out_minus.data[j]) / (2.0 * epsilon) * gy_data[j];
        }
    }

    // Compare gradients for x
    for (host_gx.data, numerical_gx) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    }

    // Expected gw: sum of gy over sliding windows, simplified check
    const expected_gw = [_]T{ 12.0, 16.0, 24.0, 28.0 }; // Manual computation
    for (host_gw.data, expected_gw) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    // Expected gb: sum of gy
    const expected_gb = [_]T{4.0};
    for (host_gb.data, expected_gb) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    std.debug.print("Conv2d backward test passed.\n", .{});
}

fn testDeconv2dForward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    // Input tensor: (1, 1, 2, 2) [[[[1, 2], [3, 4]]]]
    const input_shape = &[_]usize{ 1, 1, 2, 2 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    defer gpu_input.deinitAsync(stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Weight tensor: (1, 1, 2, 2) [[[[1, 0], [0, 1]]]]
    const weight_shape = &[_]usize{ 1, 1, 2, 2 };
    var weight_data = [_]T{ 1.0, 0.0, 0.0, 1.0 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    defer gpu_weight.deinitAsync(stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    var var_weight = try chain.createVariable(T, gpu_weight.move(), "weight");
    defer var_weight.destroy();

    // Bias tensor: (1,) [0]
    const bias_shape = &[_]usize{1};
    var bias_data = [_]T{0.0};
    var gpu_bias = try GPUTensor(T).initAsync(bias_shape, stream);
    defer gpu_bias.deinitAsync(stream);
    try gpu_bias.writeFromHostAsync(&bias_data, 0, stream);
    var var_bias = try chain.createVariable(T, gpu_bias.move(), "bias");
    defer var_bias.destroy();

    // Deconv2d options
    const option = Deconv2D(T).Option{
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
        .dilation = .{ 1, 1 },
        .outsize = .{ 3, 3 },
    };

    // Perform Deconv2d
    var var_output = try deconv2DEx(T, var_input, var_weight, var_bias, option, chain);
    defer var_output.destroy();

    var gpu_output = var_output.asUntagged(T).data;
    var host_output = try gpu_output.toHost(allocator, stream);
    defer host_output.deinit(allocator);

    try stream.sync();

    // Expected output: (1, 1, 3, 3) [[[[1, 2, 0], [3, 5, 2], [0, 3, 4]]]]
    const expected = [_]T{ 1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0 };
    const expected_shape = &[_]usize{ 1, 1, 3, 3 };
    // std.debug.print("{any} {any}", .{ host_output.data, expected });
    for (host_output.data, expected) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    try std.testing.expectEqualSlices(usize, expected_shape, host_output.base.getShapeConst());

    std.debug.print("Deconv2d forward test passed.\n", .{});
}

fn testDeconv2dBackward(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain) !void {
    const T = f32;

    // Input tensor: (1, 1, 2, 2)
    const input_shape = &[_]usize{ 1, 1, 2, 2 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // Weight tensor: (1, 1, 2, 2)
    const weight_shape = &[_]usize{ 1, 1, 2, 2 };
    var weight_data = [_]T{ 1.0, 0.0, 0.0, 1.0 };
    var gpu_weight = try GPUTensor(T).initAsync(weight_shape, stream);
    try gpu_weight.writeFromHostAsync(&weight_data, 0, stream);
    var var_weight = try chain.createVariable(T, gpu_weight.move(), "weight");
    defer var_weight.destroy();

    // Bias tensor: (1,)
    const bias_shape = &[_]usize{1};
    var bias_data = [_]T{0.0};
    var gpu_bias = try GPUTensor(T).initAsync(bias_shape, stream);
    try gpu_bias.writeFromHostAsync(&bias_data, 0, stream);
    var var_bias = try chain.createVariable(T, gpu_bias.move(), "bias");
    defer var_bias.destroy();

    // Deconv2d options
    const option = Deconv2D(T).Option{
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
        .dilation = .{ 1, 1 },
        .outsize = .{ 3, 3 },
    };

    // Forward pass
    var var_output = try deconv2DEx(T, var_input, var_weight, var_bias, option, chain);
    defer var_output.destroy();

    // Upstream gradient (gy): (1, 1, 3, 3) all ones
    const gy_shape = &[_]usize{ 1, 1, 3, 3 };
    var gy_data = [_]T{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var gpu_gy = try GPUTensor(T).initAsync(gy_shape, stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, stream);
    var var_gy = try chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_output.setGrad(var_gy);

    // Backward pass
    try var_output.backwardEx(chain);

    // Retrieve gradients
    var gpu_gx = var_input.refGrad().?.asUntagged(T).data;
    var host_gx = try gpu_gx.toHost(allocator, stream);
    defer host_gx.deinit(allocator);

    var gpu_gw = var_weight.refGrad().?.asUntagged(T).data;
    var host_gw = try gpu_gw.toHost(allocator, stream);
    defer host_gw.deinit(allocator);

    var gpu_gb = var_bias.refGrad().?.asUntagged(T).data;
    var host_gb = try gpu_gb.toHost(allocator, stream);
    defer host_gb.deinit(allocator);

    try stream.sync();

    // Numerical gradient computation for x
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
        var var_out_plus = try deconv2DEx(T, var_input_plus, var_weight, var_bias, option, chain);
        defer var_out_plus.destroy();
        var host_out_plus = try var_out_plus.asUntagged(T).data.toHost(allocator, stream);
        defer host_out_plus.deinit(allocator);

        var input_minus = input_data;
        input_minus[i] -= epsilon;
        var gpu_input_minus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_minus.writeFromHostAsync(&input_minus, 0, stream);
        const var_input_minus = try chain.createVariable(T, gpu_input_minus.move(), "input_minus");
        defer var_input_minus.destroy();
        var var_out_minus = try deconv2DEx(T, var_input_minus, var_weight, var_bias, option, chain);
        defer var_out_minus.destroy();
        var host_out_minus = try var_out_minus.asUntagged(T).data.toHost(allocator, stream);
        defer host_out_minus.deinit(allocator);

        numerical_gx[i] = 0.0;
        for (0..host_out_plus.data.len) |j| {
            numerical_gx[i] += (host_out_plus.data[j] - host_out_minus.data[j]) / (2.0 * epsilon) * gy_data[j];
        }
    }

    // Compare gradients for x
    for (host_gx.data, numerical_gx) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-2) return error.TestFailed;
    }

    // Expected gb: sum of gy
    const expected_gb = [_]T{9.0};
    for (host_gb.data, expected_gb) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    std.debug.print("Deconv2d backward test passed.\n", .{});
}

fn testBatchNormForwardBackward2D(allocator: std.mem.Allocator, stream: *Stream, chain: *Chain, context: *Context) !void {
    const T = f32;

    // **Input Setup**
    // Input shape: (N=2, C=3) => [[1, 2, 3], [4, 5, 6]]
    const input_shape = &[_]usize{ 2, 3 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_input = try GPUTensor(T).initAsync(input_shape, stream);
    defer gpu_input.deinitAsync(stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, stream);
    var var_input = try chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    // **Gamma Setup**
    // Gamma shape: (3,) => [1, 1, 1]
    const gamma_shape = &[_]usize{ 1, 3 };
    var gamma_data = [_]T{ 1.0, 1.0, 1.0 };
    var gpu_gamma = try GPUTensor(T).initAsync(gamma_shape, stream);
    defer gpu_gamma.deinitAsync(stream);
    try gpu_gamma.writeFromHostAsync(&gamma_data, 0, stream);
    var var_gamma = try chain.createVariable(T, gpu_gamma.move(), "gamma");
    defer var_gamma.destroy();

    // **Beta Setup**
    // Beta shape: (3,) => [0, 0, 0]
    const beta_shape = &[_]usize{ 1, 3 };
    var beta_data = [_]T{ 0.0, 0.0, 0.0 };
    var gpu_beta = try GPUTensor(T).initAsync(beta_shape, stream);
    defer gpu_beta.deinitAsync(stream);
    try gpu_beta.writeFromHostAsync(&beta_data, 0, stream);
    var var_beta = try chain.createVariable(T, gpu_beta.move(), "beta");
    defer var_beta.destroy();

    // **BatchNorm Options**
    var option: BatchNorm(T).Option = try .init(0.9, 1e-5, true, input_shape, context);
    errdefer option.deinit();

    // **Forward Pass**
    var var_output = try batchNormEx(T, var_input, var_gamma, var_beta, option.move(), chain);
    defer var_output.destroy();

    // **Check Output**
    var host_output = try var_output.asUntagged(T).data.toHost(allocator, stream);
    defer host_output.deinit(allocator);
    try stream.sync();

    // **Upstream Gradient for Backward Pass**
    // Shape: (2, 3), all ones => [1, 1, 1, 1, 1, 1]
    const gy_shape = &[_]usize{ 2, 3 };
    var gy_data = [_]T{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    var gpu_gy = try GPUTensor(T).initAsync(gy_shape, stream);
    defer gpu_gy.deinitAsync(stream);
    try gpu_gy.writeFromHostAsync(&gy_data, 0, stream);
    var var_gy = try chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_output.setGrad(var_gy);

    // **Backward Pass**
    try var_output.backwardEx(chain);

    // **Retrieve Gradients**
    var gpu_gx = var_input.refGrad().?.asUntagged(T).data;
    var host_gx = try gpu_gx.toHost(allocator, stream);
    defer host_gx.deinit(allocator);

    var gpu_ggamma = var_gamma.refGrad().?.asUntagged(T).data;
    var host_ggamma = try gpu_ggamma.toHost(allocator, stream);
    defer host_ggamma.deinit(allocator);

    var gpu_gbeta = var_beta.refGrad().?.asUntagged(T).data;
    var host_gbeta = try gpu_gbeta.toHost(allocator, stream);
    defer host_gbeta.deinit(allocator);

    try stream.sync();

    // **Check Gradient w.r.t. Beta**
    // Expected: sum(gy, axis=0) = [2, 2, 2]
    const expected_gbeta = [_]T{ 2.0, 2.0, 2.0 };
    for (host_gbeta.data, expected_gbeta) |got, exp| {
        if (@abs(got - exp) > 1e-6) {
            std.debug.print("gbeta mismatch: got={any}, exp={any}\n", .{ got, exp });
            return error.TestFailed;
        }
    }

    // **Check Gradient w.r.t. Gamma**
    // Expected: sum(x_normalized * gy, axis=0) = [-1*1 + 1*1, ...] = [0, 0, 0]
    const expected_ggamma = [_]T{ 0.0, 0.0, 0.0 };
    for (host_ggamma.data, expected_ggamma) |got, exp| {
        if (@abs(got - exp) > 1e-6) {
            std.debug.print("ggamma mismatch: got={any}, exp={any}\n", .{ got, exp });
            return error.TestFailed;
        }
    }

    // **Numerical Gradient Check for Input (x)**
    const epsilon = 1e-4;
    var numerical_gx = try allocator.alloc(T, input_data.len);
    defer allocator.free(numerical_gx);

    for (0..input_data.len) |i| {
        // Perturb input positively
        var option1: BatchNorm(T).Option = try .init(0.9, 1e-5, true, input_shape, context);
        errdefer option1.deinit();

        var input_plus = input_data;
        input_plus[i] += epsilon;
        var gpu_input_plus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_plus.writeFromHostAsync(&input_plus, 0, stream);
        const var_input_plus = try chain.createVariable(T, gpu_input_plus.move(), "input_plus");
        defer var_input_plus.destroy();
        var var_out_plus = try batchNormEx(T, var_input_plus, var_gamma, var_beta, option1, chain);
        defer var_out_plus.destroy();
        var host_out_plus = try var_out_plus.asUntagged(T).data.toHost(allocator, stream);
        defer host_out_plus.deinit(allocator);

        // Perturb input negatively
        var option2: BatchNorm(T).Option = try .init(0.9, 1e-5, true, input_shape, context);
        errdefer option2.deinit();

        var input_minus = input_data;
        input_minus[i] -= epsilon;
        var gpu_input_minus = try GPUTensor(T).initAsync(input_shape, stream);
        try gpu_input_minus.writeFromHostAsync(&input_minus, 0, stream);
        const var_input_minus = try chain.createVariable(T, gpu_input_minus.move(), "input_minus");
        defer var_input_minus.destroy();
        var var_out_minus = try batchNormEx(T, var_input_minus, var_gamma, var_beta, option2, chain);
        defer var_out_minus.destroy();
        var host_out_minus = try var_out_minus.asUntagged(T).data.toHost(allocator, stream);
        defer host_out_minus.deinit(allocator);

        // Compute numerical gradient
        numerical_gx[i] = 0.0;
        for (0..gy_data.len) |j| {
            const diff = host_out_plus.data[j] - host_out_minus.data[j];
            numerical_gx[i] += diff / (2.0 * epsilon) * gy_data[j];
        }
    }

    // Compare analytical and numerical gradients for x
    for (host_gx.data, numerical_gx) |analytical, numerical| {
        if (@abs(analytical - numerical) > 1e-2) {
            std.debug.print("gx mismatch: analytical={any}, numerical={any}\n", .{ analytical, numerical });
            return error.TestFailed;
        }
    }

    std.debug.print("BatchNorm forward and backward test for 2D input passed.\n", .{});
}

pub fn test1scalar3i1o() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
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

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    try testConv1dForward(allocator, &stream, base_chain);
    try testConv1dBackward(allocator, &stream, base_chain);
    try testDeconv1dForward(allocator, &stream, base_chain);
    try testDeconv1dBackward(allocator, &stream, base_chain);

    // Test Conv2d forward pass
    try testConv2dForward(allocator, &stream, base_chain);

    // Test Conv2d backward pass
    try testConv2dBackward(allocator, &stream, base_chain);

    // Test Deconv2d forward pass
    try testDeconv2dForward(allocator, &stream, base_chain);

    // Test Deconv2d backward pass
    try testDeconv2dBackward(allocator, &stream, base_chain);

    try testBatchNormForwardBackward2D(allocator, &stream, base_chain, &context);

    std.debug.print("All Conv2d and Deconv2d tests passed in test1scalar3i1o.\n", .{});
}
