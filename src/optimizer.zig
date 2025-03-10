const std = @import("std");
const TaggedVar = @import("variable.zig").TaggedVar;
const tomo = @import("tomo");
const Context = @import("context.zig").Context;
const Chain = @import("chain.zig").Chain;
const GPUTensor = tomo.tensor.GPUTensor;

const dbg = @import("util.zig").debugPrintGpuTensor;

pub fn WeightDecay(comptime T: type) type {
    return struct {
        rate: T,
        context: *Context,

        const Self = @This();

        pub fn init(rate: T, context: *Context) Self {
            return .{ .rate = rate, .context = context };
        }

        pub fn call(self: *const Self, params: []const ?*TaggedVar) !void {
            for (params) |param| {
                if (param) |p| {
                    var data = try p.asUntagged(T).data.cloneAsync(self.context.stream);
                    defer data.deinitAsync(self.context.stream);

                    try data.scale(self.rate, self.context.stream);

                    try p.refGrad().?.asUntagged(T).data.add(&data, self.context.stream);
                }
            }
        }
    };
}

pub fn ClipGrad(comptime T: type) type {
    return struct {
        max_norm: T,
        context: *Context,

        const Self = @This();

        pub fn init(max_norm: T, context: *Context) Self {
            return .{
                .max_norm = max_norm,
                .context = context,
            };
        }

        pub fn call(self: *const Self, params: []const ?*TaggedVar) !void {
            var total_norm_device: GPUTensor(T) = try .initAsync(&.{ 1, 1 }, self.context.stream);
            defer total_norm_device.deinitAsync(self.context.stream);
            try total_norm_device.fill(0.0, self.context.stream);

            for (params) |param| {
                if (param) |p| {
                    var grad_sq = try p.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
                    defer grad_sq.deinitAsync(self.context.stream);

                    try grad_sq.square(self.context.stream);

                    var sum = try grad_sq.sum(p.getContext().allocator, &.{}, true, self.context.stream);
                    defer sum.deinitAsync(self.context.stream);

                    var sum_1_1 = try sum.reshape(&.{ 1, 1 }, self.context.stream);
                    defer sum_1_1.deinitAsync(self.context.stream);

                    try total_norm_device.add(&sum_1_1, self.context.stream);
                }
            }

            var total_norm_host = try total_norm_device.toHost(self.context.allocator, self.context.stream);
            defer total_norm_host.deinit(self.context.allocator);

            try self.context.stream.sync();

            const total_norm = @sqrt(total_norm_host.at(&.{ 0, 0 }).*);
            const rate = self.max_norm / (total_norm + 1e-6);

            if (rate >= 1) return;

            for (params) |param| {
                if (param) |p| {
                    try p.refGrad().?.asUntagged(T).data.scale(rate, self.context.stream);
                }
            }
        }
    };
}

pub const FreezeParam = struct {
    freeze_params: []?*TaggedVar,
    context: *Context,

    pub fn init(params_slice: []const []?*TaggedVar, context: *Context) !FreezeParam {
        var params: std.ArrayList(?*TaggedVar) = .init(context.allocator);
        defer params.deinit();

        for (params_slice) |p_slice| {
            try params.appendSlice(p_slice);
        }

        return .{
            .freeze_params = try params.toOwnedSlice(),
            .context = context,
        };
    }

    pub fn deinit(self: *FreezeParam) void {
        self.context.allocator.free(self.freeze_params);
    }

    pub fn call(self: *FreezeParam) !void {
        for (self.freeze_params) |freeze_param| {
            if (freeze_param) |p| {
                p.setGrad(null);
            }
        }
    }
};

// pub fn HookOption(comptime T: type) type {
//     return struct {
//         weight_decay_rate: ?T = null,
//         max_norm: ?T = null,
//         freeze_params: ?[]const []?*TaggedVar = null,
//     };
// }

// pub fn OptimizerBase(comptime T: type) type {
//     return struct {
//         weight_decay: ?WeightDecay(T),
//         clip_grad: ?ClipGrad(T),
//         freeze_param: ?FreezeParam,

//         const Self = @This();

//         pub fn init(option: HookOption(T), context: *Context) !Self {
//             const weight_decay = if (option.weight_decay_rate) |r| WeightDecay(T).init(r, context) else null;
//             const clip_grad = if (option.max_norm) |m| ClipGrad(T).init(m, context) else null;
//             const freeze_param = if (option.freeze_params) |ps| try FreezeParam.init(ps, context) else null;

//             return .{
//                 .weight_decay = weight_decay,
//                 .clip_grad = clip_grad,
//                 .freeze_param = freeze_param,
//             };
//         }

//         pub fn deinit(self: *Self) void {
//             if (self.freeze_param) |*fp| {
//                 fp.deinit();
//             }
//         }
//     };
// }

pub fn Optimizer(comptime Self: type) type {
    return struct {
        pub fn update(self: *Self, params: []const ?*TaggedVar) !void {
            if (@hasDecl(Self, "preupdate")) {
                self.preupdate();
            }

            for (params) |param| {
                if (param) |p| {
                    try self.updateOne(p);
                }
            }
        }
    };
}

pub fn SGD(comptime T: type) type {
    return struct {
        hyper_params: HyperParams,

        context: *Context,
        pub usingnamespace Optimizer(Self);

        pub const HyperParams = struct {
            lr: T,
        };

        const Self = @This();

        pub fn init(hyper_params: HyperParams, context: *Context) !Self {
            return .{
                .hyper_params = hyper_params,
                .context = context,
            };
        }

        pub fn updateOne(self: *Self, param: *TaggedVar) !void {
            var gp = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer gp.deinitAsync(self.context.stream);

            try gp.scale(self.hyper_params.lr, self.context.stream);
            try param.asUntagged(T).data.sub(&gp, self.context.stream);
        }
    };
}

pub fn MomentumSGD(comptime T: type) type {
    return struct {
        hyper_params: HyperParams,
        vs: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),

        context: *Context,
        pub usingnamespace Optimizer(Self);

        pub const HyperParams = struct {
            lr: T = 0.01,
            momentum: T = 0.9,

            pub const default: HyperParams = .{};
        };

        const Self = @This();

        pub fn init(hyper_params: HyperParams, context: *Context) !Self {
            return .{
                .hyper_params = hyper_params,
                .context = context,
                .vs = .init(context.allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            var iter = self.vs.iterator();
            while (iter.next()) |e| {
                e.value_ptr.deinitAsync(self.context.stream);
            }
            self.vs.deinit();
        }

        pub fn updateOne(self: *Self, param: *TaggedVar) !void {
            if (!self.vs.contains(param)) {
                var zeros: GPUTensor(T) = try .initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros.deinitAsync(self.context.stream);

                try zeros.fill(0.0, self.context.stream);

                try self.vs.put(param, zeros.move());
            }

            var v = self.vs.getPtr(param).?;
            try v.scale(self.hyper_params.momentum, self.context.stream);

            var gw = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer gw.deinitAsync(self.context.stream);
            try gw.scale(self.hyper_params.lr, self.context.stream);

            try v.sub(&gw, self.context.stream);

            try param.asUntagged(T).data.add(v, self.context.stream);
        }
    };
}

pub fn AdaGrad(comptime T: type) type {
    return struct {
        hyper_params: HyperParams,
        hs: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),

        context: *Context,

        pub usingnamespace Optimizer(Self);

        pub const HyperParams = struct {
            lr: T = 0.001,
            eps: T = 1e-8,

            pub const default: HyperParams = .{};
        };

        const Self = @This();

        pub fn init(hyper_params: HyperParams, context: *Context) !Self {
            return .{
                .hyper_params = hyper_params,
                .context = context,
                .hs = .init(context.allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            var iter = self.hs.iterator();
            while (iter.next()) |e| {
                e.value_ptr.deinitAsync(self.context.stream);
            }
            self.hs.deinit();
        }

        pub fn updateOne(self: *Self, param: *TaggedVar) !void {
            if (!self.hs.contains(param)) {
                var zeros: GPUTensor(T) = try .initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros.deinitAsync(self.context.stream);

                try zeros.fill(0.0, self.context.stream);

                try self.hs.put(param, zeros.move());
            }

            var h = self.hs.getPtr(param).?;

            var grad_sq = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer grad_sq.deinitAsync(self.context.stream);
            try grad_sq.product(&grad_sq, self.context.stream);

            try h.add(&grad_sq, self.context.stream);

            var grad_lr = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer grad_lr.deinitAsync(self.context.stream);
            try grad_lr.scale(self.hyper_params.lr, self.context.stream);

            var h_sqrt_plus_eps = try h.cloneAsync(self.context.stream);
            defer h_sqrt_plus_eps.deinitAsync(self.context.stream);
            try h_sqrt_plus_eps.sqrt(self.context.stream);
            try h_sqrt_plus_eps.shift(self.hyper_params.eps, self.context.stream);

            try grad_lr.divide(&h_sqrt_plus_eps, self.context.stream);

            try param.asUntagged(T).data.sub(&grad_lr, self.context.stream);
        }
    };
}

pub fn AdaDelta(comptime T: type) type {
    return struct {
        hyper_params: HyperParams,
        msg: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),
        msdx: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),

        context: *Context,
        pub usingnamespace Optimizer(Self);

        pub const HyperParams = struct {
            rho: T = 0.95,
            eps: T = 1e-6,

            pub const default: HyperParams = .{};
        };

        const Self = @This();

        pub fn init(hyper_params: HyperParams, context: *Context) !Self {
            return .{
                .hyper_params = hyper_params,
                .context = context,
                .msg = .init(context.allocator),
                .msdx = .init(context.allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            var msg_iter = self.msg.iterator();
            while (msg_iter.next()) |e| {
                e.value_ptr.deinitAsync(self.context.stream);
            }
            self.msg.deinit();

            var msdx_iter = self.msdx.iterator();
            while (msdx_iter.next()) |e| {
                e.value_ptr.deinitAsync(self.context.stream);
            }
            self.msdx.deinit();
        }

        pub fn updateOne(self: *Self, param: *TaggedVar) !void {
            if (!self.msg.contains(param)) {
                var zeros_msg = try GPUTensor(T).initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros_msg.deinitAsync(self.context.stream);
                try zeros_msg.fill(0.0, self.context.stream);
                try self.msg.put(param, zeros_msg.move());

                var zeros_msdx = try GPUTensor(T).initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros_msdx.deinitAsync(self.context.stream);
                try zeros_msdx.fill(0.0, self.context.stream);
                try self.msdx.put(param, zeros_msdx.move());
            }

            var msg = self.msg.getPtr(param).?;
            var msdx = self.msdx.getPtr(param).?;

            // Clone the gradient for computation
            var grad = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer grad.deinitAsync(self.context.stream);

            // Update msg: msg = rho * msg + (1 - rho) * grad^2
            try msg.scale(self.hyper_params.rho, self.context.stream);
            var grad_sq = try grad.cloneAsync(self.context.stream);
            defer grad_sq.deinitAsync(self.context.stream);
            try grad_sq.product(&grad, self.context.stream); // grad^2
            try grad_sq.scale(1 - self.hyper_params.rho, self.context.stream);
            try msg.add(&grad_sq, self.context.stream);

            // Compute dx: dx = sqrt((msdx + eps) / (msg + eps)) * grad
            var msdx_eps = try msdx.cloneAsync(self.context.stream);
            defer msdx_eps.deinitAsync(self.context.stream);
            try msdx_eps.shift(self.hyper_params.eps, self.context.stream); // msdx + eps

            var msg_eps = try msg.cloneAsync(self.context.stream);
            defer msg_eps.deinitAsync(self.context.stream);
            try msg_eps.shift(self.hyper_params.eps, self.context.stream); // msg + eps

            var ratio = try msdx_eps.cloneAsync(self.context.stream);
            defer ratio.deinitAsync(self.context.stream);
            try ratio.divide(&msg_eps, self.context.stream); // (msdx + eps) / (msg + eps)
            try ratio.sqrt(self.context.stream); // sqrt(...)
            var dx = try ratio.cloneAsync(self.context.stream);
            defer dx.deinitAsync(self.context.stream);
            try dx.product(&grad, self.context.stream); // dx = sqrt(...) * grad

            // Update msdx: msdx = rho * msdx + (1 - rho) * dx^2
            try msdx.scale(self.hyper_params.rho, self.context.stream);
            var dx_sq = try dx.cloneAsync(self.context.stream);
            defer dx_sq.deinitAsync(self.context.stream);
            try dx_sq.product(&dx, self.context.stream); // dx^2
            try dx_sq.scale(1 - self.hyper_params.rho, self.context.stream);
            try msdx.add(&dx_sq, self.context.stream);

            // Update parameter: param -= dx
            try param.asUntagged(T).data.sub(&dx, self.context.stream);
        }
    };
}

pub fn Adam(comptime T: type) type {
    return struct {
        t: usize,
        hyper_params: HyperParams,
        ms: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),
        vs: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),

        context: *Context,

        pub const HyperParams = struct {
            alpha: T = 0.001,
            beta1: T = 0.9,
            beta2: T = 0.999,
            eps: T = 1e-8,

            pub const default: HyperParams = .{};
        };

        pub usingnamespace Optimizer(Self);

        const Self = @This();

        pub fn init(hyper_params: HyperParams, context: *Context) !Self {
            return .{
                .t = 0,
                .hyper_params = hyper_params,
                .context = context,
                .ms = .init(context.allocator),
                .vs = .init(context.allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            var ms_iter = self.ms.iterator();
            while (ms_iter.next()) |e| {
                e.value_ptr.deinitAsync(self.context.stream);
            }
            self.ms.deinit();

            var vs_iter = self.vs.iterator();
            while (vs_iter.next()) |e| {
                e.value_ptr.deinitAsync(self.context.stream);
            }
            self.vs.deinit();
        }

        pub fn preupdate(self: *Self) void {
            self.t += 1;
        }
        pub fn calclr(self: *const Self) T {
            const fix1 = 1.0 - std.math.pow(T, self.hyper_params.beta1, @floatFromInt(self.t));
            const fix2 = 1.0 - std.math.pow(T, self.hyper_params.beta2, @floatFromInt(self.t));
            return self.hyper_params.alpha * @sqrt(fix2) / fix1;
        }

        pub fn updateOne(self: *Self, param: *TaggedVar) !void {
            // Initialize moment tensors (m and v) if they don’t exist for this parameter
            if (!self.ms.contains(param)) {
                var zeros_m: GPUTensor(T) = try GPUTensor(T).initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros_m.deinitAsync(self.context.stream);
                try zeros_m.fill(0.0, self.context.stream);
                try self.ms.put(param, zeros_m.move());

                var zeros_v: GPUTensor(T) = try GPUTensor(T).initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros_v.deinitAsync(self.context.stream);
                try zeros_v.fill(0.0, self.context.stream);
                try self.vs.put(param, zeros_v.move());
            }

            // Retrieve moment tensors and gradient
            var m = self.ms.getPtr(param).?;
            var v = self.vs.getPtr(param).?;
            var grad = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer grad.deinitAsync(self.context.stream);

            // Update first moment: m = beta1 * m + (1 - beta1) * grad
            try m.scale(self.hyper_params.beta1, self.context.stream);
            var temp_m = try grad.cloneAsync(self.context.stream);
            defer temp_m.deinitAsync(self.context.stream);
            try temp_m.scale(1.0 - self.hyper_params.beta1, self.context.stream);

            try m.add(&temp_m, self.context.stream);

            // Update second moment: v = beta2 * v + (1 - beta2) * (grad * grad)
            try v.scale(self.hyper_params.beta2, self.context.stream);
            var grad_sq = try grad.cloneAsync(self.context.stream);
            defer grad_sq.deinitAsync(self.context.stream);
            try grad_sq.product(&grad, self.context.stream); // Element-wise grad * grad
            try grad_sq.scale(1.0 - self.hyper_params.beta2, self.context.stream);
            try v.add(&grad_sq, self.context.stream);

            // Compute bias correction factors
            const fix1 = 1.0 - std.math.pow(T, self.hyper_params.beta1, @floatFromInt(self.t));
            const fix2 = 1.0 - std.math.pow(T, self.hyper_params.beta2, @floatFromInt(self.t));

            // Compute bias-corrected estimates: m_hat and v_hat
            var m_hat = try m.cloneAsync(self.context.stream);
            defer m_hat.deinitAsync(self.context.stream);
            try m_hat.scale(1.0 / fix1, self.context.stream); // m_hat = m / (1 - beta1^t)

            var v_hat = try v.cloneAsync(self.context.stream);
            defer v_hat.deinitAsync(self.context.stream);
            try v_hat.scale(1.0 / fix2, self.context.stream); // v_hat = v / (1 - beta2^t)

            // Compute the update: alpha * m_hat / (sqrt(v_hat) + eps)
            var v_sqrt = try v_hat.cloneAsync(self.context.stream);
            defer v_sqrt.deinitAsync(self.context.stream);
            try v_sqrt.sqrt(self.context.stream); // sqrt(v_hat)
            try v_sqrt.shift(self.hyper_params.eps, self.context.stream); // sqrt(v_hat) + eps

            var update = try m_hat.cloneAsync(self.context.stream);
            defer update.deinitAsync(self.context.stream);
            try update.divide(&v_sqrt, self.context.stream); // m_hat / (sqrt(v_hat) + eps)
            try update.scale(self.hyper_params.alpha, self.context.stream); // alpha * m_hat / (sqrt(v_hat) + eps)

            try self.context.stream.sync();

            // Apply the update: param.data -= update
            try param.asUntagged(T).data.sub(&update, self.context.stream);
        }
    };
}
