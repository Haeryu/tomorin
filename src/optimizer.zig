const std = @import("std");
const TaggedVar = @import("variable.zig").TaggedVar;
const tomo = @import("tomo");
const Context = @import("context.zig").Context;
const Chain = @import("chain.zig").Chain;
const GPUTensor = tomo.tensor.GPUTensor;

pub fn Optimizer(comptime Self: type) type {
    return struct {
        pub fn update(self: *Self, params: []const ?*TaggedVar) !void {
            if (@hasDecl(Self, "hooks")) {
                try self.hooks(params);
            }

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
        lr: T,
        context: *Context,

        pub usingnamespace Optimizer(Self);

        const Self = @This();

        pub fn init(lr: T, context: *Context) Self {
            return .{ .lr = lr, .context = context };
        }

        pub fn updateOne(self: *Self, param: *TaggedVar) !void {
            var gp = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer gp.deinitAsync(self.context.stream);

            try gp.scale(self.lr, self.context.stream);
            try param.asUntagged(T).data.sub(&gp, self.context.stream);
        }
    };
}

pub fn MomentumSGD(comptime T: type) type {
    return struct {
        lr: T = 0.01,
        momentum: T = 0.9,
        vs: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),
        context: *Context,

        pub usingnamespace Optimizer(Self);

        const Self = @This();

        pub fn init(lr: T, momentum: T, context: *Context) Self {
            return .{
                .lr = lr,
                .momentum = momentum,
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

                try self.vs.put(param, zeros.move());
            }

            var v = self.vs.getPtr(param).?;
            try v.scale(self.momentum, self.context.stream);

            var gw = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer gw.deinitAsync(self.context.stream);
            try gw.scale(self.lr, self.context.stream);

            try v.sub(&gw, self.context.stream);

            try param.asUntagged(T).data.add(v, self.context.stream);
        }
    };
}

pub fn AdaGrad(comptime T: type) type {
    return struct {
        lr: T = 0.001,
        eps: T = 1e-8,
        hs: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),
        context: *Context,

        pub usingnamespace Optimizer(Self);

        const Self = @This();

        pub fn init(lr: T, eps: T, context: *Context) Self {
            return .{
                .lr = lr,
                .eps = eps,
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

                try self.hs.put(param, zeros.move());
            }

            var h = self.hs.getPtr(param).?;

            var grad_sq = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer grad_sq.deinitAsync(self.context.stream);
            try grad_sq.product(&grad_sq, self.context.stream);

            try h.add(&grad_sq, self.context.stream);

            var grad_lr = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer grad_lr.deinitAsync(self.context.stream);
            try grad_lr.scale(self.lr, self.context.stream);

            var h_sqrt_plus_eps = try h.cloneAsync(self.context.stream);
            defer h_sqrt_plus_eps.deinitAsync(self.context.stream);
            try h_sqrt_plus_eps.sqrt(self.context.stream);
            try h_sqrt_plus_eps.shift(self.eps, self.context.stream);

            try grad_lr.divide(&h_sqrt_plus_eps, self.context.stream);

            try param.asUntagged(T).data.sub(&grad_lr, self.context.stream);
        }
    };
}

pub fn AdaDelta(comptime T: type) type {
    return struct {
        rho: T = 0.95,
        eps: T = 1e-6,
        msg: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),
        msdx: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),
        context: *Context,

        pub usingnamespace Optimizer(Self);

        const Self = @This();

        pub fn init(rho: T, eps: T, context: *Context) Self {
            return .{
                .rho = rho,
                .eps = eps,
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
                try self.msg.put(param, zeros_msg.move());

                var zeros_msdx = try GPUTensor(T).initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros_msdx.deinitAsync(self.context.stream);
                try self.msdx.put(param, zeros_msdx.move());
            }

            var msg = self.msg.getPtr(param).?;
            var msdx = self.msdx.getPtr(param).?;

            // Clone the gradient for computation
            var grad = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer grad.deinitAsync(self.context.stream);

            // Update msg: msg = rho * msg + (1 - rho) * grad^2
            try msg.scale(self.rho, self.context.stream);
            var grad_sq = try grad.cloneAsync(self.context.stream);
            defer grad_sq.deinitAsync(self.context.stream);
            try grad_sq.product(&grad, self.context.stream); // grad^2
            try grad_sq.scale(1 - self.rho, self.context.stream);
            try msg.add(&grad_sq, self.context.stream);

            // Compute dx: dx = sqrt((msdx + eps) / (msg + eps)) * grad
            var msdx_eps = try msdx.cloneAsync(self.context.stream);
            defer msdx_eps.deinitAsync(self.context.stream);
            try msdx_eps.shift(self.eps, self.context.stream); // msdx + eps

            var msg_eps = try msg.cloneAsync(self.context.stream);
            defer msg_eps.deinitAsync(self.context.stream);
            try msg_eps.shift(self.eps, self.context.stream); // msg + eps

            var ratio = try msdx_eps.cloneAsync(self.context.stream);
            defer ratio.deinitAsync(self.context.stream);
            try ratio.divide(&msg_eps, self.context.stream); // (msdx + eps) / (msg + eps)
            try ratio.sqrt(self.context.stream); // sqrt(...)
            var dx = try ratio.cloneAsync(self.context.stream);
            defer dx.deinitAsync(self.context.stream);
            try dx.product(&grad, self.context.stream); // dx = sqrt(...) * grad

            // Update msdx: msdx = rho * msdx + (1 - rho) * dx^2
            try msdx.scale(self.rho, self.context.stream);
            var dx_sq = try dx.cloneAsync(self.context.stream);
            defer dx_sq.deinitAsync(self.context.stream);
            try dx_sq.product(&dx, self.context.stream); // dx^2
            try dx_sq.scale(1 - self.rho, self.context.stream);
            try msdx.add(&dx_sq, self.context.stream);

            // Update parameter: param -= dx
            try param.asUntagged(T).data.sub(&dx, self.context.stream);
        }
    };
}

pub fn Adam(comptime T: type) type {
    return struct {
        t: usize,
        alpha: T = 0.001,
        beta1: T = 0.9,
        beta2: T = 0.999,
        eps: T = 1e-8,
        ms: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),
        vs: std.AutoArrayHashMap(*TaggedVar, GPUTensor(T)),
        context: *Context,

        pub usingnamespace Optimizer(Self);

        const Self = @This();

        pub fn init(alpha: T, beta1: T, beta2: T, eps: T, context: *Context) Self {
            return .{
                .t = 0,
                .alpha = alpha,
                .beta1 = beta1,
                .beta2 = beta2,
                .eps = eps,
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
            const fix1 = 1.0 - std.math.pow(T, self.beta1, @floatFromInt(self.t));
            const fix2 = 1.0 - std.math.pow(T, self.beta2, @floatFromInt(self.t));
            return self.alpha * @sqrt(fix2) / fix1;
        }

        pub fn updateOne(self: *Self, param: *TaggedVar) !void {
            // Initialize moment tensors (m and v) if they don’t exist for this parameter
            if (!self.ms.contains(param)) {
                var zeros_m: GPUTensor(T) = try GPUTensor(T).initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros_m.deinitAsync(self.context.stream);
                try self.ms.put(param, zeros_m.move());

                var zeros_v: GPUTensor(T) = try GPUTensor(T).initAsync(param.asUntagged(T).data.base.getShape(), self.context.stream);
                errdefer zeros_v.deinitAsync(self.context.stream);
                try self.vs.put(param, zeros_v.move());
            }

            // Retrieve moment tensors and gradient
            var m = self.ms.getPtr(param).?;
            var v = self.vs.getPtr(param).?;
            var grad = try param.refGrad().?.asUntagged(T).data.cloneAsync(self.context.stream);
            defer grad.deinitAsync(self.context.stream);

            // Update first moment: m = beta1 * m + (1 - beta1) * grad
            try m.scale(self.beta1, self.context.stream);
            var temp_m = try grad.cloneAsync(self.context.stream);
            defer temp_m.deinitAsync(self.context.stream);
            try temp_m.scale(1 - self.beta1, self.context.stream);
            try m.add(&temp_m, self.context.stream);

            // Update second moment: v = beta2 * v + (1 - beta2) * (grad * grad)
            try v.scale(self.beta2, self.context.stream);
            var grad_sq = try grad.cloneAsync(self.context.stream);
            defer grad_sq.deinitAsync(self.context.stream);
            try grad_sq.product(&grad, self.context.stream); // Element-wise grad * grad
            try grad_sq.scale(1 - self.beta2, self.context.stream);
            try v.add(&grad_sq, self.context.stream);

            // Compute bias correction factors
            const fix1 = 1.0 - std.math.pow(T, self.beta1, @floatFromInt(self.t));
            const fix2 = 1.0 - std.math.pow(T, self.beta2, @floatFromInt(self.t));

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
            try v_sqrt.shift(self.eps, self.context.stream); // sqrt(v_hat) + eps

            var update = try m_hat.cloneAsync(self.context.stream);
            defer update.deinitAsync(self.context.stream);
            try update.divide(&v_sqrt, self.context.stream); // m_hat / (sqrt(v_hat) + eps)
            try update.scale(self.alpha, self.context.stream); // alpha * m_hat / (sqrt(v_hat) + eps)

            // Apply the update: param.data -= update
            try param.asUntagged(T).data.sub(&update, self.context.stream);
        }
    };
}
