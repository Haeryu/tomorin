const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const BF16 = tomo.BF16;
const Variable = @import("variable.zig").Variable;
const std = @import("std");

pub fn FunctionBase(comptime Self: type) type {
    return struct {
        fn forwardDecorated(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            function: *Function,
            x: *anyopaque,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !*anyopaque {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in = @ptrCast(@alignCast(x));
            self.out = try self.forward(allocator, @ptrCast(@alignCast(x)), cuda_context, stream);
            self.out.?.creator = function;
            return @ptrCast(self.out.?);
        }

        fn backwardDecorated(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            gy: *const anyopaque,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !*anyopaque {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const grad = try self.backward(allocator, @ptrCast(@alignCast(gy)), cuda_context, stream);
            self.in.?.grad = grad;
            return @ptrCast(grad);
        }

        pub fn create(allocator: std.mem.Allocator) !Function {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{};

            return .{
                .ptr = @ptrCast(self),
                .vtable = &.{
                    .forward = &forwardDecorated,
                    .backward = &backwardDecorated,
                    .destroy = &destroy,
                    .get_input = &getInput,
                    .get_output = &getOutput,
                    .get_output_grad = &getOutputGrad,
                    .set_input_grad = &setInputGrad,
                    .get_input_creator = &getInputCreator,
                    .set_output_grad_one = &setOutputGradOne,
                },
            };
        }

        pub fn destroy(ctx: *anyopaque, allocator: std.mem.Allocator, stream: *const Stream) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.out) |out| {
                out.destroy(allocator, stream);
            }
            allocator.destroy(self);
        }

        pub fn getInput(ctx: *anyopaque) ?*anyopaque {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.in;
        }

        pub fn getOutput(ctx: *anyopaque) ?*anyopaque {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.out;
        }

        pub fn getOutputGrad(ctx: *anyopaque) ?*anyopaque {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.out.?.grad;
        }

        pub fn setInputGrad(ctx: *anyopaque, grad: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in.?.grad = @ptrCast(@alignCast(grad));
        }

        pub fn getInputCreator(ctx: *anyopaque) ?*Function {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.in.?.creator;
        }

        pub fn setOutputGradOne(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            stream: *const Stream,
        ) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const GradT = @TypeOf(self.out.?.grad.?.*);
            self.out.?.grad = try GradT.create(
                allocator,
                self.out.?.data.base.shape,
                stream,
            );
            errdefer self.out.?.grad.?.destroy(allocator, stream);
            try self.out.?.grad.?.data.fill(
                if (GradT.Elemtype == BF16) BF16.fromF32(1.0) else 1.0,
                stream,
            );
        }
    };
}

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const Vtable,

    const Vtable = struct {
        forward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            function: *Function,
            x: *anyopaque,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror!*anyopaque,

        backward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            gy: *const anyopaque,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror!*anyopaque,

        destroy: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, stream: *const Stream) void,

        get_input: *const fn (ctx: *anyopaque) ?*anyopaque,
        get_output: *const fn (ctx: *anyopaque) ?*anyopaque,
        get_output_grad: *const fn (ctx: *anyopaque) ?*anyopaque,
        set_input_grad: *const fn (ctx: *anyopaque, grad: *anyopaque) void,
        get_input_creator: *const fn (ctx: *anyopaque) ?*Function,
        set_output_grad_one: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            stream: *const Stream,
        ) anyerror!void,
    };

    const Self = @This();

    pub fn forward(
        self: *Self,
        comptime InVariableType: type,
        comptime OutVariableType: type,
        allocator: std.mem.Allocator,
        x: *InVariableType,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !*OutVariableType {
        return @ptrCast(@alignCast(try self.forwardErased(allocator, x, cuda_context, stream)));
    }

    pub fn forwardErased(
        self: *Self,
        allocator: std.mem.Allocator,
        x: *anyopaque,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !*anyopaque {
        return try self.vtable.forward(self.ptr, allocator, self, x, cuda_context, stream);
    }

    pub fn backward(
        self: *const Self,
        comptime InVariableType: type,
        comptime OutVariableType: type,
        allocator: std.mem.Allocator,
        gy: *OutVariableType,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !*InVariableType {
        return @ptrCast(@alignCast(try self.backwardErased(allocator, gy, cuda_context, stream)));
    }

    pub fn backwardErased(
        self: *const Self,
        allocator: std.mem.Allocator,
        gy: *anyopaque,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !*anyopaque {
        return try self.vtable.backward(self.ptr, allocator, gy, cuda_context, stream);
    }

    pub fn destroy(self: *Self, allocator: std.mem.Allocator, stream: *const Stream) void {
        self.vtable.destroy(self.ptr, allocator, stream);
    }

    pub fn getInputErased(self: *const Self) *anyopaque {
        return self.vtable.get_input(self.ptr);
    }

    pub fn getInput(self: *const Self, comptime T: type) *T {
        return @ptrCast(@alignCast(self.getInputErased()));
    }

    pub fn getOutputErased(self: *const Self) *anyopaque {
        return self.vtable.get_output(self.ptr);
    }

    pub fn getOutput(self: *const Self, comptime T: type) *T {
        return @ptrCast(@alignCast(self.getOutputErased()));
    }

    pub fn getOutputGradErased(self: *const Self) ?*anyopaque {
        return self.vtable.get_output_grad(self.ptr);
    }

    pub fn getOutputGrad(self: *const Self, comptime T: type) *T {
        return @ptrCast(@alignCast(self.getOutputGradErased()));
    }

    pub fn setInputGradErased(self: *const Self, grad: *anyopaque) void {
        return self.vtable.set_input_grad(self.ptr, grad);
    }

    pub fn setInputGrad(self: *const Self, comptime T: type, grad: *T) void {
        return @ptrCast(@alignCast(self.setInputGradErased(grad)));
    }

    pub fn getInputCreator(self: *const Self) ?*Function {
        return self.vtable.get_input_creator(self.ptr);
    }

    pub fn setOutputGradOne(
        self: *const Self,
        allocator: std.mem.Allocator,
        stream: *const Stream,
    ) !void {
        return self.vtable.set_output_grad_one(self.ptr, allocator, stream);
    }

    pub fn numericalDiff(
        self: *const Self,
        comptime InVariableType: type,
        comptime OutVariableType: type,
        allocator: std.mem.Allocator,
        x: *const InVariableType,
        eps: InVariableType.Elemtype,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !OutVariableType {
        var x0 = try InVariableType.create(allocator, x.data.base.shape, stream);
        defer x0.destroy(allocator, stream);
        try x0.data.cloneAsync(stream);
        try x0.data.shift(-eps, stream);

        var x1 = try OutVariableType.create(allocator, x.data.base.shape, stream);
        defer x1.destroy(allocator, stream);
        try x1.data.cloneAsync(stream);
        try x1.data.shift(eps, stream);

        var y0 = try self.forward(allocator, &x0, stream);
        errdefer y0.destroy(allocator, stream);

        var y1 = try self.forward(allocator, &x1, stream);
        defer y1.destroy(allocator, stream);

        try y1.data.sub(&y0.data, cuda_context, stream);
        const dx = eps * 2.0;

        try y1.data.scale((if (InVariableType.Elemtype == BF16) BF16.fromF32(1.0) else 1.0) / dx, cuda_context, stream);

        return y1;
    }
};

pub fn Square(
    comptime T: type,
    comptime rank: comptime_int,
) type {
    return struct {
        in: ?*Variable(T, rank) = null,
        out: ?*Variable(T, rank) = null,

        const Self = @This();

        pub usingnamespace FunctionBase(Self);

        fn forward(
            _: *Self,
            allocator: std.mem.Allocator,
            x: *const Variable(T, rank),
            _: *const CudaContext,
            stream: *const Stream,
        ) !*Variable(T, rank) {
            var out = try x.clone(allocator, stream);
            errdefer out.destroy(allocator, stream);

            try out.data.square(stream);
            return out;
        }

        fn backward(
            self: *Self,
            allocator: std.mem.Allocator,
            gy: *const Variable(T, rank),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !*Variable(T, rank) {
            var out = try self.in.?.clone(allocator, stream);
            errdefer out.destroy(allocator, stream);

            try out.data.product(&gy.data, stream);
            var scaled = try out.data.scale(if (T == BF16) BF16.fromF32(2.0) else 2.0, cuda_context, stream);
            defer scaled.deinitAsync(stream);

            std.mem.swap(@TypeOf(scaled), &scaled, &out.data);

            return out;
        }
    };
}

pub fn Exp(
    comptime T: type,
    comptime rank: comptime_int,
) type {
    return struct {
        in: ?*Variable(T, rank) = null,
        out: ?*Variable(T, rank) = null,

        const Self = @This();

        pub usingnamespace FunctionBase(Self);

        fn forward(
            _: *Self,
            allocator: std.mem.Allocator,
            x: *const Variable(T, rank),
            _: *const CudaContext,
            stream: *const Stream,
        ) !*Variable(T, rank) {
            var out = try x.clone(allocator, stream);
            errdefer out.destroy(allocator, stream);

            try out.data.exp(stream);

            return out;
        }

        fn backward(
            self: *Self,
            allocator: std.mem.Allocator,
            gy: *const Variable(T, rank),
            _: *const CudaContext,
            stream: *const Stream,
        ) !*Variable(T, rank) {
            var out = try self.out.?.clone(allocator, stream);
            errdefer out.destroy(allocator, stream);

            try out.data.product(&gy.data, stream);

            return out;
        }
    };
}
