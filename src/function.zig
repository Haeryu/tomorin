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
            self.setGeneration();
            self.out = try self.forward(allocator, @ptrCast(@alignCast(x)), cuda_context, stream);
            switch (@typeInfo(@typeInfo(@TypeOf(self.out.?)).pointer.child)) {
                .array => {
                    for (self.out.?) |o| {
                        o.setCreator(function);
                    }
                },
                .@"struct" => {
                    self.out.?.setCreator(function);
                },
                else => unreachable,
            }

            return @ptrCast(self.out.?);
        }

        fn backwardDecorated(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !*anyopaque {
            const self: *Self = @ptrCast(@alignCast(ctx));

            var gy = switch (@typeInfo(@typeInfo(@TypeOf(self.out.?)).pointer.child)) {
                .array => |child| blk: {
                    const len = child.len;
                    var gy_arr: [len]@TypeOf(self.out.?[0].grad.?) = undefined;
                    for (&gy_arr, 0..) |*gy_elem, i| {
                        gy_elem.* = self.out.?[i].grad.?;
                    }
                    break :blk gy_arr;
                },
                .@"struct" => blk: {
                    break :blk self.out.?.grad.?;
                },
                else => unreachable,
            };

            const gy_ptr = switch (@typeInfo(@TypeOf(gy))) {
                .array => &gy,
                .pointer => gy,
                else => unreachable,
            };

            var grad = try self.backward(allocator, gy_ptr, cuda_context, stream);

            switch (@typeInfo(@typeInfo(@TypeOf(grad)).pointer.child)) {
                .array => {
                    for (self.in.?, grad) |i, g| {
                        if (i.grad) |old_grad| {
                            old_grad.destroy(allocator, stream);
                            i.grad = g;
                        } else {
                            if (i.grad) |old_grad| {
                                old_grad.destroy(allocator, stream);
                                i.grad = g;
                            }
                            i.grad = g;
                        }
                    }
                    return @ptrCast(&grad);
                },
                .pointer => {
                    self.in.?.grad = grad;
                    return @ptrCast(grad);
                },
                else => unreachable,
            }
        }

        fn setGeneration(self: *Self) void {
            switch (@typeInfo(@typeInfo(@TypeOf(self.in.?)).pointer.child)) {
                .array => |child| {
                    const len = child.len;

                    var generations: [len]usize = .{0} ** len;
                    for (&generations, self.in.?) |*generation, in| {
                        generation.* = in.generation;
                    }
                    self.generation = std.mem.max(usize, &generations);
                },
                .@"struct" => {
                    self.generation = self.in.?.generation;
                },
                else => unreachable,
            }
        }

        fn getGeneration(ctx: *const anyopaque) usize {
            const self: *const Self = @ptrCast(@alignCast(ctx));
            return self.generation;
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
                    .get_generation = &getGeneration,
                    .destroy = &destroy,
                    //.get_output_grad = &getOutputGrad,
                    .push_inputs_creator = &pushInputsCreator,
                },
            };
        }

        pub fn destroy(ctx: *anyopaque, allocator: std.mem.Allocator, stream: *const Stream) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.out) |out| {
                switch (@typeInfo(@typeInfo(@TypeOf(out)).pointer.child)) {
                    .array => {
                        for (out) |o| {
                            o.destroy(allocator, stream);
                        }
                        allocator.free(out);
                    },
                    .@"struct" => {
                        out.destroy(allocator, stream);
                    },
                    else => unreachable,
                }
                self.out = null;
            }
            allocator.destroy(self);
        }

        // fn getOutputGrad(ctx: *anyopaque) ?*anyopaque {
        //     const self: *Self = @ptrCast(@alignCast(ctx));
        //     return self.out.?.grad;
        // }

        fn pushInputsCreator(ctx: *anyopaque, funcs: *Function.Queue, seen_set: *std.AutoHashMap(*Function, void)) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            switch (@typeInfo(@typeInfo(@TypeOf(self.in.?)).pointer.child)) {
                .array => {
                    for (self.in.?) |in| {
                        if (in.creator) |c| {
                            if (seen_set.get(c) == null) {
                                try funcs.add(c);
                            }
                        }
                    }
                },
                .@"struct" => {
                    if (self.in.?.creator) |c| {
                        if (seen_set.get(c) == null) {
                            try funcs.add(c);
                        }
                    }
                },
                else => unreachable,
            }
        }
    };
}

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const Vtable,

    pub const Queue = std.PriorityQueue(*Function, void, struct {
        fn compare(_: void, a: *Function, b: *Function) std.math.Order {
            return std.math.order(b.getGeneration(), a.getGeneration());
        }
    }.compare);

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
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror!*anyopaque,

        get_generation: *const fn (ctx: *const anyopaque) usize,

        destroy: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, stream: *const Stream) void,

        //  get_output_grad: *const fn (ctx: *anyopaque) ?*anyopaque,

        push_inputs_creator: *const fn (
            ctx: *anyopaque,
            funcs: *Function.Queue,
            seen_set: *std.AutoHashMap(*Function, void),
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
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !*InVariableType {
        return @ptrCast(@alignCast(try self.backwardErased(allocator, cuda_context, stream)));
    }

    pub fn backwardErased(
        self: *const Self,
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !*anyopaque {
        return try self.vtable.backward(self.ptr, allocator, cuda_context, stream);
    }

    pub fn getGeneration(self: *const Self) usize {
        return self.vtable.get_generation(self);
    }

    pub fn destroy(self: *Self, allocator: std.mem.Allocator, stream: *const Stream) void {
        self.vtable.destroy(self.ptr, allocator, stream);
    }

    pub fn getOutputGradErased(self: *const Self) ?*anyopaque {
        return self.vtable.get_output_grad(self.ptr);
    }

    pub fn getOutputGrad(self: *const Self, comptime T: type) *T {
        return @ptrCast(@alignCast(self.getOutputGradErased()));
    }

    pub fn pushInputsCreator(
        self: *const Self,
        funcs: *Function.Queue,
        seen_set: *std.AutoHashMap(*Function, void),
    ) !void {
        try self.vtable.push_inputs_creator(self.ptr, funcs, seen_set);
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
        generation: usize = 0,

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
        generation: usize = 0,

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

pub fn Add(
    comptime T: type,
    comptime rank: comptime_int,
    comptime num: comptime_int,
) type {
    return struct {
        in: ?*[num]*Variable(T, rank) = null,
        out: ?*Variable(T, rank) = null,
        generation: usize = 0,

        const Self = @This();

        pub usingnamespace FunctionBase(Self);

        fn forward(
            _: *Self,
            allocator: std.mem.Allocator,
            xs: *[num]*const Variable(T, rank),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !*Variable(T, rank) {
            var accumulate = try Variable(T, rank).create(allocator, xs[0].data.base.shape, stream);
            errdefer accumulate.destroy(allocator, stream);
            try accumulate.data.fill(if (T == BF16) BF16.fromF32(0.0) else 0.0, stream);

            for (xs) |x| {
                var add = try accumulate.data.add(&x.data, cuda_context, stream);
                defer add.deinitAsync(stream);

                std.mem.swap(@TypeOf(accumulate.data), &accumulate.data, &add);
            }

            return accumulate;
        }

        fn backward(
            _: *Self,
            allocator: std.mem.Allocator,
            gy: *Variable(T, rank),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !*[num]*Variable(T, rank) {
            var clone = try Clone(T, rank, num).create(allocator); // leak
            const res = try clone.forwardErased(allocator, gy, cuda_context, stream);
            return @ptrCast(@alignCast(res));
        }
    };
}

pub fn Clone(
    comptime T: type,
    comptime rank: comptime_int,
    comptime num: comptime_int,
) type {
    return struct {
        in: ?*Variable(T, rank) = null,
        out: ?*[num]*Variable(T, rank) = null,
        generation: usize = 0,

        const Self = @This();

        pub usingnamespace FunctionBase(Self);

        fn forward(
            _: *Self,
            allocator: std.mem.Allocator,
            x: *const Variable(T, rank),
            _: *const CudaContext,
            stream: *const Stream,
        ) !*[num]*Variable(T, rank) {
            const out_slice = try allocator.alloc(*Variable(T, rank), num);
            errdefer allocator.free(out_slice);

            for (out_slice, 0..) |*out, i| {
                errdefer {
                    for (out_slice[0..i]) |o| {
                        o.destroy(allocator, stream);
                    }
                }
                out.* = try x.clone(allocator, stream);
            }

            return @ptrCast(out_slice.ptr);
        }

        fn backward(
            _: *Self,
            allocator: std.mem.Allocator,
            gys: *[num]*Variable(T, rank),
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !*Variable(T, rank) {
            var add = try Add(T, rank, num).create(allocator); // leak
            const res = try add.forwardErased(allocator, @ptrCast(gys), cuda_context, stream);
            return @ptrCast(@alignCast(res));
        }
    };
}
