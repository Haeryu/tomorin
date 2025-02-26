const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;

const PVariable = @import("variable.zig").PVariable;
const PVarTagged = @import("variable.zig").PVarTagged;
const Variable = @import("variable.zig").Variable;
const PVariableWeak = @import("variable.zig").PVariableWeak;

const sliceCast = @import("util.zig").sliceCast;

pub const PFunction = struct {
    pfn: Rc(FunctionErased, FunctionErased.Destructor),

    pub fn forward(
        self: *PFunction,
        allocator: std.mem.Allocator,
        pxs_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![FunctionErased.max_args_out]?PVarTagged {
        return try self.pfn.get().?.forward(allocator, self.clone(), pxs_tagged, cuda_context, stream);
    }

    pub fn backward(
        self: *PFunction,
        allocator: std.mem.Allocator,
        pgys_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![FunctionErased.max_args_out]?PVarTagged {
        return try self.pfn.get().?.backward(allocator, pgys_tagged, cuda_context, stream);
    }

    pub fn clone(self: PFunction) PFunction {
        return .{ .pfn = self.pfn.clone() };
    }

    pub fn move(self: PFunction) PFunction {
        return .{ .pfn = self.pfn.move() };
    }

    pub fn release(self: *PFunction, allocator: std.mem.Allocator) void {
        self.pfn.release(allocator);
    }
};

pub const PFunctionErased = Rc(FunctionErased, FunctionErased.Destructor);

pub const FunctionErased = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const max_args_out: comptime_int = 2;

    const VTable = struct {
        destroy: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
        forward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            pcreator: PFunction,
            pxs_tagged: []?PVarTagged,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror![max_args_out]?PVarTagged,

        backward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            pgys_tagged: []?PVarTagged,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror![FunctionErased.max_args_out]?PVarTagged,
    };

    pub fn destroy(self: *FunctionErased, allocator: std.mem.Allocator) void {
        self.vtable.destroy(self.ptr, allocator);
    }

    pub fn forward(
        self: *FunctionErased,
        allocator: std.mem.Allocator,
        pcreator: PFunction,
        pxs_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![max_args_out]?PVarTagged {
        return try self.vtable.forward(self.ptr, allocator, pcreator, pxs_tagged, cuda_context, stream);
    }

    pub fn backward(
        self: *FunctionErased,
        allocator: std.mem.Allocator,
        pgys_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![max_args_out]?PVarTagged {
        return try self.vtable.backward(self.ptr, allocator, pgys_tagged, cuda_context, stream);
    }

    pub const Destructor = struct {
        allocator: std.mem.Allocator,

        pub fn destroy(self: *Destructor, function: *FunctionErased) void {
            function.destroy(self.allocator);
        }
    };
};

pub fn FuncDecorator1in1out(comptime Self: type) type {
    return struct {
        pub fn create(allocator: std.mem.Allocator) !PFunction {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .in = null,
                .out = null,
            };

            const function: FunctionErased = .{
                .ptr = self,
                .vtable = &.{
                    .forward = &forwardDecorated,
                    .backward = &backwardDecorated,
                    .destroy = &destroy,
                },
            };

            var pfn = try PFunctionErased.create(allocator, function, .{ .allocator = allocator });
            defer pfn.release(allocator);

            return .{ .pfn = pfn.move() };
        }

        pub fn destroy(ctx: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.in) |*in| {
                in.release(allocator);
            }
            self.in = null;
            if (self.out) |*out| {
                out.release(allocator);
            }
            self.out = null;
            allocator.destroy(self);
        }

        pub fn forwardDecorated(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            pcreator: PFunction,
            pxs_tagged: []?PVarTagged,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) ![FunctionErased.max_args_out]?PVarTagged {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in = pxs_tagged[0].?.untagMove(Self.Elem);

            var y = try self.forward(&self.in.?.get().?.data, cuda_context, stream);
            errdefer y.deinitAsync(stream);

            var py = try Variable(Self.Elem).create(allocator, y, stream);
            errdefer py.release(allocator);

            py.get().?.creator = pcreator;

            self.out = py.downgrade();

            return .{PVarTagged.init(Self.Elem, py.move())} ++ .{null} ** (FunctionErased.max_args_out - 1);
        }

        pub fn backwardDecorated(
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            pgys_tagged: []?PVarTagged,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) ![FunctionErased.max_args_out]?PVarTagged {
            const self: *Self = @ptrCast(@alignCast(ctx));
            var gy = pgys_tagged[0].?.untagMove(Self.Elem);

            var gx = try self.backward(&gy.get().?.data, cuda_context, stream);
            errdefer gx.deinitAsync(stream);

            var pgx = try Variable(Self.Elem).create(allocator, gx.move(), stream);
            errdefer pgx.release(allocator);

            return .{PVarTagged.init(Self.Elem, pgx.move())} ++ .{null} ** (FunctionErased.max_args_out - 1);
        }
    };
}

pub fn Exp(comptime T: type) type {
    return struct {
        in: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        const Elem = T;

        pub usingnamespace FuncDecorator1in1out(Exp(T));

        const Self = Exp(T);

        pub fn forward(_: *Self, x: *const GPUTensor(T), _: *const CudaContext, stream: *const Stream) !GPUTensor(T) {
            var y = try x.cloneAsync(stream);
            errdefer y.deinitAsync(stream);

            try y.exp(stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *const GPUTensor(T), _: *const CudaContext, stream: *const Stream) !GPUTensor(T) {
            var gx = try self.in.?.get().?.data.cloneAsync(stream);
            errdefer gx.deinitAsync(stream);

            try gx.product(gy, stream);

            return gx;
        }
    };
}
