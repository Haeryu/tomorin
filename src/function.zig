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

pub const PFunction = Rc(Function, Function.Destructor);

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const max_args: comptime_int = 2;

    const VTable = struct {
        destroy: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
        forward: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            pxs_tagged: []?PVarTagged,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror![max_args]?PVarTagged,
    };

    pub fn destroy(self: *Function, allocator: std.mem.Allocator) void {
        self.vtable.destroy(self.ptr, allocator);
    }

    pub fn forward(
        self: *Function,
        allocator: std.mem.Allocator,
        pxs_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![max_args]?PVarTagged {
        return try self.vtable.forward(self.ptr, allocator, pxs_tagged, cuda_context, stream);
    }

    pub const Destructor = struct {
        allocator: std.mem.Allocator,

        pub fn destroy(self: *Destructor, function: *Function) void {
            function.destroy(self.allocator);
        }
    };
};

pub fn FuncDecorator1(comptime Self: type) type {
    return struct {
        pub fn create(allocator: std.mem.Allocator) !PFunction {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.* = .{
                .in = null,
                .out = null,
            };

            const function: Function = .{
                .ptr = self,
                .vtable = &.{
                    .forward = &forwardDecorated,
                    .destroy = &destroy,
                },
            };

            var pfn = try PFunction.create(allocator, function, .{ .allocator = allocator });
            defer pfn.release(allocator);

            return pfn.move();
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
            pxs_opaque: []?PVarTagged,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) ![Function.max_args]?PVarTagged {
            std.debug.assert(pxs_opaque.len == 1 or (pxs_opaque.len == 2 and pxs_opaque[1] == null));
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in = pxs_opaque[0].?.untagMove(Self.Elem);

            var y = try self.forward(&self.in.?.get().?.data, cuda_context, stream);
            errdefer y.deinitAsync(stream);

            var py = try Variable(Self.Elem).createTagged(allocator, y, stream);
            errdefer py.release(allocator);

            return .{py} ++ .{null} ** (Function.max_args - 1);
        }
    };
}

pub fn Exp(comptime T: type) type {
    return struct {
        in: ?PVariable(T) = null,
        out: ?PVariableWeak(T) = null,
        const Elem = T;

        pub usingnamespace FuncDecorator1(Exp(T));

        const Self = Exp(T);

        pub fn forward(_: *Self, x: *const GPUTensor(T), _: *const CudaContext, stream: *const Stream) !GPUTensor(T) {
            var y = try x.cloneAsync(stream);
            errdefer y.deinitAsync(stream);

            try y.exp(stream);

            return y;
        }
    };
}
