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
    pfn: Rc(Function, Function.Destructor),

    pub const Queue = std.PriorityQueue(*Function, void, struct {
        fn comp(_: void, a: *Function, b: *Function) std.math.Order {
            return std.math.order(b.generation, a.generation);
        }
    }.comp);

    pub fn forward(
        self: PFunction,
        allocator: std.mem.Allocator,
        pxs_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![Function.max_args_out]?PVarTagged {
        var mutpfn = self.pfn;
        return try mutpfn.get().?.forward(allocator, self.move(), pxs_tagged, cuda_context, stream);
    }

    pub fn backward(
        self: *PFunction,
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !void {
        try self.pfn.get().?.backward(allocator, cuda_context, stream);
    }

    pub fn clone(self: PFunction) PFunction {
        return .{ .pfn = self.pfn.clone() };
    }

    pub fn move(self: PFunction) PFunction {
        var mutself = self;
        return .{ .pfn = mutself.pfn.move() };
    }

    pub fn get(self: *PFunction) ?*Function {
        return self.pfn.get();
    }

    pub fn getConst(self: *const PFunction) ?*const Function {
        return self.pfn.getConst();
    }

    pub fn release(self: *PFunction, allocator: std.mem.Allocator) void {
        self.pfn.release(allocator);
    }

    pub fn addInputsCreators(
        self: *const PFunction,
        queue: *Queue,
        seen_set: *std.AutoHashMap(PFunction, void),
    ) !void {
        try self.pfn.getConst().?.addInputsCreators(queue, seen_set);
    }
};

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    generation: usize,
    pub const max_args_out: comptime_int = 2;

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
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) anyerror!void,

        add_inputs_creators: *const fn (
            ctx: *anyopaque,
            queue: *PFunction.Queue,
            seen_set: *std.AutoHashMap(*const Function, void),
        ) anyerror!void,
    };

    pub fn destroy(self: *Function, allocator: std.mem.Allocator) void {
        self.vtable.destroy(self.ptr, allocator);
    }

    pub fn forward(
        self: *Function,
        allocator: std.mem.Allocator,
        pcreator: PFunction,
        pxs_tagged: []?PVarTagged,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) ![max_args_out]?PVarTagged {
        for (pxs_tagged) |px_tagged| {
            if (px_tagged) |px_tagged_nonnull| {
                if (px_tagged_nonnull.getGeneration() > self.generation) {
                    self.generation = px_tagged_nonnull.getGeneration();
                }
            } else {
                break;
            }
        }
        return try self.vtable.forward(self.ptr, allocator, pcreator, pxs_tagged, cuda_context, stream);
    }

    pub fn backward(
        self: *Function,
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
    ) !void {
        try self.vtable.backward(self.ptr, allocator, cuda_context, stream);
    }

    pub fn addInputsCreators(
        self: *const Function,
        queue: *PFunction.Queue,
        seen_set: *std.AutoHashMap(*const Function, void),
    ) !void {
        try self.vtable.add_inputs_creators(self.ptr, queue, seen_set);
    }

    pub const Destructor = struct {
        allocator: std.mem.Allocator,

        pub fn destroy(self: *Destructor, function: *Function) void {
            function.destroy(self.allocator);
        }
    };
};

pub usingnamespace @import("function1in1out.zig");
// pub usingnamespace @import("function1in2out.zig");
pub usingnamespace @import("function2in1out.zig");
