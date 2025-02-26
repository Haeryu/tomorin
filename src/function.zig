const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;

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
        pxs_tagged: []PVarTagged,
    ) ![Function.max_args_out]?PVarTagged {
        var mutpfn = self.pfn;
        return try mutpfn.get().?.forward(self.move(), pxs_tagged);
    }

    pub fn backward(
        self: *PFunction,
    ) !void {
        try self.pfn.get().?.backward();
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
        destroy: *const fn (ctx: *anyopaque) void,
        forward: *const fn (
            ctx: *anyopaque,
            pcreator: PFunction,
            pxs_tagged: []PVarTagged,
        ) anyerror![max_args_out]?PVarTagged,

        backward: *const fn (
            ctx: *anyopaque,
        ) anyerror!void,

        add_inputs_creators: *const fn (
            ctx: *anyopaque,
            queue: *PFunction.Queue,
            seen_set: *std.AutoHashMap(*const Function, void),
        ) anyerror!void,
    };

    pub fn destroy(self: *Function) void {
        self.vtable.destroy(self.ptr);
    }

    pub fn forward(
        self: *Function,
        pcreator: PFunction,
        pxs_tagged: []PVarTagged,
    ) ![max_args_out]?PVarTagged {
        for (pxs_tagged) |px_tagged| {
            if (px_tagged.getGeneration() > self.generation) {
                self.generation = px_tagged.getGeneration();
            }
        }
        return try self.vtable.forward(self.ptr, pcreator, pxs_tagged);
    }

    pub fn backward(
        self: *Function,
    ) !void {
        try self.vtable.backward(self.ptr);
    }

    pub fn addInputsCreators(
        self: *const Function,
        queue: *PFunction.Queue,
        seen_set: *std.AutoHashMap(*const Function, void),
    ) !void {
        try self.vtable.add_inputs_creators(self.ptr, queue, seen_set);
    }

    pub const Destructor = struct {
        pub fn destroy(_: *Destructor, function: *Function) void {
            function.destroy();
        }
    };
};

pub usingnamespace @import("function1in1out.zig");
// pub usingnamespace @import("function1in2out.zig");
pub usingnamespace @import("function2in1out.zig");
