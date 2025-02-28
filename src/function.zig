const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;
const VarKey = @import("context.zig").VarKey;
const FuncKey = @import("context.zig").FuncKey;

const Variable = @import("variable.zig").Variable;
const TaggedVar = @import("variable.zig").TaggedVar;
const PTaggedVar = @import("variable.zig").PTaggedVar;

const sliceCast = @import("util.zig").sliceCast;

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        destroy: *const fn (ctx: *anyopaque) void,

        forward: *const fn (ctx: *anyopaque, args: []VarKey) anyerror![]VarKey,

        backward: *const fn (ctx: *anyopaque, args: []VarKey) anyerror![]VarKey,
    };

    pub fn destroy(self: *Function) void {
        self.vtable.destroy(self.ptr);
    }

    pub fn forward(self: *Function, args: []VarKey) ![]VarKey {
        return try self.vtable.forward(self.ptr, args);
    }

    pub fn backward(self: *Function, args: []VarKey) ![]VarKey {
        try self.vtable.backward(self.ptr, args);
    }
};

pub const FunctionBase = struct {
    self_key: FuncKey,
    context: *Context,
    generation: usize = 0,
};

pub usingnamespace @import("function1in1out.zig");
// pub usingnamespace @import("function1in2out.zig");
// pub usingnamespace @import("function2in1out.zig");
// pub usingnamespace @import("function1scalar1in1out.zig");
