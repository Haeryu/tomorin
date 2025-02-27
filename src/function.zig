const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;

const Variable = @import("variable.zig").Variable;
const TaggedVar = @import("variable.zig").TaggedVar;
const PTaggedVar = @import("variable.zig").PTaggedVar;

const sliceCast = @import("util.zig").sliceCast;

pub const Function = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        destroy: *const fn (ctx: *anyopaque) void,

        forward: *const fn (ctx: *anyopaque, args: []*TaggedVar) anyerror![]*TaggedVar,

        backward: *const fn (ctx: *anyopaque, args: []PTaggedVar) anyerror![]PTaggedVar,
    };

    pub fn destroy(self: *Function) void {
        self.vtable.destroy(self.ptr);
    }

    pub fn forward(self: *Function, args: []*TaggedVar) ![]*TaggedVar {
        return try self.vtable.forward(self.ptr, args);
    }

    pub fn backward(self: *Function, args: []*TaggedVar) ![]*TaggedVar {
        try self.vtable.backward(self.ptr, args);
    }
};

pub usingnamespace @import("function1in1out.zig");
// pub usingnamespace @import("function1in2out.zig");
// pub usingnamespace @import("function2in1out.zig");
// pub usingnamespace @import("function1scalar1in1out.zig");
