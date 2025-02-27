const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;

const Function = @import("function.zig").Function;

// variable in stack -> owns tensor
// variable out stack (userdata) -> don't own tensor
pub fn Variable(comptime T: type) type {
    return struct {
        data: GPUTensor(T),
        context: *Context,
        name: ?[]u8 = null,
        level: usize = 0,
        grad_index: ?usize = null,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.data.deinitAsync(self.context.stream);
            //self.creator_index = null;
        }
    };
}

pub const TaggedVar = union(enum) {
    bf16: Variable(BF16),
    f16: Variable(f16),
    f32: Variable(f32),
    f64: Variable(f64),

    pub fn init(comptime T: type, variable: Variable(T)) TaggedVar {
        return switch (T) {
            BF16 => .{ .bf16 = variable },
            f16 => .{ .f16 = variable },
            f32 => .{ .f32 = variable },
            f64 => .{ .f64 = variable },
            else => unreachable,
        };
    }

    pub fn deinit(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.deinit(),
        }
    }

    pub fn asUntagged(self: *TaggedVar, comptime T: type) *Variable(T) {
        return switch (T) {
            BF16 => &self.bf16,
            f16 => &self.f16,
            f32 => &self.f32,
            f64 => &self.f64,
            else => unreachable,
        };
    }

    pub fn asUntaggedConst(self: *const TaggedVar, comptime T: type) *const Variable(T) {
        return switch (T) {
            BF16 => &self.bf16,
            f16 => &self.f16,
            f32 => &self.f32,
            f64 => &self.f64,
            else => unreachable,
        };
    }
};

pub const PTaggedVar = union(enum) {
    bf16: *Variable(BF16),
    f16: *Variable(f16),
    f32: *Variable(f32),
    f64: *Variable(f64),

    pub fn init(comptime T: type, variable: *Variable(T)) PTaggedVar {
        return switch (T) {
            BF16 => .{ .bf16 = variable },
            f16 => .{ .f16 = variable },
            f32 => .{ .f32 = variable },
            f64 => .{ .f64 = variable },
            else => unreachable,
        };
    }

    pub fn asUntagged(self: *PTaggedVar, comptime T: type) *Variable(T) {
        return switch (T) {
            BF16 => self.bf16,
            f16 => self.f16,
            f32 => self.f32,
            f64 => self.f64,
            else => unreachable,
        };
    }

    pub fn asUntaggedConst(self: *const PTaggedVar, comptime T: type) *const Variable(T) {
        return switch (T) {
            BF16 => self.bf16,
            f16 => self.f16,
            f32 => self.f32,
            f64 => self.f64,
            else => unreachable,
        };
    }
};
