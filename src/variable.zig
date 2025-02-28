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

const Function = @import("function.zig").Function;

// variable in stack -> owns tensor
// variable out stack (userdata) -> don't own tensor
pub fn Variable(comptime T: type) type {
    return struct {
        data: GPUTensor(T),
        name: ?[]const u8 = null,
        context: *Context,
        generation: usize = 0,
        grad: ?VarKey = null,
        creator: ?FuncKey = null,
        self_key: VarKey,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.data.deinitAsync(self.context.stream);
            //self.creator_index = null;
        }

        pub fn setCreator(self: *Self, creator: FuncKey, creator_generation: usize) void {
            self.creator = creator;
            self.generation = creator_generation + 1;
        }

        pub fn getCreator(self: *const Self) ?*Function {
            return self.context.getFunction(self.creator orelse return null);
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

    pub fn getGeneration(self: *TaggedVar) usize {
        switch (self.*) {
            inline else => |*v| v.generation,
        }
    }

    pub fn getCreator(self: *TaggedVar) FuncKey {
        switch (self.*) {
            inline else => |*v| v.creator,
        }
    }

    pub fn getSelfkey(self: *TaggedVar) VarKey {
        switch (self.*) {
            inline else => |*v| v.var_key,
        }
    }
};
