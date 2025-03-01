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

pub fn Variable(comptime T: type) type {
    return struct {
        data: GPUTensor(T),
        name: ?[]const u8 = null,
        generation: usize = 0,
        grad: ?VarKey = null,
        creator: ?FuncKey = null,
        self_key: VarKey,
        refcount: usize,

        const Self = @This();

        pub fn deinit(self: *Self) void {
            self.data.deinitAsync(self.self_key.context.stream);
        }

        pub fn setCreator(self: *Self, creator: FuncKey, creator_generation: usize) void {
            self.creator = creator;
            self.generation = creator_generation + 1;
        }

        pub fn refCreator(self: *const Self) ?*Function {
            return self.self_key.context.refFunction(self.creator orelse return null);
        }

        pub fn refGrad(self: *const Self) ?*TaggedVar {
            return self.self_key.context.refVariable(self.grad orelse return null);
        }

        pub fn refGradConst(self: *const Self) ?*const TaggedVar {
            return self.self_key.context.refVariableConst(self.grad orelse return null);
        }

        pub fn setGrad(self: *Self, grad: VarKey) void {
            self.grad = grad;
        }

        pub fn acquire(self: *Self) void {
            self.refcount += 1;
        }

        pub fn release(self: *Self) void {
            self.refcount -= 1;
        }

        pub fn acquireGrad(self: *Self) void {
            return self.self_key.context.refVariable(self.grad.?).acquire();
        }

        pub fn releaseGrad(self: *Self) void {
            return self.self_key.context.refVariable(self.grad.?).release();
        }

        pub fn setSelfkey(self: *Self, self_key: VarKey) void {
            self.self_key = self_key;
        }

        pub fn isDataNull(self: *const Self) bool {
            return self.data.ptr == null;
        }

        pub fn clearGrad(self: *Self) void {
            self.grad.?.release();
            self.grad = null;
        }

        pub fn setName(self: *Self, name: []const u8) void {
            self.name = name;
        }

        pub fn getDotAlloc(self: *const Self, type_tag: @typeInfo(TaggedVar).@"union".tag_type.?) ![]u8 {
            const fmt = comptime "{} [label=\"{s}\", color=orange, style=filled]\n";
            const name = self.name orelse "";

            if (self.self_key.context.options.verbose_dot) {
                const name_alloc = try std.fmt.allocPrint(self.self_key.context.allocator, "{s} {s}, shape: {any}", .{
                    name,
                    @tagName(type_tag),
                    self.data.base.getShapeConst(),
                });
                defer self.self_key.context.allocator.free(name_alloc);

                return try std.fmt.allocPrint(self.self_key.context.allocator, fmt, .{
                    @intFromPtr(self.self_key.refConst()),
                    std.mem.trim(u8, name_alloc, " "),
                });
            } else {
                return try std.fmt.allocPrint(self.self_key.context.allocator, fmt, .{
                    @intFromPtr(self.self_key.refConst()),
                    std.mem.trim(u8, name, " "),
                });
            }
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

    pub fn getGeneration(self: *const TaggedVar) usize {
        return switch (self.*) {
            inline else => |*v| v.generation,
        };
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

    pub fn setSelfkey(self: *TaggedVar, self_key: VarKey) void {
        switch (self.*) {
            inline else => |*v| v.setSelfkey(self_key),
        }
    }

    pub fn acquire(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.acquire(),
        }
    }

    pub fn release(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.release(),
        }
    }

    pub fn refGrad(self: *TaggedVar) ?*TaggedVar {
        return switch (self.*) {
            inline else => |*v| v.refGrad(),
        };
    }

    pub fn refGradConst(self: *const TaggedVar) ?*const TaggedVar {
        return switch (self.*) {
            inline else => |*v| v.refGradConst(),
        };
    }

    pub fn getRefCount(self: *const TaggedVar) usize {
        return switch (self.*) {
            inline else => |*v| v.refcount,
        };
    }

    pub fn acquireGrad(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.acquireGrad(),
        }
    }

    pub fn releaseGrad(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.releaseGrad(),
        }
    }

    pub fn isDataNull(self: *TaggedVar) bool {
        return switch (self.*) {
            inline else => |*v| v.isDataNull(),
        };
    }

    pub fn getDotAlloc(self: *const TaggedVar) ![]u8 {
        return switch (self.*) {
            inline else => |*v, t| try v.getDotAlloc(t),
        };
    }

    pub fn setName(self: *TaggedVar, name: []const u8) void {
        switch (self.*) {
            inline else => |*v| v.setName(name),
        }
    }

    pub fn setGrad(self: *TaggedVar, grad: VarKey) void {
        switch (self.*) {
            inline else => |*v| v.setGrad(grad),
        }
    }
};
