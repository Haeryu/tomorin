const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;
const FuncKey = @import("context.zig").FuncKey;

const Function = @import("function.zig").Function;

pub fn Variable(comptime T: type) type {
    return struct {
        data: GPUTensor(T),
        name: ?[]const u8 = null,
        generation: usize = 0,
        grad: ?*TaggedVar = null,
        creator: ?FuncKey = null,
        context: *Context,
        protected: bool = false,

        const Self = @This();

        pub fn destroy(self: *Self) void {
            if (self.data.isInvalidated()) {
                return;
            }
            self.data.deinitAsync(self.context.stream);
            self.data.ptr = null;

            if (self.context.options.count_variables) {
                self.context.variable_count -= 1;
            }
            if (self.grad) |grad| {
                grad.release();
            }
        }

        pub fn release(self: *Self) void {
            if (!self.protected) {
                self.destroy();
            }
        }

        pub fn protect(self: *Self) void {
            self.protected = true;
        }

        pub fn unprotect(self: *Self) void {
            self.protected = false;
        }

        pub fn setCreator(self: *Self, creator: FuncKey, creator_generation: usize) void {
            self.creator = creator;
            self.generation = creator_generation + 1;
        }

        pub fn refCreator(self: *const Self) ?*Function {
            return self.self_key.context.refFunction(self.creator orelse return null);
        }

        pub fn refGrad(self: *const Self) ?*TaggedVar {
            return self.grad orelse return null;
        }

        pub fn refGradConst(self: *const Self) ?*const TaggedVar {
            return self.grad orelse return null;
        }

        pub fn setGrad(self: *Self, grad: *TaggedVar) void {
            self.grad = grad;
        }

        pub fn setSelfkey(self: *Self, self_key: *TaggedVar) void {
            self.self_key = self_key;
        }

        pub fn setName(self: *Self, name: []const u8) void {
            self.name = name;
        }

        pub fn getDotAlloc(self: *const Self, type_tag: @typeInfo(TaggedVar).@"union".tag_type.?) ![]u8 {
            const fmt = comptime "{} [label=\"{s}\", color=orange, style=filled]\n";
            const name = self.name orelse "";

            if (self.context.options.verbose_dot) {
                const name_alloc = try std.fmt.allocPrint(self.context.allocator, "{s} {s}\nshape: {any}\nprotected: {}\ngeneration: {}", .{
                    name,
                    @tagName(type_tag),
                    self.data.base.getShapeConst(),
                    self.protected,
                    self.generation,
                });
                defer self.context.allocator.free(name_alloc);

                return try std.fmt.allocPrint(self.context.allocator, fmt, .{
                    @intFromPtr(self),
                    std.mem.trim(u8, name_alloc, " "),
                });
            } else {
                return try std.fmt.allocPrint(self.context.allocator, fmt, .{
                    @intFromPtr(self),
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

    pub fn destroy(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.destroy(),
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

    pub fn getCreator(self: *TaggedVar) ?FuncKey {
        return switch (self.*) {
            inline else => |*v| v.creator,
        };
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

    pub fn setGrad(self: *TaggedVar, grad: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.setGrad(grad),
        }
    }

    pub fn protect(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.protect(),
        }
    }

    pub fn unprotect(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.unprotect(),
        }
    }

    pub fn release(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.release(),
        }
    }

    pub fn getContext(self: *TaggedVar) *Context {
        return switch (self.*) {
            inline else => |*v| v.context,
        };
    }

    pub fn getContextConst(self: *const TaggedVar) *const Context {
        return switch (self.*) {
            inline else => |*v| v.context,
        };
    }
};
