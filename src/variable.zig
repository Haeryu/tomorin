const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;
const Chain = @import("chain.zig").Chain;

const Function = @import("function.zig").Function;

const constSliceCast = @import("util.zig").constSliceCast;

pub fn Variable(comptime T: type) type {
    return struct {
        data: GPUTensor(T),
        name: ?[]const u8 = null,
        generation: usize = 0,
        grad: ?*TaggedVar = null,
        creator: ?*Function = null,
        context: *Context,
        protected: bool = false,

        chain: *Chain,

        prev: ?*TaggedVar = null,
        next: ?*TaggedVar = null,

        self_ptr: *TaggedVar,

        const Self = @This();

        pub fn destroy(self: *Self) void {
            if (self.data.isInvalidated()) {
                return;
            }

            self.data.deinitAsync(self.context.stream);

            self.unchain();

            self.creator = null;
            self.grad = null;
        }

        pub fn unchain(self: *Self) void {
            if (self.chain.var_chain) |head| {
                if (head == self.self_ptr) {
                    self.chain.var_chain = self.getNext();
                }
            }

            if (self.prev) |prev| {
                prev.setNext(self.next);
            }
            if (self.next) |next| {
                next.setPrev(self.prev);
            }

            self.prev = null;
            self.next = null;
        }

        pub fn len(self: *const Self) usize {
            return self.data.calcLen();
        }

        pub fn setPrev(self: *Self, prev: ?*TaggedVar) void {
            self.prev = prev;
        }

        pub fn setNext(self: *Self, next: ?*TaggedVar) void {
            self.next = next;
        }

        pub fn getPrev(self: *const Self) ?*TaggedVar {
            return self.prev;
        }

        pub fn getNext(self: *const Self) ?*TaggedVar {
            return self.next;
        }

        pub fn release(self: *Self) void {
            if (!self.protected) {
                self.destroy();
            } else {
                if (self.prev) |prev| {
                    prev.setNext(self.next);
                }
                if (self.next) |next| {
                    next.setPrev(self.prev);
                }
                self.prev = null;
                self.next = null;
            }
        }

        pub fn protect(self: *Self) void {
            self.protected = true;
            if (self.grad) |g| {
                g.protect();
            }
        }

        pub fn unprotect(self: *Self) void {
            self.protected = false;
        }

        pub fn setCreator(self: *Self, creator: *Function) void {
            self.creator = creator;
            self.generation = creator.getGeneration() + 1;
        }

        pub fn resetCreator(self: *Self) void {
            self.creator = null;
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

        pub fn setGrad(self: *Self, grad: ?*TaggedVar) void {
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

        // pub fn calcLen(self: *const Self) usize {
        //     var count: usize = 1;
        //     var now = self.getPrev();
        //     while (now) |nonnull_now| : (now = nonnull_now.getPrev()) {
        //         count += 1;
        //     }

        //     return count;
        // }
    };
}

pub const TaggedVar = union(enum) {
    bf16: Variable(BF16),
    f16: Variable(f16),
    f32: Variable(f32),
    f64: Variable(f64),
    usize: Variable(usize),

    pub const SeenSet = std.AutoHashMap(*TaggedVar, void);

    pub fn init(comptime T: type, variable: Variable(T)) TaggedVar {
        return switch (T) {
            BF16 => .{ .bf16 = variable },
            f16 => .{ .f16 = variable },
            f32 => .{ .f32 = variable },
            f64 => .{ .f64 = variable },
            usize => .{ .usize = variable },
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
            usize => &self.usize,
            else => unreachable,
        };
    }

    pub fn asUntaggedConst(self: *const TaggedVar, comptime T: type) *const Variable(T) {
        return switch (T) {
            BF16 => &self.bf16,
            f16 => &self.f16,
            f32 => &self.f32,
            f64 => &self.f64,
            usize => &self.usize,
            else => unreachable,
        };
    }

    pub fn len(self: *const TaggedVar) usize {
        return switch (self.*) {
            inline else => |*v| v.len(),
        };
    }

    pub fn getGeneration(self: *const TaggedVar) usize {
        return switch (self.*) {
            inline else => |*v| v.generation,
        };
    }

    pub fn getCreator(self: *const TaggedVar) ?*Function {
        return switch (self.*) {
            inline else => |*v| v.creator,
        };
    }

    pub fn resetCreator(self: *TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.resetCreator(),
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

    pub fn setGrad(self: *TaggedVar, grad: ?*TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.setGrad(grad),
        }
    }

    pub fn clearGrad(self: *TaggedVar) void {
        self.setGrad(null);
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

    pub fn getPrev(self: *const TaggedVar) ?*TaggedVar {
        return switch (self.*) {
            inline else => |*v| v.getPrev(),
        };
    }

    pub fn setPrev(self: *TaggedVar, prev: ?*TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.setPrev(prev),
        }
    }

    pub fn getNext(self: *const TaggedVar) ?*TaggedVar {
        return switch (self.*) {
            inline else => |*v| v.getNext(),
        };
    }

    pub fn setNext(self: *TaggedVar, next: ?*TaggedVar) void {
        switch (self.*) {
            inline else => |*v| v.setNext(next),
        }
    }

    pub fn writeFlowDot(self: *const TaggedVar, writer: anytype) !void {
        // arraylist will work but i dont want make another virtual function
        var funcs = Function.Queue.init(self.getContextConst().allocator, {});
        defer funcs.deinit();

        var func_seen_set = Function.SeenSet.init(self.getContextConst().allocator);
        defer func_seen_set.deinit();

        var var_seen_set = TaggedVar.SeenSet.init(self.getContextConst().allocator);
        defer var_seen_set.deinit();

        try writer.writeAll("digraph g{\n");

        try funcs.add(self.getCreator().?);
        try func_seen_set.put(self.getCreator().?, {});

        while (funcs.removeOrNull()) |func| {
            const dot_str = try func.getDotAlloc(&var_seen_set);
            defer self.getContextConst().allocator.free(dot_str);

            try writer.writeAll(dot_str);

            try func.enqueue(&funcs, &func_seen_set);
        }

        try writer.writeAll("}");
    }

    pub fn saveDot(self: *const TaggedVar, file_name: []const u8) !void {
        const file = try std.fs.cwd().createFile(file_name, .{});
        defer file.close();

        try self.writeFlowDot(file.writer());
    }

    // pub fn calcLen(self: *const TaggedVar) usize {
    //     return switch (self.*) {
    //         inline else => |*v| v.calcLen(),
    //     };
    // }

    pub fn backward(
        self: *TaggedVar,
        // comptime T: type,
    ) !void {
        // switch (self.*) {
        //     inline else => try self.getContext().backward(T, self),
        // }
        switch (self.*) {
            .bf16 => try self.getContext().backward(BF16, self),
            .f16 => try self.getContext().backward(f16, self),
            .f32 => try self.getContext().backward(f32, self),
            .f64 => try self.getContext().backward(f64, self),
            .usize => try self.getContext().backward(usize, self),
        }
    }

    pub fn backwardEx(self: *TaggedVar, chain: *Chain) !void {
        // switch (self.*) {
        //     inline else => try self.getContext().backward(T, self),
        // }
        switch (self.*) {
            .bf16 => try self.getContext().backwardEx(BF16, self, chain),
            .f16 => try self.getContext().backwardEx(f16, self, chain),
            .f32 => try self.getContext().backwardEx(f32, self, chain),
            .f64 => try self.getContext().backwardEx(f64, self, chain),
            .usize => try self.getContext().backwardEx(usize, self, chain),
        }
    }

    pub fn detatchGrad(self: *TaggedVar) *TaggedVar {
        const grad = self.refGrad();
        self.setGrad(null);
        return grad.?;
    }

    pub fn getShape(self: *const TaggedVar) []const usize {
        return switch (self.*) {
            inline else => |*v| v.data.base.getShapeConst(),
        };
    }

    pub fn getRow(self: *const TaggedVar) usize {
        return switch (self.*) {
            inline else => |*v| v.data.base.getRow(),
        };
    }

    pub fn getCol(self: *const TaggedVar) usize {
        return switch (self.*) {
            inline else => |*v| v.data.base.getCol(),
        };
    }

    pub fn calcLen(self: *const TaggedVar) usize {
        return switch (self.*) {
            inline else => |*v| v.data.base.calcLen(),
        };
    }

    pub fn writeJsonString(self: *const TaggedVar, allocator: std.mem.Allocator, writer: anytype) !void {
        switch (self.*) {
            inline else => |v| {
                var host = try v.data.toHost(allocator, v.context.stream);
                defer host.deinit(allocator);
                try std.json.stringify(host, .{}, writer);
            },
        }
    }

    pub fn fromJsonValue(comptime T: type, allocator: std.mem.Allocator, value: std.json.Value, context: *Context) !*TaggedVar {
        var host = try std.json.parseFromValue(tomo.tensor.CPUTensor(T), allocator, value, .{});
        defer host.deinit();

        var device = try host.value.toDevice(context.stream);
        errdefer device.deinitAsync(context.stream);

        return switch (T) {
            BF16 => .{ .bf16 = device },
            f16 => .{ .f16 = device },
            f32 => .{ .f32 = device },
            f64 => .{ .f64 = device },
            usize => .{ .usize = device },
            else => unreachable,
        };
    }

    pub fn readJsonValue(self: *TaggedVar, allocator: std.mem.Allocator, value: std.json.Value) !void {
        switch (self.*) {
            .bf16 => |*v| {
                var host = try std.json.parseFromValue(tomo.tensor.CPUTensor(BF16), allocator, value, .{});
                defer host.deinit();

                try v.data.writeFromHostAsync(host.value.data, 0, v.context.stream);
            },
            .f16 => |*v| {
                var host = try std.json.parseFromValue(tomo.tensor.CPUTensor(f16), allocator, value, .{});
                defer host.deinit();

                try v.data.writeFromHostAsync(host.value.data, 0, v.context.stream);
            },
            .f32 => |*v| {
                var host = try std.json.parseFromValue(tomo.tensor.CPUTensor(f32), allocator, value, .{});
                defer host.deinit();

                try v.data.writeFromHostAsync(host.value.data, 0, v.context.stream);
            },
            .f64 => |*v| {
                var host = try std.json.parseFromValue(tomo.tensor.CPUTensor(f64), allocator, value, .{});
                defer host.deinit();

                try v.data.writeFromHostAsync(host.value.data, 0, v.context.stream);
            },
            .usize => |*v| {
                var host = try std.json.parseFromValue(tomo.tensor.CPUTensor(usize), allocator, value, .{});
                defer host.deinit();

                try v.data.writeFromHostAsync(host.value.data, 0, v.context.stream);
            },
        }
    }

    pub fn writeBinary(self: *TaggedVar, allocator: std.mem.Allocator, writer: anytype) !void {
        switch (self.*) {
            inline else => |*v| {
                var host = try v.data.toHost(allocator, v.context.stream);
                defer host.deinit(allocator);

                try v.context.stream.sync();
                try writer.writeAll(constSliceCast(u8, host.data));
            },
        }
    }

    pub fn readBinary(self: *TaggedVar, slice: []const u8) ![]const u8 {
        switch (self.*) {
            .bf16 => |*v| {
                const end = v.data.calcLen() * @sizeOf(BF16);
                try v.data.writeFromHostAsync(constSliceCast(BF16, slice[0..end]), 0, v.context.stream);
                return slice[end..];
            },
            .f16 => |*v| {
                const end = v.data.calcLen() * @sizeOf(f16);
                try v.data.writeFromHostAsync(constSliceCast(f16, slice[0..end]), 0, v.context.stream);
                return slice[end..];
            },
            .f32 => |*v| {
                const end = v.data.calcLen() * @sizeOf(f32);
                try v.data.writeFromHostAsync(constSliceCast(f32, slice[0..end]), 0, v.context.stream);
                return slice[end..];
            },
            .f64 => |*v| {
                const end = v.data.calcLen() * @sizeOf(f64);
                try v.data.writeFromHostAsync(constSliceCast(f64, slice[0..end]), 0, v.context.stream);
                return slice[end..];
            },
            .usize => |*v| {
                const end = v.data.calcLen() * @sizeOf(usize);
                try v.data.writeFromHostAsync(constSliceCast(usize, slice[0..end]), 0, v.context.stream);
                return slice[end..];
            },
        }
    }
};
