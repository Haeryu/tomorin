const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;

const PFunction = @import("function.zig").PFunction;
const Function = @import("function.zig").Function;

pub fn PVariable(comptime T: type) type {
    return Rc(Variable(T), Variable(T).Destructor);
}

pub fn PVariableWeak(comptime T: type) type {
    return Weak(Variable(T), Variable(T).Destructor);
}

pub fn Variable(comptime T: type) type {
    return struct {
        data: GPUTensor(T),
        grad: ?PVariable(T),
        creator: ?PFunction,
        generation: usize,
        name: ?[]u8,
        context: *const Context,

        const Self = @This();

        pub const Destructor = struct {
            pub fn destroy(_: *Destructor, variable: *Variable(T)) void {
                variable.data.deinitAsync(variable.context.stream);
                if (variable.grad) |*grad| {
                    grad.release(variable.context.variable_allocator);
                    variable.grad = null;
                }

                if (variable.creator) |*creator| {
                    creator.release(variable.context.variable_allocator);
                    variable.creator = null;
                }
            }
        };

        pub fn create(
            data: GPUTensor(T),
            name: ?[]u8,
            context: *const Context,
        ) !PVariable(T) {
            var pvar = try PVariable(T).create(context.variable_allocator, .{
                .data = data,
                .grad = null,
                .creator = null,
                .generation = 0,
                .name = name,
                .context = context,
            }, .{});
            defer pvar.release(context.variable_allocator);

            return pvar.move();
        }

        pub fn createTagged(
            data: GPUTensor(T),
            name: ?[]u8,
            context: *const Context,
        ) !PVarTagged {
            var pvar = try Self.create(
                data,
                name,
                context,
            );
            defer pvar.release(context.variable_allocator);

            return PVarTagged.init(T, pvar.move());
        }

        pub fn setCreator(self: *Self, creator: PFunction) void {
            std.debug.assert(self.creator == null);
            self.generation = creator.pfn.getConst().?.generation + 1;
            self.creator = creator.move();
        }

        pub fn backward(self: *Self) !void {
            if (self.grad == null) {
                var ones = try GPUTensor(T).initAsync(self.data.base.getShape(), self.context.stream);
                defer ones.deinitAsync(self.context.stream);

                try ones.fill(1.0, self.context.stream);

                self.grad = try Self.create(
                    ones.move(),
                    null,
                    self.context,
                );
            }

            var funcs = PFunction.Queue.init(self.context.multi_purpose_allocator, {});
            defer funcs.deinit();

            var seen_set = std.AutoHashMap(*const Function, void).init(self.context.multi_purpose_allocator);
            defer seen_set.deinit();

            try funcs.add(self.creator.?.get().?);
            try seen_set.put(self.creator.?.getConst().?, {});

            while (funcs.removeOrNull()) |f| {
                try f.backward();
                try f.addInputsCreators(&funcs, &seen_set);
            }
        }

        pub fn cleargrad(self: *Self) void {
            if (self.grad) |*grad| {
                grad.release(self.context.variable_allocator);
                self.grad = null;
            }
        }
    };
}

pub const PVarTagged = union(enum) {
    bf16: PVariable(BF16),
    f16: PVariable(f16),
    f32: PVariable(f32),
    f64: PVariable(f64),

    pub fn init(comptime T: type, pvar: PVariable(T)) PVarTagged {
        return switch (T) {
            BF16 => .{ .bf16 = pvar },
            f16 => .{ .f16 = pvar },
            f32 => .{ .f32 = pvar },
            f64 => .{ .f64 = pvar },
            else => unreachable,
        };
    }

    pub fn untagClone(self: PVarTagged, comptime T: type) PVariable(T) {
        return switch (T) {
            BF16 => self.bf16.clone(),
            f16 => self.f16.clone(),
            f32 => self.f32.clone(),
            f64 => self.f64.clone(),
            else => unreachable,
        };
    }

    pub fn untagMove(self: *PVarTagged, comptime T: type) PVariable(T) {
        return switch (T) {
            BF16 => self.bf16.move(),
            f16 => self.f16.move(),
            f32 => self.f32.move(),
            f64 => self.f64.move(),
            else => unreachable,
        };
    }

    pub fn untag(self: *PVarTagged, comptime T: type) PVariable(T) {
        return switch (T) {
            BF16 => self.bf16,
            f16 => self.f16,
            f32 => self.f32,
            f64 => self.f64,
            else => unreachable,
        };
    }

    pub fn move(self: *PVarTagged) PVarTagged {
        return switch (self.*) {
            inline else => |*p, tag| @unionInit(PVarTagged, @tagName(tag), p.move()),
        };
    }

    pub fn clone(self: *PVarTagged) PVarTagged {
        return switch (self.*) {
            inline else => |p, tag| @unionInit(PVarTagged, @tagName(tag), p.clone()),
        };
    }

    pub fn release(self: *PVarTagged, allocator: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*p| p.release(allocator),
        }
    }

    pub fn getGeneration(self: *const PVarTagged) usize {
        return switch (self.*) {
            inline else => |p| p.getConst().?.generation,
        };
    }
};
