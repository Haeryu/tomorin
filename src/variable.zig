const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;

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

        const Self = @This();

        pub const Destructor = struct {
            stream: *const Stream,
            allocator: std.mem.Allocator,

            pub fn destroy(self: *Destructor, variable: *Variable(T)) void {
                variable.data.deinitAsync(self.stream);
                if (variable.grad) |*grad| {
                    grad.release(self.allocator);
                    variable.grad = null;
                }
            }
        };

        pub fn create(allocator: std.mem.Allocator, data: GPUTensor(T), stream: *const Stream) !PVariable(T) {
            var pvar = try PVariable(T).create(allocator, .{
                .data = data,
                .grad = null,
            }, .{
                .allocator = allocator,
                .stream = stream,
            });
            defer pvar.release(allocator);

            return pvar.clone();
        }

        pub fn createTagged(allocator: std.mem.Allocator, data: GPUTensor(T), stream: *const Stream) !PVarTagged {
            var pvar = try Self.create(allocator, data, stream);
            errdefer pvar.release(allocator);

            return PVarTagged.init(T, pvar.move());
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
};
