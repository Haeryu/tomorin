const std = @import("std");

pub fn ControlBlock(comptime T: type, comptime Destructor: type) type {
    return struct {
        strong_count: usize,
        weak_count: usize,
        data: T,
        destructor: Destructor,

        const Self = @This();

        fn destroy(self: *Self) void {
            if (Destructor != void) {
                self.destructor.destroy(&self.data);
            }
        }
    };
}

pub fn Rc(comptime T: type, comptime Destructor: type) type {
    return struct {
        cb_ptr: ?*CB = null,

        const CB = ControlBlock(T, Destructor);
        const Self = Rc(T, Destructor);

        pub fn create(allocator: std.mem.Allocator, data: T, destructor: Destructor) !Self {
            const cb_ptr = try allocator.create(CB);
            errdefer allocator.destroy(cb_ptr);

            cb_ptr.* = .{
                .strong_count = 1,
                .weak_count = 0,
                .data = data,
                .destructor = destructor,
            };

            return .{ .cb_ptr = cb_ptr };
        }

        pub fn clone(self: Self) Self {
            if (self.cb_ptr) |cb| {
                cb.strong_count += 1;
            }
            return self;
        }

        pub fn release(self: *Self, allocator: std.mem.Allocator) void {
            if (self.cb_ptr) |cb| {
                std.debug.assert(cb.strong_count > 0);
                cb.strong_count -= 1;

                if (cb.strong_count == 0) {
                    cb.destroy();
                    if (cb.weak_count == 0) {
                        allocator.destroy(cb);
                    }
                }
                self.cb_ptr = null;
            }
        }

        pub fn get(self: *Self) ?*T {
            if (!self.cb_ptr) return null;
            if (self.cb_ptr.strong_count == 0) {
                return null;
            }
            return &self.cb_ptr.data;
        }

        pub fn getConst(self: *const Self) ?*const T {
            if (!self.cb_ptr) return null;
            if (self.cb_ptr.strong_count == 0) {
                return null;
            }
            return &self.cb_ptr.data;
        }

        pub fn downgrade(self: *Self) Weak(T) {
            if (self.cb_ptr) |cb| {
                cb.weak_count += 1;
                return .{ .cb_ptr = cb };
            }
            return .{ .cb_ptr = null };
        }
    };
}

pub fn Weak(comptime T: type, comptime Destructor: type) type {
    return struct {
        cb_ptr: ?*ControlBlock(T, Destructor) = null,

        const CB = ControlBlock(T, Destructor);
        const Self = Weak(T, Destructor);

        pub fn upgrade(self: *Self) ?Rc(T, Destructor) {
            if (self.cb_ptr) |cb| {
                if (cb.strong_count > 0) {
                    cb.strong_count += 1;
                    return .{ .cb_ptr = cb };
                } else {
                    return null;
                }
            }
            return null;
        }

        pub fn release(self: *Self, allocator: std.mem.Allocator) void {
            if (self.cb_ptr) |cb| {
                std.debug.assert(cb.weak_count > 0);
                cb.weak_count -= 1;
                if (cb.strong_count == 0 and cb.weak_count == 0) {
                    allocator.destroy(cb);
                }
                self.cb_ptr = null;
            }
        }
    };
}
