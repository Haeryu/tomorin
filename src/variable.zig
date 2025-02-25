const std = @import("std");
pub const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;

pub fn Variable(comptime T: type) type {
    return struct {
        data: GPUTensor(T),
        grad: ?PVariable,

        const Self = @This();
        const PVariable = Rc(Self, Self.Destructor);

        pub const Destructor = struct {
            stream: *const Stream,
            allocator: std.mem.Allocator,

            pub fn destroy(self: *Destructor, variable: *Variable(T)) void {
                variable.data.deinitAsync(self.stream);
                if (variable.grad) |grad| {
                    grad.release(self.allocator);
                    variable.grad = null;
                }
            }
        };

        pub fn init(allocator: std.mem.Allocator, data: GPUTensor(T), stream: *const Stream) !PVariable {
            errdefer data.deinitAsync(stream);

            var pvar = try PVariable.create(allocator, .{
                .data = data,
                .grad = null,
            }, .{
                .allocator = allocator,
                .stream = stream,
            });
            defer pvar.release(allocator);

            return pvar.clone();
        }
    };
}
