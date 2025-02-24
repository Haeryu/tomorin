const tomorin = @import("tomorin");
const tomo = @import("tomo");
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream = try tomo.stream.Stream.create();
    defer stream.destroy();

    var cuda_context = try tomo.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    const F = f32;
    var A = try tomorin.function.Square(F, 1).create(allocator);
    defer A.destroy(allocator, &stream);
    var B = try tomorin.function.Exp(F, 1).create(allocator);
    defer B.destroy(allocator, &stream);
    var C = try tomorin.function.Square(F, 1).create(allocator);
    defer C.destroy(allocator, &stream);

    const V = tomorin.variable.Variable(F, 1);

    var x = try V.create(allocator, .{1}, &stream);
    defer x.destroy(allocator, &stream);

    try x.data.writeFromHostAsync(&.{0.5}, 0, &stream);

    const a = try A.forward(V, V, allocator, x, &cuda_context, &stream);

    const b = try B.forward(V, V, allocator, a, &cuda_context, &stream);

    var y = try C.forward(V, V, allocator, b, &cuda_context, &stream);

    try y.backward(allocator, &cuda_context, &stream);

    try stream.sync();

    std.debug.assert(y.creator != null);

    var cpu_tensor = try x.grad.?.data.toHost(allocator, &stream);
    defer cpu_tensor.deinit(allocator);

    try stream.sync();

    std.debug.print("{d}", .{cpu_tensor});
}
