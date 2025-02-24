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

    const V = tomorin.variable.Variable(F, 1);

    var add = try tomorin.function.Add(F, 1, 2).create(allocator);
    defer add.destroy(allocator, &stream);

    var x = try V.create(allocator, .{1}, &stream);
    defer x.destroy(allocator, &stream);
    try x.data.writeFromHostAsync(&.{3.0}, 0, &stream);

    var x2 = try V.create(allocator, .{1}, &stream);
    defer x2.destroy(allocator, &stream);
    try x2.data.writeFromHostAsync(&.{3.0}, 0, &stream);

    var xs = .{ x, x2 };
    const z_erased = try add.forwardErased(allocator, &xs, &cuda_context, &stream);

    const z: *V = @ptrCast(@alignCast(z_erased));

    try z.backward(allocator, &cuda_context, &stream);

    try stream.sync();

    var z_cpu_tensor = try z.data.toHost(allocator, &stream);
    defer z_cpu_tensor.deinit(allocator);

    var x_grad_cpu_tensor = try x.grad.?.data.toHost(allocator, &stream);
    defer x_grad_cpu_tensor.deinit(allocator);

    try stream.sync();

    std.debug.print("{d}", .{z_cpu_tensor});
    std.debug.print("{d}", .{x_grad_cpu_tensor});
}
