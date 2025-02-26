const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream = try tomo.stream.Stream.create();
    defer stream.destroy();

    var cuda_context = try tomo.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    var data = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 1, 1 }, &stream);
    errdefer data.deinitAsync(&stream);
    try data.writeFromHostAsync(&.{1.0}, 0, &stream);

    var v1 = try tomorin.variable.Variable(f32).createTagged(allocator, data, &stream);
    defer v1.release(allocator);

    var data2 = try data.cloneAsync(&stream);
    defer data2.deinitAsync(&stream);

    var exp = try tomorin.function.Exp(f32).create(allocator);
    defer exp.release(allocator);

    var tagged = [1]?tomorin.variable.PVarTagged{v1.move()};

    const v2s = try exp.get().?.forward(allocator, &tagged, &cuda_context, &stream);
    var v2 = v2s[0].?;
    defer v2.release(allocator);

    var v2_untagged = v2.untagMove(f32);
    defer v2_untagged.release(allocator);

    var v2_cpu = try v2_untagged.get().?.data.toHost(allocator, &stream);
    defer v2_cpu.deinit(allocator);

    try stream.sync();

    std.debug.print("{any}", .{v2_untagged});
}
