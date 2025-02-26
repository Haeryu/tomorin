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
    //defer exp.release(allocator);

    var exp2 = try tomorin.function.Exp(f32).create(allocator);
    //defer exp2.release(allocator);

    var tagged = [1]?tomorin.variable.PVarTagged{v1.clone()};

    var v2s = try exp.forward(allocator, &tagged, &cuda_context, &stream);

    const v3s = try exp2.forward(allocator, &v2s, &cuda_context, &stream);
    var v3 = v3s[0].?;
    defer v3.release(allocator);

    var v3_untagged = v3.untag(f32);

    try v3_untagged.get().?.backward(allocator, &cuda_context, &stream);

    var v3_cpu = try v1.untag(f32).getConst().?.grad.?.getConst().?.data.toHost(allocator, &stream);
    defer v3_cpu.deinit(allocator);

    try stream.sync();

    std.debug.print("{any}", .{v3_cpu});
}
