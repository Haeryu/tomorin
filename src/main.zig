const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");

// TODO: graphviz first, bugfix next

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream = try tomo.stream.Stream.create();
    defer stream.destroy();

    var cuda_context = try tomo.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    var context = tomorin.context.Context.init(allocator, &cuda_context, &stream, true);
    defer context.deinit();

    try context.makeNewLevel();

    var v1 = try tomo.tensor.GPUTensor(f32).initAsync(&.{ 2, 2 }, &stream);
    errdefer v1.deinitAsync(&stream);
    try v1.fill(2.0, &stream);

    const x1 = tomorin.variable.Variable(f32){
        .data = v1,
        .level = 0,
        .context = &context,
    };

    const y = try tomorin.function.neg(f32, try tomorin.function.neg(f32, try tomorin.function.neg(f32, &x1)));

    var v_host = try y.data.toHost(allocator, &stream);
    defer v_host.deinit(allocator);

    std.debug.print("{d}", .{v_host});
}
