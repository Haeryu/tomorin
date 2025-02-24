const tomorin = @import("tomorin");
const tomo = @import("tomo");
const std = @import("std");

pub fn main() !void {
    var stream = try tomo.stream.Stream.create();
    defer stream.destroy();

    var in_data = tomo.tensor.GPUTensor(f32, 1){};
    try in_data.initAsync(.{3}, &stream);
    defer in_data.deinitAsync(&stream);

    var out_data = tomo.tensor.GPUTensor(f32, 1){};
    try out_data.initAsync(.{3}, &stream);
    defer out_data.deinitAsync(&stream);

    var out_host = try tomo.tensor.CPUTensor(f32, 1).init(tomo.allocator.cuda_pinned_allocator, .{3});
    defer out_host.deinit(tomo.allocator.cuda_pinned_allocator);

    var x = tomorin.variable.Variable(f32, 1){
        .data = &in_data,
    };

    var y = tomorin.variable.Variable(f32, 1){
        .data = &out_data,
    };

    try stream.sync();

    try in_data.writeFromHostAsync(&.{ 0, 1, 2 }, 0, &stream);
    var sq = tomorin.function.Square(f32, 1, f32, 1){ .in = &x };
    var f = sq.function();
    try f.forward(&x, &stream, &y);

    try out_host.writeFromDevice(y.data.ptr.?, y.data.calcLen(), 0, &stream);

    try stream.sync();

    std.debug.print("{d}", .{out_host});
}
