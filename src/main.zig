const std = @import("std");
const tomo = @import("tomo");
const tomorin = @import("tomorin");

pub fn main() !void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();

    // const allocator = gpa.allocator();

    var stream = try tomo.stream.Stream.create();
    defer stream.destroy();

    var v1 = try tomorin.variable.Variable.init(f32, &.{ 1, 1 }, &stream);
    defer v1.deinit(&stream);

    try v1.writeFromHostAsync(f32, &.{1.0}, 0, &stream);

    // var v1_cpu = try v1.toCpu(f32, allocator, &stream);
    // defer v1_cpu.deinit(allocator);

    try stream.sync();

    std.debug.print("{}", .{v1});
}
