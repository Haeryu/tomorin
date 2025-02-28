const std = @import("std");

const Function = @import("function.zig").Function;
const Context = @import("context.zig").Context;

pub fn LevelStack(comptime T: type) type {
    return struct {
        levels: std.ArrayList(std.ArrayList(T)),

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .levels = std.ArrayList(std.ArrayList(T)).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.levels.items) |*level| {
                level.deinit();
            }
            self.levels.deinit();
        }

        pub fn getTopLevel(self: *Self) *std.ArrayList(T) {
            return &self.levels.items[self.levels.items.len - 1];
        }

        pub fn getTopLevelTopItemIndex(self: *Self) usize {
            const top_level = self.getTopLevel();
            return top_level.items.len - 1;
        }

        pub fn getTopLevelTopItem(self: *Self) *T {
            const top_level = self.getTopLevel();
            return &top_level.items[top_level.items.len - 1];
        }

        pub fn getTopLevelConst(self: *const Self) *const std.ArrayList(T) {
            return &self.levels.items[self.levels.items.len - 1];
        }

        pub fn getTopLevelTopItemConst(self: *const Self) *const T {
            const top_level = self.getTopLevelConst();
            return &top_level.items[top_level.items.len - 1];
        }

        pub fn makeNewLevel(self: *Self) !void {
            var new_level = std.ArrayList(T).init(self.levels.allocator);
            errdefer new_level.deinit();

            try self.levels.append(new_level);
        }

        pub fn pushAtTopLevel(self: *Self, val: T) !void {
            try self.getTopLevel().append(val);
        }
    };
}
