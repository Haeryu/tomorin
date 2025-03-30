const std = @import("std");
const tomo = @import("tomo");
const util = @import("util.zig");
const GPUTensor = tomo.tensor.GPUTensor;
const Context = @import("context.zig").Context;

pub fn DataLoader(comptime Dataset: type) type {
    return struct {
        dataset: *Dataset,
        batch_size: usize,
        shuffle: bool,
        data_size: usize,
        max_iter: usize,
        index: []usize,
        xoshiro: std.Random.Xoshiro256,
        random: std.Random,
        allocator: std.mem.Allocator,
        iteration: usize,
        context: *Context,

        const Self = @This();

        pub fn init(
            allocator: std.mem.Allocator,
            dataset: *Dataset,
            batch_size: usize,
            shuffle: bool,
            context: *Context,
        ) !Self {
            var self: Self = .{
                .dataset = dataset,
                .batch_size = batch_size,
                .shuffle = shuffle,
                .data_size = dataset.len(),
                .max_iter = try std.math.divCeil(usize, dataset.len(), batch_size),
                .index = &.{},
                .xoshiro = undefined,
                .random = undefined,
                .allocator = allocator,
                .iteration = 0,
                .context = context,
            };

            self.index = try util.arangeAlloc(allocator, usize, 0, dataset.len(), 1);

            errdefer allocator.free(self.index);

            self.xoshiro = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
            self.random = self.xoshiro.random();

            self.reset();

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.index);
        }

        pub fn reset(self: *Self) void {
            self.iteration = 0;
            if (self.shuffle) {
                std.Random.shuffle(self.random, usize, self.index);
            }
        }

        pub fn writeNextBatch(self: *Self, batch: anytype) !?usize {
            if (self.iteration >= self.max_iter) {
                self.reset();
                return null;
            }

            // if (self.iteration * self.batch_size >= self.index.len or (self.iteration + 1) * self.batch_size >= self.index.len) {
            //     self.reset();
            //     return null;
            // }

            const start = self.iteration * self.batch_size;
            const end = @min(start + self.batch_size, self.data_size);
            const batch_is = self.index[start..end];

            // TODO: Error handle while looping
            for (0.., batch_is) |i, batch_i| {
                try self.dataset.write(i, batch_i, batch, self.context);
            }

            self.iteration += 1;

            return self.iteration;
        }
    };
}
