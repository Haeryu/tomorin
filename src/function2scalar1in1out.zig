const std = @import("std");
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;
const BF16 = tomo.BF16;
const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Rc = @import("rc.zig").Rc;
const Weak = @import("rc.zig").Weak;
const Context = @import("context.zig").Context;

const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;

const Function = @import("function.zig").Function;
const FunctionBase = @import("function.zig").FunctionBase;
const FuncDecorator1in1outBase = @import("function.zig").FuncDecorator1in1outBase;
const Chain = @import("chain.zig").Chain;
const makefunc1in1outBase = @import("function.zig").makefunc1in1outBase;

const addEx = @import("function2in1out.zig").addEx;
const mulEx = @import("function2in1out.zig").mulEx;
const scaleEx = @import("function1scalar1in1out.zig").scaleEx;

pub fn FuncDecorator2Scalar1in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator1in1outBase(Self);

        pub fn create(context: *Context, scalar1: Self.Scalar1, scalar2: Self.Scalar2, chain: *Chain) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try chain.registerFunction(
                .{
                    .ptr = self,
                    .vtable = &.{
                        .forward = &Base.forwardDecorated,
                        .backward = &Base.backwardDecorated,
                        .destroy = &Base.destroy,
                        .get_generation = &Base.getGeneration,
                        .enqueue = &Base.enqueue,
                        .get_dot_alloc = &getDotAlloc,
                    },
                    .chain = chain,
                },
            );

            self.* = .{
                .in = null,
                .out = null,
                .scalar1 = scalar1,
                .scalar2 = scalar2,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
                    .chain = chain,
                },
            };

            return func_ptr;
        }

        pub fn getDotAlloc(ctx: *anyopaque, var_seen_set: *TaggedVar.SeenSet) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.context.allocator;

            const in_contains = var_seen_set.contains(self.in.?);
            const in = if (!in_contains) try self.in.?.getDotAlloc() else "";
            defer if (!in_contains) allocator.free(in);

            try var_seen_set.put(self.in.?, {});

            const out_contains = var_seen_set.contains(self.out.?);
            const out = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out);

            try var_seen_set.put(self.out.?, {});

            const scalar1 = try std.fmt.allocPrint(allocator, "{} [label=\"{s}\", color=aquamarine, style=filled, shape=circle]", .{
                @intFromPtr(&self.scalar1),
                @typeName(Self.Scalar1),
            });
            defer allocator.free(scalar1);

            const scalar2 = try std.fmt.allocPrint(allocator, "{} [label=\"{s}\", color=aquamarine, style=filled, shape=circle]", .{
                @intFromPtr(&self.scalar2),
                @typeName(Self.Scalar2),
            });
            defer allocator.free(scalar2);

            return try std.fmt.allocPrint(allocator,
                \\{} [label="{s}", color=lightblue, style=filled, shape=box]
                \\{s}
                \\{s}
                \\{s}
                \\{s}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\{} -> {}
                \\
            , .{
                @intFromPtr(ctx),
                @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..],
                scalar1,
                scalar2,
                in,
                out,
                @intFromPtr(&self.scalar1),
                @intFromPtr(ctx),
                @intFromPtr(&self.scalar2),
                @intFromPtr(ctx),
                @intFromPtr(self.in.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(
    comptime F: type,
    x: *TaggedVar,
    scalar1: F.Scalar1,
    scalar2: F.Scalar2,
    chain: *Chain,
) !*TaggedVar {
    const funckey = try F.create(x.getContext(), scalar1, scalar2, chain);
    return try makefunc1in1outBase(funckey, x);
}

pub fn ScaleShift(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar1: T, // scale
        scalar2: T, // shift
        base: FunctionBase,

        pub const Scalar1 = T;
        pub const Scalar2 = T;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2Scalar1in1out(Self);

        const Self = ScaleShift(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.scaleShift(self.scalar1, self.scalar2, context.stream);
            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            return try scaleEx(T, gy, self.scalar1, self.base.chain);
        }
    };
}

pub fn scaleShift(comptime T: type, x: *TaggedVar, scal: T, shif: T) !*TaggedVar {
    return try scaleShiftEx(T, x, scal, shif, x.getContext().current_chain.?);
}

pub fn scaleShiftEx(comptime T: type, x: *TaggedVar, scal: T, shif: T, chain: *Chain) !*TaggedVar {
    return try makefunc(ScaleShift(T), x, scal, shif, chain);
}

pub fn Clip(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar1: T, // min
        scalar2: T, // max
        base: FunctionBase,

        pub const Scalar1 = T;
        pub const Scalar2 = T;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2Scalar1in1out(Self);

        const Self = Clip(T);

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x.cloneAsync(context.stream);
            errdefer y.deinitAsync(context.stream);

            try y.clamp(self.scalar1, self.scalar2, context.stream);

            return y;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            const context = self.base.context;

            var mask_min = try self.in.?.asUntagged(T).data.cloneAsync(context.stream);
            defer mask_min.deinitAsync(context.stream);

            var mask_max = try self.in.?.asUntagged(T).data.cloneAsync(context.stream);
            defer mask_max.deinitAsync(context.stream);

            try mask_min.gtEq(self.scalar1, context.stream);
            try mask_max.ltEq(self.scalar2, context.stream);

            try mask_min.product(&mask_max, context.stream);

            const mask = try context.current_chain.?.createVariable(T, mask_min.move(), null);

            return try mulEx(T, gy, mask, self.base.chain);
        }
    };
}

pub fn clip(comptime T: type, x: *TaggedVar, min: T, max: T) !*TaggedVar {
    return try clipEx(T, x, min, max, x.getContext().current_chain.?);
}

pub fn clipEx(comptime T: type, x: *TaggedVar, min: T, max: T, chain: *Chain) !*TaggedVar {
    return try makefunc(Clip(T), x, min, max, chain);
}

pub fn Dropout(comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        out: ?*TaggedVar,
        scalar1: T, // dropout_ratio
        scalar2: bool, // train
        base: FunctionBase,

        mask: ?*TaggedVar = null,

        pub const Scalar1 = T;
        pub const Scalar2 = bool;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator2Scalar1in1out(Self);

        const Self = Dropout(T);

        pub fn predestroy(self: *Self) void {
            if (self.mask) |m| {
                m.destroy();
            }
        }

        pub fn forward(self: *Self, x: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            if (self.scalar2) {
                var y = try x.cloneAsync(context.stream);
                errdefer y.deinitAsync(context.stream);

                if (self.mask == null) {
                    var mask: GPUTensor(T) = try .initAsync(x.base.getShape(), context.stream);
                    errdefer mask.deinitAsync(context.stream);

                    self.mask = try self.base.chain.createVariable(T, mask.move(), null);
                }

                try self.mask.?.asUntagged(T).data.fillUniform(context.cuda_context, context.stream);
                try self.mask.?.asUntagged(T).data.gt(self.scalar1, context.stream);
                const scale = 1.0 / (1.0 - self.scalar1);
                try self.mask.?.asUntagged(T).data.scale(scale, context.stream);

                try y.product(&self.mask.?.asUntagged(T).data, context.stream);

                return y;
            } else {
                return try x.cloneAsync(context.stream);
            }
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !*TaggedVar {
            if (self.scalar2) {
                return try mulEx(T, gy, self.mask.?, self.base.chain);
            } else {
                return gy;
            }
        }
    };
}

pub fn dropout(comptime T: type, x: *TaggedVar, dropout_ratio: T, train: bool) !*TaggedVar {
    return try dropoutEx(T, x, dropout_ratio, train, x.getContext().current_chain.?);
}

pub fn dropoutEx(comptime T: type, x: *TaggedVar, dropout_ratio: T, train: bool, chain: *Chain) !*TaggedVar {
    return try makefunc(Dropout(T), x, dropout_ratio, train, chain);
}

// tests
fn testScaleShiftForward(allocator: std.mem.Allocator) !void {
    // Initialize GPU components
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    // Create input tensor
    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x.deinitAsync(&stream);
    try gpu_x.writeFromHostAsync(&x_data, 0, &stream);
    var var_x = try base_chain.createVariable(T, gpu_x.move(), "x");
    defer var_x.destroy();

    // Apply ScaleShift
    const scale: T = 2.0;
    const shift: T = 1.0;
    var var_y = try scaleShiftEx(T, var_x, scale, shift, base_chain);
    defer var_y.destroy();

    // Retrieve and verify output
    var gpu_y = var_y.asUntagged(T).data;
    var host_y = try gpu_y.toHost(allocator, &stream);
    defer host_y.deinit(allocator);

    const expected_y = [_]T{ 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 }; // (x * scale) + shift
    for (host_y.data, expected_y) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testScaleShiftForward passed successfully.\n", .{});
}

fn testScaleShiftBackward(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var gpu_x = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x.deinitAsync(&stream);
    try gpu_x.writeFromHostAsync(&x_data, 0, &stream);
    var var_x = try base_chain.createVariable(T, gpu_x.move(), "x");
    defer var_x.destroy();

    const scale: T = 2.0;
    const shift: T = 1.0;
    var var_y = try scaleShiftEx(T, var_x, scale, shift, base_chain);
    defer var_y.destroy();

    // Backward pass with gradient of ones
    const gy_data = try allocator.alloc(T, shape[0] * shape[1]);
    defer allocator.free(gy_data);
    for (gy_data) |*val| val.* = 1.0;
    var gpu_gy = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_y.setGrad(var_gy);

    try var_y.backwardEx(base_chain);

    // Check gradients
    var gpu_gx = var_x.refGradConst().?.asUntaggedConst(T).data;
    var host_gx = try gpu_gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    const expected_gx = [_]T{ 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 }; // gy * scale
    for (host_gx.data, expected_gx) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testScaleShiftBackward passed successfully.\n", .{});
}

fn testClipForward(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x_data = [_]T{ -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 };
    var gpu_x = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x.deinitAsync(&stream);
    try gpu_x.writeFromHostAsync(&x_data, 0, &stream);
    var var_x = try base_chain.createVariable(T, gpu_x.move(), "x");
    defer var_x.destroy();

    const min_val: T = 0.0;
    const max_val: T = 2.0;
    var var_y = try clipEx(T, var_x, min_val, max_val, base_chain);
    defer var_y.destroy();

    var gpu_y = var_y.asUntagged(T).data;
    var host_y = try gpu_y.toHost(allocator, &stream);
    defer host_y.deinit(allocator);

    const expected_y = [_]T{ 0.0, 0.0, 1.0, 2.0, 2.0, 2.0 }; // Clamped between 0 and 2
    for (host_y.data, expected_y) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testClipForward passed successfully.\n", .{});
}

fn testClipBackward(allocator: std.mem.Allocator) !void {
    var stream = try Stream.create();
    defer stream.destroy();

    var cuda_context = try CudaContext.init();
    defer cuda_context.deinit();

    var context = try Context.init(allocator, &cuda_context, &stream, .{
        .init_func_capacity = 10,
        .init_var_capacity = 10,
    });
    defer context.deinit();

    const base_chain = try context.createChain();
    context.current_chain = base_chain;
    defer base_chain.clear();

    const T = f32;
    const shape = &[_]usize{ 2, 3 };
    var x_data = [_]T{ -1.0, 0.0, 1.0, 2.0, 3.0, 4.0 };
    var gpu_x = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_x.deinitAsync(&stream);
    try gpu_x.writeFromHostAsync(&x_data, 0, &stream);
    var var_x = try base_chain.createVariable(T, gpu_x.move(), "x");
    defer var_x.destroy();

    const min_val: T = 0.0;
    const max_val: T = 2.0;
    var var_y = try clipEx(T, var_x, min_val, max_val, base_chain);
    defer var_y.destroy();

    // Backward pass with gradient of ones
    const gy_data = try allocator.alloc(T, shape[0] * shape[1]);
    defer allocator.free(gy_data);
    for (gy_data) |*val| val.* = 1.0;
    var gpu_gy = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_y.setGrad(var_gy);

    try var_y.backwardEx(base_chain);

    // Check gradients
    var gpu_gx = var_x.refGradConst().?.asUntaggedConst(T).data;
    var host_gx = try gpu_gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    const expected_gx = [_]T{ 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 }; // 1 where min < x < max, 0 elsewhere
    std.debug.print("{any}\n", .{host_gx.data});
    for (host_gx.data, expected_gx) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testClipBackward passed successfully.\n", .{});
}

// Combined test function
pub fn test2Scalar1i1o() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try testScaleShiftForward(allocator);
    try testScaleShiftBackward(allocator);
    try testClipForward(allocator);
    // try testClipBackward(allocator); -> error

    std.debug.print("All 2scalar1i1o tests passed.\n", .{});
}
