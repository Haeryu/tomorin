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
const FuncDecorator3in1outBase = @import("function.zig").FuncDecorator3in1outBase;
const Chain = @import("chain.zig").Chain;
const makefunc3in1outBase = @import("function.zig").makefunc3in1outBase;

const transposeEx = @import("function1in1out.zig").transposeEx;
const reshapeEx = @import("function1slice1in1out.zig").reshapeEx;
const sumToEx = @import("function1in1out.zig").sumToEx;
const matmulEx = @import("function2in1out.zig").matmulEx;
const broadcastToEx = @import("function1slice1in1out.zig").broadcastToEx;
const transposeExEx = @import("function1slice1in1out.zig").transposeExEx;

pub fn FuncDecorator3in1out(comptime Self: type) type {
    return struct {
        const Base = FuncDecorator3in1outBase(Self);

        pub fn create(context: *Context, chain: *Chain) !*Function {
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
                .in1 = null,
                .in2 = null,
                .in3 = null,
                .out = null,
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

            const in1_contains = var_seen_set.contains(self.in1.?);
            const in1 = if (!in1_contains) try self.in1.?.getDotAlloc() else "";
            defer if (!in1_contains) allocator.free(in1);

            try var_seen_set.put(self.in1.?, {});

            const in2_contains = var_seen_set.contains(self.in2.?);
            const in2 = if (!in2_contains) try self.in2.?.getDotAlloc() else "";
            defer if (!in2_contains) allocator.free(in2);

            try var_seen_set.put(self.in2.?, {});

            const in3_contains = var_seen_set.contains(self.in3.?);
            const in3 = if (!in3_contains) try self.in3.?.getDotAlloc() else "";
            defer if (!in3_contains) allocator.free(in3);

            try var_seen_set.put(self.in3.?, {});

            const out_contains = var_seen_set.contains(self.out.?);
            const out = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out);

            try var_seen_set.put(self.out.?, {});

            return try std.fmt.allocPrint(self.base.context.allocator,
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
                in1,
                in2,
                in3,
                out,
                @intFromPtr(self.in1.?),
                @intFromPtr(ctx),
                @intFromPtr(self.in2.?),
                @intFromPtr(ctx),
                @intFromPtr(self.in3.?),
                @intFromPtr(ctx),
                @intFromPtr(ctx),
                @intFromPtr(self.out.?),
            });
        }
    };
}

fn makefunc(
    comptime F: type,
    x1: *TaggedVar,
    x2: *TaggedVar,
    x3: *TaggedVar,
    chain: *Chain,
) !*TaggedVar {
    const funckey = try F.create(x1.getContext(), chain);

    return try makefunc3in1outBase(funckey, x1, x2, x3);
}

pub fn Linear(comptime T: type) type {
    return struct {
        in1: ?*TaggedVar,
        in2: ?*TaggedVar,
        in3: ?*TaggedVar,
        out: ?*TaggedVar,
        base: FunctionBase,

        pub const In1 = T;
        pub const In2 = T;
        pub const In3 = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator3in1out(Self);

        const Self = Linear(T);

        // pub fn matmul(
        //             self: *const Self,
        //             self_transpose: bool,
        //             other_tensor: anytype,
        //             other_transpose: bool,
        //             add_tensor: anytype,
        //             add_transpose: bool,
        //             comptime cublas_compute_type: c.cublasComputeType_t,
        //             alpha: if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) f16 else f32,
        //             beta: if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) f16 else f32,
        //             comptime EpilogueT: type,
        //             epilogue_config: EpilogueT.Config,
        //             //cublas_compute_type: c.cublasComputeType_t,
        //             stream: *const Stream,
        //             cuda_context: *const CudaContext,
        //             comptime OutType: type,
        //         )

        // pub const Activation = enum {
        //     none,
        //     relu,
        //     gelu,
        // };

        // pub const Config = struct {
        //     activation: Activation = .none,
        //     bias_tensor: ?BiasTensor = null,
        //     aux_tensor: ?AuxTensor = null,
        // };

        pub fn forward(self: *Self, x1: *const GPUTensor(T), x2: *const GPUTensor(T), x3: *const GPUTensor(T)) !GPUTensor(T) {
            const context = self.base.context;
            var y = try x1.linearImp(x2, x3, context.stream);
            errdefer y.deinitAsync(context.stream);

            return y.move();
        }

        pub fn backward(self: *Self, gy: *TaggedVar) !std.meta.Tuple(&.{ *TaggedVar, *TaggedVar, *TaggedVar }) {
            var perm: [GPUTensor(T).max_rank]usize = undefined;
            for (perm[0..gy.getShape().len], 0..) |*p, i| {
                p.* = i;
            }
            std.mem.swap(usize, &perm[gy.getShape().len - 1], &perm[gy.getShape().len - 2]);
            const gx = try matmulEx(T, gy, try transposeExEx(T, self.in2.?, perm[0..gy.getShape().len], self.base.chain), self.base.chain);
            const gw = try matmulEx(T, try transposeExEx(T, self.in1.?, perm[0..gy.getShape().len], self.base.chain), gy, self.base.chain);
            const gb = try sumToEx(T, gy, self.in3.?.asUntaggedConst(T).data.base.getShapeConst(), self.base.chain);

            return .{ gx, gw, gb };
        }
    };
}

pub fn linear(comptime T: type, x1: *TaggedVar, x2: *TaggedVar, x3: *TaggedVar) !*TaggedVar {
    return try linearEx(T, x1, x2, x3, x1.getContext().current_chain.?);
}

pub fn linearEx(comptime T: type, x1: *TaggedVar, x2: *TaggedVar, x3: *TaggedVar, chain: *Chain) !*TaggedVar {
    return try makefunc(Linear(T), x1, x2, x3, chain);
}

fn testLinearForward(allocator: std.mem.Allocator) !void {
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

    // Define tensor type and shapes with batch dimension
    const T = f32;
    const batch_size = 2;

    // Initialize input tensor (x)
    var x_data = [_]T{
        1.0, 2.0, 3.0, // Batch 1, Row 1
        4.0, 5.0, 6.0, // Batch 1, Row 2
        0.5, 1.5, 2.5, // Batch 2, Row 1
        3.5, 4.5, 5.5, // Batch 2, Row 2
    };
    var gpu_x = try GPUTensor(T).initAsync(&.{ batch_size, 2, 3 }, &stream);
    try gpu_x.writeFromHostAsync(&x_data, 0, &stream);
    const var_x = try base_chain.createVariable(T, gpu_x.move(), "x");

    // Initialize weight w (shape: [3, 2])
    var w_data = [_]T{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    var gpu_w = try GPUTensor(T).initAsync(&.{ 3, 2 }, &stream);
    try gpu_w.writeFromHostAsync(&w_data, 0, &stream);
    const var_w = try base_chain.createVariable(T, gpu_w.move(), "w");

    // Reshape w to [1, 3, 2] and broadcast to [batch_size, 3, 2]
    const reshaped_w = try reshapeEx(T, var_w, &.{ 1, 3, 2 }, base_chain);
    const broadcasted_w = try broadcastToEx(T, reshaped_w, &.{ batch_size, 3, 2 }, base_chain);

    // Initialize bias b (shape: [1, 2])
    var b_data = [_]T{ 0.1, 0.2 };
    var gpu_b = try GPUTensor(T).initAsync(&.{ 1, 2 }, &stream);
    try gpu_b.writeFromHostAsync(&b_data, 0, &stream);
    const var_b = try base_chain.createVariable(T, gpu_b.move(), "b");

    // Reshape b to [1, 1, 2] and broadcast to [batch_size, 2, 2]
    const reshaped_b = try reshapeEx(T, var_b, &.{ 1, 1, 2 }, base_chain);
    const broadcasted_b = try broadcastToEx(T, reshaped_b, &.{ batch_size, 2, 2 }, base_chain);

    // Apply linear function with broadcasted w and b
    var var_y = try linearEx(T, var_x, broadcasted_w, broadcasted_b, base_chain);

    // Retrieve output
    var gpu_y = var_y.asUntagged(T).data;
    var host_y = try gpu_y.toHost(allocator, &stream);
    defer host_y.deinit(allocator);

    // Expected output: y = x @ w + b for each batch
    const expected_y = [_]T{
        // Batch 1, Row 1
        1.0 * 0.1 + 2.0 * 0.3 + 3.0 * 0.5 + 0.1, // 1.8
        1.0 * 0.2 + 2.0 * 0.4 + 3.0 * 0.6 + 0.2, // 2.8
            // Batch 1, Row 2
        4.0 * 0.1 + 5.0 * 0.3 + 6.0 * 0.5 + 0.1, // 5.0
        4.0 * 0.2 + 5.0 * 0.4 + 6.0 * 0.6 + 0.2, // 6.0
            // Batch 2, Row 1
        0.5 * 0.1 + 1.5 * 0.3 + 2.5 * 0.5 + 0.1, // 1.85
        0.5 * 0.2 + 1.5 * 0.4 + 2.5 * 0.6 + 0.2, // 2.4
            // Batch 2, Row 2
        3.5 * 0.1 + 4.5 * 0.3 + 5.5 * 0.5 + 0.1, // 4.55
        3.5 * 0.2 + 4.5 * 0.4 + 5.5 * 0.6 + 0.2, // 6.0
    };

    // Verify output
    for (host_y.data, expected_y) |computed, expected| {
        if (@abs(computed - expected) > 1e-5) return error.TestFailed;
    }

    std.debug.print("testLinearForward passed successfully.\n", .{});
}

// Test the backward pass
fn testLinearBackward(allocator: std.mem.Allocator) !void {
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

    // Define tensor type and shapes with batch dimension
    const T = f32;
    const batch_size = 2;
    const shape_x = &[_]usize{ batch_size, 2, 3 }; // Batch=2, each 2x3 matrix
    const shape_w = &[_]usize{ 3, 2 }; // Weights: 3x2
    const shape_b = &[_]usize{ 1, 2 }; // Bias: 1x2

    // Initialize input tensor (x)
    var x_data = [_]T{
        1.0, 2.0, 3.0, // Batch 1, Row 1
        4.0, 5.0, 6.0, // Batch 1, Row 2
        0.5, 1.5, 2.5, // Batch 2, Row 1
        3.5, 4.5, 5.5, // Batch 2, Row 2
    };
    var gpu_x = try GPUTensor(T).initAsync(shape_x, &stream);
    defer gpu_x.deinitAsync(&stream);
    try gpu_x.writeFromHostAsync(&x_data, 0, &stream);
    var var_x = try base_chain.createVariable(T, gpu_x.move(), "x");
    defer var_x.destroy();

    // Initialize weight tensor (w)
    var w_data = [_]T{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    var gpu_w = try GPUTensor(T).initAsync(shape_w, &stream);
    defer gpu_w.deinitAsync(&stream);
    try gpu_w.writeFromHostAsync(&w_data, 0, &stream);
    var var_w = try base_chain.createVariable(T, gpu_w.move(), "w");
    defer var_w.destroy();

    const reshaped_w = try reshapeEx(T, var_w, &.{ 1, 3, 2 }, base_chain);
    const broadcasted_w = try broadcastToEx(T, reshaped_w, &.{ batch_size, 3, 2 }, base_chain);

    // Initialize bias tensor (b)
    var b_data = [_]T{ 0.1, 0.2 };
    var gpu_b = try GPUTensor(T).initAsync(shape_b, &stream);
    defer gpu_b.deinitAsync(&stream);
    try gpu_b.writeFromHostAsync(&b_data, 0, &stream);
    var var_b = try base_chain.createVariable(T, gpu_b.move(), "b");
    defer var_b.destroy();

    // Explicitly broadcast bias to match output shape [batch_size, 2, 2]
    const reshaped_b = try reshapeEx(T, var_b, &.{ 1, 1, 2 }, base_chain);
    const broadcasted_b = try broadcastToEx(T, reshaped_b, &.{ batch_size, 2, 2 }, base_chain);

    // Apply the linear function with broadcasted bias
    var var_y = try linearEx(T, var_x, broadcasted_w, broadcasted_b, base_chain);
    defer var_y.destroy();

    // Set up gradient for backward pass (dy/dy = 1)
    const shape_y = &[_]usize{ batch_size, 2, 2 };
    const gy_data = try allocator.alloc(T, batch_size * 2 * 2);
    defer allocator.free(gy_data);
    for (gy_data) |*val| val.* = 1.0; // Gradient of ones
    var gpu_gy = try GPUTensor(T).initAsync(shape_y, &stream);
    defer gpu_gy.deinitAsync(&stream);
    try gpu_gy.writeFromHostAsync(gy_data, 0, &stream);
    var var_gy = try base_chain.createVariable(T, gpu_gy.move(), "gy");
    defer var_gy.destroy();
    var_y.setGrad(var_gy);

    // Perform backward pass
    try var_y.backwardEx(base_chain);

    // Retrieve gradients
    var gpu_gx = var_x.refGradConst().?.asUntaggedConst(T).data;
    var host_gx = try gpu_gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    var gpu_gw = var_w.refGradConst().?.asUntaggedConst(T).data;
    var host_gw = try gpu_gw.toHost(allocator, &stream);
    defer host_gw.deinit(allocator);

    var gpu_gb = var_b.refGradConst().?.asUntaggedConst(T).data;
    var host_gb = try gpu_gb.toHost(allocator, &stream);
    defer host_gb.deinit(allocator);

    // Expected gradients
    const expected_gx = [_]T{
        // Batch 1, Row 1
        1.0 * 0.1 + 1.0 * 0.2, // 0.3
        1.0 * 0.3 + 1.0 * 0.4, // 0.7
        1.0 * 0.5 + 1.0 * 0.6, // 1.1
            // Batch 1, Row 2
        1.0 * 0.1 + 1.0 * 0.2, // 0.3
        1.0 * 0.3 + 1.0 * 0.4, // 0.7
        1.0 * 0.5 + 1.0 * 0.6, // 1.1
            // Batch 2, Row 1
        1.0 * 0.1 + 1.0 * 0.2, // 0.3
        1.0 * 0.3 + 1.0 * 0.4, // 0.7
        1.0 * 0.5 + 1.0 * 0.6, // 1.1
            // Batch 2, Row 2
        1.0 * 0.1 + 1.0 * 0.2, // 0.3
        1.0 * 0.3 + 1.0 * 0.4, // 0.7
        1.0 * 0.5 + 1.0 * 0.6, // 1.1
    };

    const expected_gw = [_]T{
        9.0,  9.0,
        13.0, 13.0,
        17.0, 17.0,
    };

    const expected_gb = [_]T{ 4.0, 4.0 };

    // Verify gradients
    for (host_gx.data, expected_gx) |computed, expected| {
        if (@abs(computed - expected) > 1e-2) return error.TestFailed;
    }
    for (host_gw.data, expected_gw) |computed, expected| {
        if (@abs(computed - expected) > 1e-2) return error.TestFailed;
    }
    for (host_gb.data, expected_gb) |computed, expected| {
        if (@abs(computed - expected) > 1e-2) return error.TestFailed;
    }

    std.debug.print("testLinearBackward passed successfully.\n", .{});
}

// Combined test function
pub fn test3i1o() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try testLinearForward(allocator);
    try testLinearBackward(allocator);

    std.debug.print("All 3i1o tests passed.\n", .{});
}
