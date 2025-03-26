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
const Chain = @import("chain.zig").Chain;

const addEx = @import("function2in1out.zig").addEx;
const concatEx = @import("function1scalarmanyin1out.zig").concatEx;

pub fn FuncDecorator1Scalar1inManyout(comptime Self: type) type {
    return struct {
        pub fn create(context: *Context, scalar: Self.Scalar, chain: *Chain) !*Function {
            const self = try context.allocator.create(Self);
            errdefer context.allocator.destroy(self);

            const func_ptr = try chain.registerFunction(
                .{
                    .ptr = self,
                    .vtable = &.{
                        .forward = &forwardDecorated,
                        .backward = &backwardDecorated,
                        .destroy = &destroy,
                        .get_generation = &getGeneration,
                        .enqueue = &enqueue,
                        .get_dot_alloc = &getDotAlloc,
                    },
                    .chain = chain,
                },
            );

            self.* = .{
                .in = null,
                .outs = .{null} ** Self.out, // Initialize with nulls
                .scalar = scalar,
                .base = .{
                    .func_ptr = func_ptr,
                    .context = context,
                    .chain = chain,
                },
            };

            return func_ptr;
        }

        pub fn forwardDecorated(ctx: *anyopaque, args: []*TaggedVar, out: []?*TaggedVar) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.in = args[0];

            const outputs = try self.forward(&self.in.?.asUntaggedConst(Self.In).data);

            for (&outputs, 0..) |output_tensor, i| {
                const var_out = try self.base.chain.createVariable(Self.Out, output_tensor, null);
                var_out.asUntagged(Self.Out).setCreator(
                    self.base.func_ptr,
                );
                self.outs[i] = var_out;
                out[i] = var_out;
            }

            self.base.generation = self.in.?.getGeneration();
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            var gy_list: [Self.out]*TaggedVar = undefined;
            for (self.outs, 0..) |out_opt, i| {
                const out = out_opt orelse return error.MissingOutput;
                gy_list[i] = out.asUntaggedConst(Self.Out).grad orelse return error.MissingGradient;
            }

            const gx = try self.backward(&gy_list);

            if (self.in.?.asUntaggedConst(Self.In).grad) |in_grad| {
                self.in.?.setGrad(try addEx(Self.In, in_grad, gx, self.base.chain));
            } else {
                self.in.?.setGrad(gx);
            }
        }

        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.context;

            if (@hasDecl(Self, "predestroy")) {
                self.predestroy();
            }

            self.in = null;
            for (&self.outs) |*out_opt| {
                if (out_opt.*) |out| {
                    out.resetCreator();
                    out_opt.* = null;
                }
            }
            context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            if (self.in.?.asUntaggedConst(Self.In).creator) |creator| {
                if (!seen_set.contains(creator)) {
                    try seen_set.put(creator, {});
                    try queue.add(creator);
                }
            }
        }

        pub fn getDotAlloc(ctx: *anyopaque, var_seen_set: *TaggedVar.SeenSet) ![]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const allocator = self.base.context.allocator;

            const func_label = @typeName(Self)[std.mem.indexOf(u8, @typeName(Self), ".").? + 1 ..];
            const func_node = try std.fmt.allocPrint(allocator, "{} [label=\"{s}\", color=lightblue, style=filled, shape=box]\n", .{
                @intFromPtr(self),
                func_label,
            });
            defer allocator.free(func_node);

            const scalar_label = try std.fmt.allocPrint(allocator, "Scalar: axis={}", .{self.scalar.axis});
            const scalar_node = try std.fmt.allocPrint(allocator, "{} [label=\"{s}\", color=aquamarine, style=filled, shape=circle]\n", .{
                @intFromPtr(&self.scalar),
                scalar_label,
            });
            defer allocator.free(scalar_label);
            defer allocator.free(scalar_node);

            const in_contains = var_seen_set.contains(self.in.?);
            const in_dot = if (!in_contains) try self.in.?.getDotAlloc() else "";
            defer if (!in_contains) allocator.free(in_dot);
            try var_seen_set.put(self.in.?, {});

            var out_dots: [Self.out][]u8 = undefined;
            for (self.outs, 0..) |out_opt, i| {
                if (out_opt) |out| {
                    const out_contains = var_seen_set.contains(out);
                    out_dots[i] = if (!out_contains) try out.getDotAlloc() else "";
                    try var_seen_set.put(out, {});
                } else {
                    out_dots[i] = "";
                }
            }
            defer for (out_dots) |dot| allocator.free(dot);

            var result = try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
                func_node,
                scalar_node,
                in_dot,
            });
            for (out_dots) |dot| {
                result = try std.fmt.allocPrint(allocator, "{s}{s}", .{ result, dot });
            }

            result = try std.fmt.allocPrint(allocator, "{s}{} -> {}\n", .{
                result,
                @intFromPtr(self.in.?),
                @intFromPtr(self),
            });
            result = try std.fmt.allocPrint(allocator, "{s}{} -> {}\n", .{
                result,
                @intFromPtr(&self.scalar),
                @intFromPtr(self),
            });
            for (self.outs) |out_opt| {
                if (out_opt) |out| {
                    result = try std.fmt.allocPrint(allocator, "{s}{} -> {}\n", .{
                        result,
                        @intFromPtr(self),
                        @intFromPtr(out),
                    });
                }
            }

            return result;
        }
    };
}

pub fn Split(comptime n_out: usize, comptime T: type) type {
    return struct {
        in: ?*TaggedVar,
        outs: [n_out]?*TaggedVar,
        scalar: Scalar,
        base: FunctionBase,

        pub const Scalar = struct {
            axis: usize,
        };
        pub const out = n_out;
        pub const In = T;
        pub const Out = T;

        pub usingnamespace FuncDecorator1Scalar1inManyout(Self);

        const Self = @This();

        pub fn forward(self: *Self, x: *const GPUTensor(T)) ![n_out]GPUTensor(T) {
            const allocator = self.base.context.allocator;
            const stream = self.base.context.stream;
            const axis = self.scalar.axis;

            const original_shape = x.base.getShapeConst();
            if (original_shape.len <= axis) return error.InvalidAxis;
            if (original_shape[axis] % n_out != 0) return error.InvalidSplit;

            const split_size = original_shape[axis] / n_out;
            const rank = original_shape.len;

            var outputs: [n_out]GPUTensor(T) = undefined;

            for (0..n_out) |i| {
                var slices: [GPUTensor(T).max_rank]GPUTensor(T).Slice = undefined;
                for (0..rank) |d| {
                    if (d == axis) {
                        const start = i * split_size;
                        const stop = (i + 1) * split_size;
                        slices[d] = .{ .start = @intCast(start), .stop = @intCast(stop), .step = 1 };
                    } else {
                        slices[d] = .all;
                    }
                }
                var split_tensor = try x.getItem(allocator, slices[0..rank], stream);
                outputs[i] = split_tensor.move();
            }

            return outputs;
        }
        pub fn backward(self: *Self, gys: *[n_out]*TaggedVar) !*TaggedVar {
            // Concatenate the output gradients along the original axis
            return try concatEx(n_out, T, gys.*, self.scalar.axis, self.base.chain);
        }
    };
}

pub fn splitEx(comptime n_out: usize, comptime T: type, x: *TaggedVar, axis: usize, chain: *Chain) ![n_out]*TaggedVar {
    const scalar = Split(n_out, T).Scalar{ .axis = axis };
    const func = try Split(n_out, T).create(x.getContext(), scalar, chain);

    var out: [n_out]?*TaggedVar = .{null} ** n_out;
    var in: [1]*TaggedVar = .{x};

    try func.vtable.forward(func.ptr, &in, &out);

    var result: [n_out]*TaggedVar = undefined;
    for (out, 0..) |opt, i| {
        result[i] = opt orelse return error.NullOutput;
    }
    return result;
}

// Tests

fn createOnesTensor(shape: []const usize, chain: *Chain) !*TaggedVar {
    var tensor = try GPUTensor(f32).initAsync(shape, chain.context.stream);
    try tensor.fill(1.0, chain.context.stream);
    return try chain.createVariable(f32, tensor.move(), null);
}

fn testSplit(allocator: std.mem.Allocator) !void {
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
    const shape = &[_]usize{ 2, 4 };
    var input_data = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var gpu_input = try GPUTensor(T).initAsync(shape, &stream);
    defer gpu_input.deinitAsync(&stream);
    try gpu_input.writeFromHostAsync(&input_data, 0, &stream);

    var var_input = try base_chain.createVariable(T, gpu_input.move(), "input");
    defer var_input.destroy();

    var outputs = try splitEx(2, T, var_input, 1, base_chain);
    defer for (outputs) |out| out.destroy();

    var host_outputs = try allocator.alloc(tomo.tensor.CPUTensor(T), outputs.len);
    defer {
        for (host_outputs) |*h| h.deinit(allocator);
        allocator.free(host_outputs);
    }

    for (outputs, 0..) |output, i| {
        host_outputs[i] = try output.asUntagged(T).data.toHost(allocator, &stream);
    }
    try stream.sync();

    const expected0 = [_]T{ 1.0, 2.0, 5.0, 6.0 };
    const expected1 = [_]T{ 3.0, 4.0, 7.0, 8.0 };
    for (host_outputs[0].data, expected0) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }
    for (host_outputs[1].data, expected1) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    var gy0 = try createOnesTensor(outputs[0].getShape(), base_chain);
    defer gy0.destroy();
    var gy1 = try createOnesTensor(outputs[1].getShape(), base_chain);
    defer gy1.destroy();
    outputs[0].setGrad(gy0);
    outputs[1].setGrad(gy1);

    const split_func = outputs[0].asUntagged(T).creator.?;
    try split_func.vtable.backward(split_func.ptr);

    var gx = var_input.refGrad().?;
    var host_gx = try gx.asUntagged(T).data.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);
    try stream.sync();

    for (host_gx.data) |val| {
        if (@abs(val - 1.0) > 1e-6) return error.TestFailed;
    }

    std.debug.print("Split test passed.\n", .{});
}

pub fn test1s1imanyo() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    try testSplit(allocator);
}
