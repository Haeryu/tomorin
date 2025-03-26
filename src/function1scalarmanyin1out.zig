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
const splitEx = @import("function1scalar1inmanyout.zig").splitEx;

pub fn FuncDecorator1ScalarManyin1out(comptime Self: type) type {
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
                .ins = .{null} ** Self.in, // Initialize array with nulls
                .out = null,
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
            if (args.len != Self.in) return error.InvalidInputCount;
            for (args, 0..) |arg, i| {
                self.ins[i] = arg;
            }

            var inputs: [Self.in]GPUTensor(Self.In) = undefined;
            for (self.ins, 0..) |in_opt, i| {
                const in_var = in_opt orelse return error.MissingInput;
                inputs[i] = in_var.asUntaggedConst(Self.In).data;
            }

            const output_tensor = try self.forward(&inputs);

            const var_out = try self.base.chain.createVariable(Self.Out, output_tensor, null);
            var_out.asUntagged(Self.Out).setCreator(self.base.func_ptr);
            self.out = var_out;
            out[0] = var_out;

            // Set generation as the maximum of input generations
            self.base.generation = 0;
            for (self.ins) |in_opt| {
                if (in_opt) |in_var| {
                    self.base.generation = @max(self.base.generation, in_var.getGeneration());
                }
            }
        }

        pub fn backwardDecorated(ctx: *anyopaque) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const gy = self.out.?.asUntaggedConst(Self.Out).grad orelse return error.MissingGradient;

            const gxs = try self.backward(gy);

            for (self.ins, gxs) |in_opt, gx| {
                const in_var = in_opt orelse return error.MissingInput;
                if (in_var.asUntaggedConst(Self.In).grad) |in_grad| {
                    in_var.setGrad(try addEx(Self.In, in_grad, gx, self.base.chain));
                } else {
                    in_var.setGrad(gx);
                }
            }
        }

        pub fn destroy(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const context = self.base.context;

            if (@hasDecl(Self, "predestroy")) {
                self.predestroy();
            }

            for (&self.ins) |*in_opt| {
                in_opt.* = null;
            }
            if (self.out) |out| {
                out.resetCreator();
                self.out = null;
            }
            context.allocator.destroy(self);
        }

        pub fn getGeneration(ctx: *anyopaque) usize {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.base.generation;
        }

        pub fn enqueue(ctx: *anyopaque, queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            for (self.ins) |in_opt| {
                if (in_opt) |in_var| {
                    if (in_var.asUntaggedConst(Self.In).creator) |creator| {
                        if (!seen_set.contains(creator)) {
                            try seen_set.put(creator, {});
                            try queue.add(creator);
                        }
                    }
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

            var in_dots: [Self.in][]u8 = undefined;
            for (self.ins, 0..) |in_opt, i| {
                if (in_opt) |in_var| {
                    const in_contains = var_seen_set.contains(in_var);
                    in_dots[i] = if (!in_contains) try in_var.getDotAlloc() else "";
                    try var_seen_set.put(in_var, {});
                } else {
                    in_dots[i] = "";
                }
            }
            defer for (in_dots) |dot| allocator.free(dot);

            const out_contains = var_seen_set.contains(self.out.?);
            const out_dot = if (!out_contains) try self.out.?.getDotAlloc() else "";
            defer if (!out_contains) allocator.free(out_dot);
            try var_seen_set.put(self.out.?, {});

            var result = try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{
                func_node,
                scalar_node,
                out_dot,
            });
            for (in_dots) |dot| {
                result = try std.fmt.allocPrint(allocator, "{s}{s}", .{ result, dot });
            }

            for (self.ins) |in_opt| {
                if (in_opt) |in_var| {
                    result = try std.fmt.allocPrint(allocator, "{s}{} -> {}\n", .{
                        result,
                        @intFromPtr(in_var),
                        @intFromPtr(self),
                    });
                }
            }
            result = try std.fmt.allocPrint(allocator, "{s}{} -> {}\n", .{
                result,
                @intFromPtr(&self.scalar),
                @intFromPtr(self),
            });
            result = try std.fmt.allocPrint(allocator, "{s}{} -> {}\n", .{
                result,
                @intFromPtr(self),
                @intFromPtr(self.out.?),
            });

            return result;
        }
    };
}

pub fn Concat(comptime n_in: usize, comptime T: type) type {
    return struct {
        ins: [n_in]?*TaggedVar,
        out: ?*TaggedVar,
        scalar: Scalar,
        base: FunctionBase,

        pub const Scalar = struct {
            axis: usize,
        };
        pub const in = n_in;
        pub const Out = T;
        pub const In = T;

        pub usingnamespace FuncDecorator1ScalarManyin1out(Self);

        const Self = @This();

        pub fn forward(self: *Self, xs: *[n_in]GPUTensor(T)) !GPUTensor(T) {
            const allocator = self.base.context.allocator;
            const stream = self.base.context.stream;
            const axis = self.scalar.axis;

            // Check input compatibility
            const rank = xs[0].base.getShapeConst().len;
            for (xs[1..]) |x| {
                if (x.base.getShapeConst().len != rank) return error.ShapeMismatch;
            }

            // Compute output shape
            var output_shape = try allocator.dupe(usize, xs[0].base.getShapeConst());
            defer allocator.free(output_shape);
            for (xs[1..]) |x| {
                for (0..rank) |d| {
                    if (d != axis) {
                        if (x.base.getShapeConst()[d] != output_shape[d]) return error.ShapeMismatch;
                    } else {
                        output_shape[d] += x.base.getShapeConst()[d];
                    }
                }
            }

            // Create output tensor
            var output = try GPUTensor(T).initAsync(output_shape, stream);
            errdefer output.deinitAsync(stream);

            // Concatenate inputs
            var offset: usize = 0;
            for (xs) |x| {
                var slices: [GPUTensor(T).max_rank]GPUTensor(T).Slice = undefined;
                for (0..rank) |d| {
                    if (d == axis) {
                        const size = x.base.getShapeConst()[d];
                        slices[d] = .{ .start = @intCast(offset), .stop = @intCast(offset + size), .step = 1 };
                        offset += size;
                    } else {
                        slices[d] = .all;
                    }
                }
                try output.setItem(allocator, slices[0..rank], &x, stream);
            }

            return output;
        }

        pub fn backward(self: *Self, gy: *TaggedVar) ![n_in]*TaggedVar {
            return try splitEx(n_in, T, gy, self.scalar.axis, self.base.chain);
        }
    };
}

pub fn concatEx(comptime n_in: usize, comptime T: type, xs: [n_in]*TaggedVar, axis: usize, chain: *Chain) !*TaggedVar {
    const scalar = Concat(n_in, T).Scalar{ .axis = axis };
    const func = try Concat(n_in, T).create(chain.context, scalar, chain);

    var out: [1]?*TaggedVar = .{null};
    var in: [n_in]*TaggedVar = xs;

    try func.vtable.forward(func.ptr, &in, &out);

    return out[0] orelse error.NullOutput;
}

fn createOnesTensor(shape: []const usize, chain: *Chain) !*TaggedVar {
    var tensor = try GPUTensor(f32).initAsync(shape, chain.context.stream);
    try tensor.fill(1.0, chain.context.stream);
    return try chain.createVariable(f32, tensor.move(), null);
}

fn testConcat(allocator: std.mem.Allocator) !void {
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
    const shape0 = &[_]usize{ 2, 2 };
    const shape1 = &[_]usize{ 2, 2 };
    var input_data0 = [_]T{ 1.0, 2.0, 5.0, 6.0 };
    var input_data1 = [_]T{ 3.0, 4.0, 7.0, 8.0 };

    var gpu_input0 = try GPUTensor(T).initAsync(shape0, &stream);
    defer gpu_input0.deinitAsync(&stream);
    try gpu_input0.writeFromHostAsync(&input_data0, 0, &stream);

    var gpu_input1 = try GPUTensor(T).initAsync(shape1, &stream);
    defer gpu_input1.deinitAsync(&stream);
    try gpu_input1.writeFromHostAsync(&input_data1, 0, &stream);

    var var_input0 = try base_chain.createVariable(T, gpu_input0.move(), "input0");
    defer var_input0.destroy();
    var var_input1 = try base_chain.createVariable(T, gpu_input1.move(), "input1");
    defer var_input1.destroy();

    // Concatenate along axis 1
    const axis = 1;
    const inputs = [2]*TaggedVar{ var_input0, var_input1 };
    var output = try concatEx(2, T, inputs, axis, base_chain);
    defer output.destroy();

    // Check forward pass
    var host_output = try output.asUntagged(T).data.toHost(allocator, &stream);
    defer host_output.deinit(allocator);
    try stream.sync();

    const expected_output = [_]T{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    for (host_output.data, expected_output) |got, exp| {
        if (@abs(got - exp) > 1e-6) return error.TestFailed;
    }

    // Set gradient for output
    var gy = try createOnesTensor(output.getShape(), base_chain);
    defer gy.destroy();
    output.setGrad(gy);

    // Perform backward pass
    const concat_func = output.asUntagged(T).creator.?;
    try concat_func.vtable.backward(concat_func.ptr);

    // Check gradients for inputs
    var gx0 = var_input0.refGrad().?;
    var host_gx0 = try gx0.asUntagged(T).data.toHost(allocator, &stream);
    defer host_gx0.deinit(allocator);

    var gx1 = var_input1.refGrad().?;
    var host_gx1 = try gx1.asUntagged(T).data.toHost(allocator, &stream);
    defer host_gx1.deinit(allocator);

    try stream.sync();

    for (host_gx0.data) |val| {
        if (@abs(val - 1.0) > 1e-6) return error.TestFailed;
    }
    for (host_gx1.data) |val| {
        if (@abs(val - 1.0) > 1e-6) return error.TestFailed;
    }

    std.debug.print("Concat test passed.\n", .{});
}

pub fn testManyin1out() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    try testConcat(allocator);
}
