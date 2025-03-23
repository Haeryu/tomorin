const std = @import("std");
const TaggedVar = @import("variable.zig").TaggedVar;
const tomo = @import("tomo");
const Context = @import("context.zig").Context;
const Chain = @import("chain.zig").Chain;
const GPUTensor = tomo.tensor.GPUTensor;

const function = @import("function.zig");
const linearEx = function.linearEx;
const broadcastToEx = function.broadcastToEx;
const matmulEx = function.matmulEx;
const conv2DEx = function.conv2DEx;
const reluEx = function.reluEx;
const maxPoolingEx = function.maxPoolingEx;
const reshapeEx = function.reshapeEx;
const dropoutEx = function.dropoutEx;

pub fn LayerFieldsFactory(
    comptime param_names: []const [:0]const u8,
    comptime layer_names_types: []const std.meta.Tuple(&.{ [:0]const u8, type }),
) type {
    var fields_info: [param_names.len + layer_names_types.len]std.builtin.Type.StructField = undefined;

    var i: comptime_int = 0;
    for (param_names) |param_name| {
        fields_info[i] = .{
            .name = param_name,
            .type = ?*TaggedVar,
            .default_value_ptr = null,
            .is_comptime = false,
            .alignment = @alignOf(?*TaggedVar),
        };
        i += 1;
    }
    for (layer_names_types) |layer_name_type| {
        fields_info[i] = .{
            .name = layer_name_type[0],
            .type = layer_name_type[1],
            .default_value_ptr = null,
            .is_comptime = false,
            .alignment = @alignOf(layer_name_type[1]),
        };
        i += 1;
    }

    const fields: std.builtin.Type = .{ .@"struct" = .{
        .layout = .auto,
        .backing_integer = null,
        .fields = &fields_info,
        .decls = &.{},
        .is_tuple = false,
    } };

    return @Type(fields);
}

pub fn LayerDecorator(comptime Self: type) type {
    return struct {
        pub fn calcParamNum() comptime_int {
            const fields_info = @typeInfo(@FieldType(Self, "fields")).@"struct";

            var param_num: comptime_int = 0;
            for (fields_info.fields) |field| {
                if (field.type == ?*TaggedVar) {
                    param_num += 1;
                } else {
                    param_num += field.type.calcParamNum();
                }
            }
            return param_num;
        }

        pub fn getParams(self: *Self) [@This().calcParamNum()]?*TaggedVar {
            const fields_info = @typeInfo(@FieldType(Self, "fields")).@"struct";

            var params: [@This().calcParamNum()]?*TaggedVar = undefined;

            var i: usize = 0;
            inline for (fields_info.fields) |field| {
                if (field.type == ?*TaggedVar) {
                    params[i] = @field(self.fields, field.name);
                    i += 1;
                } else {
                    var field_params = @field(self.fields, field.name).getParams();
                    for (&field_params) |field_param| {
                        params[i] = field_param;
                        i += 1;
                    }
                }
            }

            return params;
        }

        pub fn destroy(self: *Self) void {
            const fields_info = @typeInfo(@FieldType(Self, "fields")).@"struct";

            if (@hasDecl(Self, "destroyElse")) {
                self.destroyElse();
            }

            inline for (fields_info.fields) |field| {
                if (field.type == ?*TaggedVar) {
                    if (@field(self.fields, field.name)) |f| {
                        f.destroy();
                    }
                } else {
                    @field(self.fields, field.name).destroy();
                }
            }
        }

        pub fn clearGrads(self: *Self) void {
            const params = self.getParams();
            for (&params) |param| {
                if (param) |p| {
                    p.setGrad(null);
                }
            }
        }

        fn writeJsonStringParam(name: []const u8, param: ?*TaggedVar, allocator: std.mem.Allocator, writer: anytype) !void {
            try writer.print(
                \\"param.{s}":
            ,
                .{name},
            );
            if (param) |p| {
                try p.writeJsonString(allocator, writer);
            } else {
                try std.json.stringify(null, .{}, writer);
            }
        }

        pub fn writeJsonStringField(self: *const Self, allocator: std.mem.Allocator, writer: anytype) !void {
            const fields_info = @typeInfo(@FieldType(Self, "fields")).@"struct";
            try writer.writeAll("{");
            inline for (fields_info.fields, 0..) |field, i| {
                if (field.type == ?*TaggedVar) {
                    const param = @field(self.fields, field.name);
                    try writeJsonStringParam(field.name, param, allocator, writer);
                } else {
                    var layer = @field(self.fields, field.name);
                    try writer.print(
                        \\"layer.{s}":
                    ,
                        .{field.name},
                    );
                    try layer.writeJsonStringField(allocator, writer);
                }
                if (i != fields_info.fields.len - 1) {
                    try writer.writeAll(",");
                }
            }
            try writer.writeAll("}");
        }

        pub fn saveJsonStringField(self: *const Self, allocator: std.mem.Allocator, sub_path: []const u8) !void {
            var file = try std.fs.cwd().createFile(sub_path, .{});
            defer file.close();

            var buf_writer = std.io.bufferedWriter(file.writer());

            try self.writeJsonStringField(allocator, file.writer());

            try buf_writer.flush();
        }

        fn readJsonValueField(self: *Self, allocator: std.mem.Allocator, value: std.json.Value) !void {
            if (value != .object) return error.NotObject;

            var it = value.object.iterator();

            const fields_info = @typeInfo(@FieldType(Self, "fields")).@"struct";

            while (it.next()) |entry| {
                if (std.mem.startsWith(u8, entry.key_ptr.*, "param.")) {
                    inline for (fields_info.fields) |field| {
                        if (field.type == ?*TaggedVar and std.mem.eql(u8, field.name, entry.key_ptr.*["param.".len..])) {
                            if (entry.value_ptr.* == .null) {
                                @field(self.fields, field.name) = null;
                            } else {
                                try @field(self.fields, field.name).?.readJsonValue(allocator, entry.value_ptr.*);
                            }
                        }
                    }
                } else if (std.mem.startsWith(u8, entry.key_ptr.*, "layer.")) {
                    inline for (fields_info.fields) |field| {
                        if (field.type != ?*TaggedVar and std.mem.eql(u8, field.name, entry.key_ptr.*["layer.".len..])) {
                            try @field(self.fields, field.name).readJsonValueField(allocator, entry.value_ptr.*);
                        }
                    }
                } else {
                    return error.UnknownField;
                }
            }
        }

        pub fn readJsonStringField(self: *Self, allocator: std.mem.Allocator, src: []const u8) !void {
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, src, .{});
            defer parsed.deinit();

            const value = parsed.value;

            try self.readJsonValueField(allocator, value);
        }

        pub fn loadJsonStringField(self: *Self, allocator: std.mem.Allocator, sub_path: []const u8) !void {
            var file = try std.fs.cwd().openFile(sub_path, .{ .mode = .read_only });
            defer file.close();

            var bufreader = std.io.bufferedReader(file.reader());
            var reader = bufreader.reader();

            const src = try reader.readAllAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(src);

            try self.readJsonStringField(allocator, src);
        }
    };
}

// TODO plplot? raylib??
// pub fn Model1in(comptime Self: type) type {
//     return struct {
//         pub fn plot(self: *Self, x: *TaggedVar, file_name: []const u8) !void {}
//     };
// }

pub fn Linear(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{
                "w",
                "b",
            },
            &.{},
        ),
        in_size: ?usize,
        out_size: usize,
        context: *Context,
        chain: *Chain,
        winit: WInit,

        pub const WInit = enum {
            xavier,
            he_normal,
            he_uniform,
        };

        const Self = @This();

        pub fn init(
            in_size: ?usize,
            out_size: usize,
            no_bias: bool,
            winit: WInit,
            context: *Context,
            chain: *Chain,
        ) !Self {
            const b = if (no_bias) null else blk: {
                var b_tensor: GPUTensor(T) = try .initAsync(&.{ 1, out_size }, context.stream);
                errdefer b_tensor.deinitAsync(context.stream);
                try b_tensor.fill(0.0, context.stream);
                break :blk try chain.createVariable(T, b_tensor.move(), "b");
            };

            var self: Self = .{
                .fields = .{
                    .w = null,
                    .b = b,
                },
                .in_size = in_size,
                .out_size = out_size,
                .context = context,
                .chain = chain,
                .winit = winit,
            };
            errdefer self.destroy();

            if (in_size != null) {
                self.fields.w = try self.initW();
            }

            return self;
        }

        fn initW(self: *Self) !*TaggedVar {
            var w_tensor: GPUTensor(T) = try .initAsync(&.{ self.in_size.?, self.out_size }, self.context.stream);
            errdefer w_tensor.deinitAsync(self.context.stream);
            // try w_tensor.fillNormalDistribution(0.0, 1.0, self.context.cuda_context, self.context.stream);
            // try w_tensor.scale(1.0 / @as(T, @floatFromInt(self.in_size.?)), self.context.stream);

            switch (self.winit) {
                .xavier => try w_tensor.fillXavierUniform(self.context.cuda_context, self.context.stream),
                .he_normal => try w_tensor.fillHeNormal(self.context.cuda_context, self.context.stream),
                .he_uniform => try w_tensor.fillHeUniform(self.context.cuda_context, self.context.stream),
            }

            const w = try self.chain.createVariable(T, w_tensor.move(), "w");
            return w;
        }

        pub fn forward(
            self: *Self,
            x: *TaggedVar,
            chain: *Chain,
        ) !*TaggedVar {
            if (self.fields.w == null) {
                self.in_size = x.getCol();
                self.fields.w = try self.initW();
            }

            return if (self.fields.b) |b| try linearEx(
                T,
                x,
                self.fields.w.?,
                try broadcastToEx(
                    T,
                    b,
                    &.{ x.getRow(), self.fields.w.?.getCol() },
                    chain,
                ),
                chain,
            ) else try matmulEx(
                T,
                x,
                self.fields.w.?,
                chain,
            );
        }
    };
}

pub fn MLP(
    comptime T: type,
    comptime layers_count: comptime_int,
    // putting activation fn here makes compile error(with no error message...). lol
) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &makeFields(),
        ),

        const Self = @This();

        fn makeFields() [layers_count]std.meta.Tuple(&.{ [:0]const u8, type }) {
            var fields: [layers_count]std.meta.Tuple(&.{ [:0]const u8, type }) = undefined;
            for (&fields, 0..) |*field, i| {
                field.* = .{ std.fmt.comptimePrint("l{}", .{i}), Linear(T) };
            }
            return fields;
        }

        pub fn init(
            out_sizes: *const [layers_count]usize,
            winit: Linear(T).WInit,
            context: *Context,
            chain: *Chain,
        ) !Self {
            var self: Self = undefined;
            inline for (0..layers_count) |i| {
                errdefer {
                    inline for (0..i) |j| {
                        @field(self.fields, std.fmt.comptimePrint("l{}", .{j})).destroy();
                    }
                }
                @field(self.fields, std.fmt.comptimePrint("l{}", .{i})) = try .init(
                    null,
                    out_sizes[i],
                    false,
                    winit,
                    context,
                    chain,
                );
            }

            return self;
        }

        pub fn forward(
            self: *Self,
            x: *TaggedVar,
            activation: *const fn (comptime T: type, x: *TaggedVar, chain: *Chain) anyerror!*TaggedVar,
            chain: *Chain,
        ) !*TaggedVar {
            var y: *TaggedVar = x;
            inline for (0..layers_count - 1) |i| {
                y = try @field(self.fields, std.fmt.comptimePrint("l{}", .{i})).forward(y, chain);
                y = try activation(T, y, chain);
            }
            return try @field(self.fields, std.fmt.comptimePrint("l{}", .{layers_count - 1})).forward(y, chain);
        }
    };
}

pub fn Conv2d(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{
                "w",
                "b",
            },
            &.{},
        ),
        in_channels: ?usize,
        out_channels: usize,
        kernel_size: [2]usize,
        stride: [2]usize,
        padding: [2]usize,
        dilation: [2]usize,
        winit: WInit,
        context: *Context,
        chain: *Chain,

        const Self = @This();

        pub const WInit = enum {
            xavier,
            he_normal,
            he_uniform,
        };

        pub fn init(
            in_channels: ?usize,
            out_channels: usize,
            kernel_size: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            dilation: [2]usize,
            no_bias: bool,
            winit: WInit,
            context: *Context,
            chain: *Chain,
        ) !Self {
            const b = if (no_bias) null else blk: {
                var b_tensor: GPUTensor(T) = try .initAsync(&.{ 1, out_channels, 1, 1 }, context.stream);
                errdefer b_tensor.deinitAsync(context.stream);
                try b_tensor.fill(0.0, context.stream);
                break :blk try chain.createVariable(T, b_tensor.move(), "b");
            };

            var self: Self = .{
                .fields = .{
                    .w = null,
                    .b = b,
                },
                .in_channels = in_channels,
                .out_channels = out_channels,
                .kernel_size = kernel_size,
                .stride = stride,
                .padding = padding,
                .dilation = dilation,
                .context = context,
                .winit = winit,
                .chain = chain,
            };
            errdefer self.destroy();

            if (in_channels != null) {
                self.fields.w = try self.initW();
            }

            return self;
        }

        pub fn initW(self: *Self) !*TaggedVar {
            const c = self.in_channels.?;
            const oc = self.out_channels;
            const kh, const kw = self.kernel_size;
            const scale = std.math.sqrt(1.0 / @as(T, @floatFromInt(c * kh * kw)));

            var w: GPUTensor(T) = try .initAsync(
                &.{ oc, c, kh, kw },
                self.context.stream,
            );
            errdefer w.deinitAsync(self.context.stream);

            switch (self.winit) {
                .he_normal => try w.fillHeNormal(self.context.cuda_context, self.context.stream),
                .he_uniform => try w.fillHeUniform(self.context.cuda_context, self.context.stream),
                .xavier => try w.fillXavierUniform(self.context.cuda_context, self.context.stream),
            }

            try w.scale(scale, self.context.stream);

            return try self.chain.createVariable(T, w.move(), "w");
        }

        pub fn forward(self: *Self, x: *TaggedVar, chain: *Chain) !*TaggedVar {
            if (self.fields.w == null) {
                self.in_channels = x.getShape()[1];
                self.fields.w = try self.initW();
            }

            return try conv2DEx(
                T,
                x,
                self.fields.w.?,
                self.fields.b.?,
                .{
                    .dilation = self.dilation,
                    .padding = self.padding,
                    .stride = self.stride,
                },
                chain,
            );
        }
    };
}

pub fn VGG16(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &.{
                .{ "conv1_1", Conv2d(T) },
                .{ "conv1_2", Conv2d(T) },
                .{ "conv2_1", Conv2d(T) },
                .{ "conv2_2", Conv2d(T) },
                .{ "conv3_1", Conv2d(T) },
                .{ "conv3_2", Conv2d(T) },
                .{ "conv3_3", Conv2d(T) },
                .{ "conv4_1", Conv2d(T) },
                .{ "conv4_2", Conv2d(T) },
                .{ "conv4_3", Conv2d(T) },
                .{ "conv5_1", Conv2d(T) },
                .{ "conv5_2", Conv2d(T) },
                .{ "conv5_3", Conv2d(T) },
                .{ "fc6", Linear(T) },
                .{ "fc7", Linear(T) },
                .{ "fc8", Linear(T) },
            },
        ),

        const Self = @This();

        pub fn init(context: *Context, chain: *Chain) !Self {
            var conv1_1: Conv2d(T) = try .init(
                null,
                64,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv1_1.destroy();
            var conv1_2: Conv2d(T) = try .init(
                null,
                64,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv1_2.destroy();
            var conv2_1: Conv2d(T) = try .init(
                null,
                128,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv2_1.destroy();
            var conv2_2: Conv2d(T) = try .init(
                null,
                128,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv2_2.destroy();
            var conv3_1: Conv2d(T) = try .init(
                null,
                256,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv3_1.destroy();
            var conv3_2: Conv2d(T) = try .init(
                null,
                256,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv3_2.destroy();
            var conv3_3: Conv2d(T) = try .init(
                null,
                256,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv3_3.destroy();
            var conv4_1: Conv2d(T) = try .init(
                null,
                512,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv4_1.destroy();
            var conv4_2: Conv2d(T) = try .init(
                null,
                512,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv4_2.destroy();
            var conv4_3: Conv2d(T) = try .init(
                null,
                512,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv4_3.destroy();
            var conv5_1: Conv2d(T) = try .init(
                null,
                512,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv5_1.destroy();
            var conv5_2: Conv2d(T) = try .init(
                null,
                512,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv5_2.destroy();
            var conv5_3: Conv2d(T) = try .init(
                null,
                512,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv5_3.destroy();
            var fc6: Linear(T) = try .init(
                null,
                // 4096,
                512,
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer fc6.destroy();
            var fc7: Linear(T) = try .init(
                null,
                // 4096,
                512,
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer fc7.destroy();
            var fc8: Linear(T) = try .init(
                null,
                // 1000,
                10,
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer fc8.destroy();

            return .{
                .fields = .{
                    .conv1_1 = conv1_1,
                    .conv1_2 = conv1_2,
                    .conv2_1 = conv2_1,
                    .conv2_2 = conv2_2,
                    .conv3_1 = conv3_1,
                    .conv3_2 = conv3_2,
                    .conv3_3 = conv3_3,
                    .conv4_1 = conv4_1,
                    .conv4_2 = conv4_2,
                    .conv4_3 = conv4_3,
                    .conv5_1 = conv5_1,
                    .conv5_2 = conv5_2,
                    .conv5_3 = conv5_3,
                    .fc6 = fc6,
                    .fc7 = fc7,
                    .fc8 = fc8,
                },
            };
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            var y = try reluEx(T, try self.fields.conv1_1.forward(x, chain), chain);
            y = try reluEx(T, try self.fields.conv1_2.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 2, 2 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reluEx(T, try self.fields.conv2_1.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv2_2.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 2, 2 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reluEx(T, try self.fields.conv3_1.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv3_2.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv3_3.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 2, 2 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reluEx(T, try self.fields.conv4_1.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv4_2.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv4_3.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 2, 2 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reluEx(T, try self.fields.conv5_1.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv5_2.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv5_3.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 2, 2 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reshapeEx(T, y, &.{ y.getShape()[0], y.getShape()[1] * y.getShape()[2] * y.getShape()[3] }, chain);

            y = try dropoutEx(T, try reluEx(T, try self.fields.fc6.forward(y, chain), chain), 0.5, train, chain);
            y = try dropoutEx(T, try reluEx(T, try self.fields.fc7.forward(y, chain), chain), 0.5, train, chain);
            y = try self.fields.fc8.forward(y, chain);

            return y;
        }
    };
}
