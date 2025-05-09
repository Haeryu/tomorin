const std = @import("std");
const TaggedVar = @import("variable.zig").TaggedVar;
const tomo = @import("tomo");
const BF16 = tomo.BF16;
const Context = @import("context.zig").Context;
const Chain = @import("chain.zig").Chain;
const GPUTensor = tomo.tensor.GPUTensor;

const function = @import("function.zig");
const linearEx = function.linearEx;
const broadcastToEx = function.broadcastToEx;
const matmulEx = function.matmulEx;
const conv2DEx = function.conv2DEx;
const conv2DNoBiasEx = function.conv2DNoBiasEx;
const reluEx = function.reluEx;
const maxPoolingEx = function.maxPoolingEx;
const reshapeEx = function.reshapeEx;
const dropoutEx = function.dropoutEx;
const batchNormEx = function.batchNormEx;
const layerNormEx = function.layerNormEx;
const averagePoolingEx = function.averagePoolingEx;
const addEx = function.addEx;
const embedEx = function.embedEx;
const splitEx = function.splitEx;
const transposeExEx = function.transposeExEx;
const scaleEx = function.scaleEx;
const maskedFillEx = function.maskedFillEx;
const softmaxEx = function.softmaxEx;
const geluEx = function.geluEx;
const getItemEx = function.getItemEx;
const castEx = function.castEx;

const dbg = @import("util.zig").debugPrintGpuTensor;

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

            if (@hasDecl(Self, "predestroy")) {
                self.predestroy();
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
            const writer = buf_writer.writer();

            try self.writeJsonStringField(allocator, writer);

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

        pub fn writeBinary(self: *Self, allocator: std.mem.Allocator, writer: anytype) !void {
            const params = self.getParams();
            for (params) |param| {
                if (param) |p| {
                    try p.writeBinary(allocator, writer);
                }
            }
        }

        pub fn saveBinary(self: *Self, allocator: std.mem.Allocator, sub_path: []const u8) !void {
            var file = try std.fs.cwd().createFile(sub_path, .{});
            defer file.close();

            var buf_writer = std.io.bufferedWriter(file.writer());
            const writer = buf_writer.writer();

            try self.writeBinary(allocator, writer);

            try buf_writer.flush();
        }

        pub fn readBinary(self: *Self, src: []const u8) !void {
            const params = self.getParams();
            var next = src;
            for (params) |param| {
                if (param) |p| {
                    next = try p.readBinary(next);
                }
            }

            if (next.len != 0) {
                return error.DataLeft;
            }
        }

        pub fn loadBinary(self: *Self, allocator: std.mem.Allocator, sub_path: []const u8) !void {
            var file = try std.fs.cwd().openFile(sub_path, .{ .mode = .read_only });
            defer file.close();

            var bufreader = std.io.bufferedReader(file.reader());
            var reader = bufreader.reader();

            const src = try reader.readAllAlloc(allocator, std.math.maxInt(usize));
            defer allocator.free(src);

            try self.readBinary(src);
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
            normal,
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

        pub fn fullInit(
            in_size: usize,
            out_size: usize,
            no_bias: bool,
            winit: WInit,
            context: *Context,
            chain: *Chain,
        ) !Self {
            var self: Self = try .init(in_size, out_size, no_bias, winit, context, chain);
            errdefer self.destroy();
            self.fields.w = try self.initW();

            return self;
        }

        fn initW(self: *Self) !*TaggedVar {
            var w_tensor: GPUTensor(T) = try .initAsync(&.{ self.in_size.?, self.out_size }, self.context.stream);
            errdefer w_tensor.deinitAsync(self.context.stream);
            // try w_tensor.fillNormalDistribution(0.0, 1.0, self.context.cuda_context, self.context.stream);
            // try w_tensor.scale(1.0 / @as(T, @floatFromInt(self.in_size.?)), self.context.stream);

            switch (self.winit) {
                .normal => try w_tensor.fillNormalDistribution(0.0, 1.0, self.context.cuda_context, self.context.stream),
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

            if (x.getShape().len == 2) {
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
            const x_flat = try reshapeEx(T, x, &.{ x.asUntagged(T).data.calcLen() / x.getCol(), x.getCol() }, chain);

            const out_flat = if (self.fields.b) |b| try linearEx(
                T,
                x_flat,
                self.fields.w.?,
                try broadcastToEx(
                    T,
                    b,
                    &.{ x.asUntagged(T).data.calcLen() / x.getCol(), self.fields.w.?.getCol() },
                    chain,
                ),
                chain,
            ) else try matmulEx(
                T,
                x_flat,
                self.fields.w.?,
                chain,
            );
            var out_shape: [GPUTensor(T).max_rank]usize = undefined;
            for (x.getShape(), 0..) |xs, i| {
                out_shape[i] = xs;
            }
            out_shape[x.getShape().len - 1] = out_flat.getCol();
            const out = try reshapeEx(T, out_flat, out_shape[0..x.getShape().len], chain);

            return out;
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

        pub fn fullInit(
            in_out_sizes: *const [layers_count][2]usize,
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
                @field(self.fields, std.fmt.comptimePrint("l{}", .{i})) = try .fullInit(
                    in_out_sizes[i][0],
                    in_out_sizes[i][1],
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

        pub fn fullInit(
            in_channels: usize,
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
            var self: Self = try .init(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                no_bias,
                winit,
                context,
                chain,
            );
            errdefer self.destroy();
            self.fields.w = try self.initW();
            return self;
        }

        pub fn initW(self: *Self) !*TaggedVar {
            const c = self.in_channels.?;
            const oc = self.out_channels;
            const kh, const kw = self.kernel_size;

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

            return try self.chain.createVariable(T, w.move(), "w");
        }

        pub fn forward(self: *Self, x: *TaggedVar, chain: *Chain) !*TaggedVar {
            if (self.fields.w == null) {
                self.in_channels = x.getShape()[1];
                self.fields.w = try self.initW();
            }

            if (self.fields.b) |b| {
                return try conv2DEx(
                    T,
                    x,
                    self.fields.w.?,
                    b,
                    .{
                        .dilation = self.dilation,
                        .padding = self.padding,
                        .stride = self.stride,
                    },
                    chain,
                );
            } else {
                return try conv2DNoBiasEx(
                    T,
                    x,
                    self.fields.w.?,
                    .{
                        .dilation = self.dilation,
                        .padding = self.padding,
                        .stride = self.stride,
                    },
                    chain,
                );
            }
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
                .{ "out", Linear(T) },
            },
        ),

        const Self = @This();

        pub fn init(out_size: usize, context: *Context, chain: *Chain) !Self {
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
                4096,
                // 512,
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer fc6.destroy();
            var fc7: Linear(T) = try .init(
                null,
                4096,
                // 512,
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer fc7.destroy();
            var fc8: Linear(T) = try .init(
                null,
                1000,
                //100,
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer fc8.destroy();
            var out: Linear(T) = try .init(
                null,
                out_size,
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer out.destroy();

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
                    .out = out,
                },
            };
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            var y = try reluEx(T, try self.fields.conv1_1.forward(x, chain), chain);
            y = try reluEx(T, try self.fields.conv1_2.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 0, 0 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reluEx(T, try self.fields.conv2_1.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv2_2.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 0, 0 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reluEx(T, try self.fields.conv3_1.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv3_2.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv3_3.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 0, 0 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reluEx(T, try self.fields.conv4_1.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv4_2.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv4_3.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 0, 0 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reluEx(T, try self.fields.conv5_1.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv5_2.forward(y, chain), chain);
            y = try reluEx(T, try self.fields.conv5_3.forward(y, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 2, 2 },
                .padding = .{ 0, 0 },
                .stride = .{ 2, 2 },
            }, chain);

            y = try reshapeEx(T, y, &.{ y.getShape()[0], y.getShape()[1] * y.getShape()[2] * y.getShape()[3] }, chain);

            y = try dropoutEx(T, try reluEx(T, try self.fields.fc6.forward(y, chain), chain), 0.5, train, chain);
            y = try dropoutEx(T, try reluEx(T, try self.fields.fc7.forward(y, chain), chain), 0.5, train, chain);

            y = try dropoutEx(T, try reluEx(T, try self.fields.fc8.forward(y, chain), chain), 0.5, train, chain);

            // y = try reluEx(T, try self.fields.fc6.forward(y, chain), chain);
            // y = try reluEx(T, try self.fields.fc7.forward(y, chain), chain);
            // y = try reluEx(T, try self.fields.fc8.forward(y, chain), chain);
            y = try self.fields.out.forward(y, chain);

            // try dbg(T, &y.asUntaggedConst(T).data, y.getContext());

            return y;
        }
    };
}

pub fn BatchNorm(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{
                "avg_mean",
                "avg_var",
                "gamma",
                "beta",
            },
            &.{},
        ),
        context: *Context,
        chain: *Chain,
        const Self = @This();

        pub fn init(
            context: *Context,
            chain: *Chain,
        ) Self {
            return .{
                .fields = .{
                    .avg_mean = null,
                    .avg_var = null,
                    .gamma = null,
                    .beta = null,
                },
                .context = context,
                .chain = chain,
            };
        }

        pub fn initParams(self: *Self, x: *TaggedVar) !void {
            const allocator = self.context.allocator;
            const x_shape = x.getShape();
            var axes: []const isize = undefined;
            if (x_shape.len == 4) {
                axes = &.{ 0, 2, 3 }; // Batch, Height, Width for 4D inputs
            } else if (x_shape.len == 2) {
                axes = &.{0}; // Batch for 2D inputs
            } else {
                return error.UnsupportedInputShape;
            }
            const keepdims = try GPUTensor(T).computeOutShape(allocator, x_shape, axes, true);
            defer allocator.free(keepdims);

            if (self.fields.avg_mean == null) {
                var zero: GPUTensor(T) = try .initAsync(keepdims, self.context.stream);
                errdefer zero.deinitAsync(self.context.stream);
                try zero.fill(0.0, self.context.stream);

                self.fields.avg_mean = try self.chain.createVariable(T, zero.move(), "avg_mean");
            }
            if (self.fields.avg_var == null) {
                var one: GPUTensor(T) = try .initAsync(keepdims, self.context.stream);
                errdefer one.deinitAsync(self.context.stream);
                try one.fill(1.0, self.context.stream);

                self.fields.avg_var = try self.chain.createVariable(T, one.move(), "avg_var");
            }
            if (self.fields.gamma == null) {
                var one: GPUTensor(T) = try .initAsync(keepdims, self.context.stream);
                errdefer one.deinitAsync(self.context.stream);
                try one.fill(1.0, self.context.stream);

                self.fields.gamma = try self.chain.createVariable(T, one.move(), "gamma");
            }
            if (self.fields.beta == null) {
                var zero: GPUTensor(T) = try .initAsync(keepdims, self.context.stream);
                errdefer zero.deinitAsync(self.context.stream);
                try zero.fill(0.0, self.context.stream);

                self.fields.beta = try self.chain.createVariable(T, zero.move(), "beta");
            }
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            if (self.fields.avg_mean == null) {
                try self.initParams(x);
            }

            return try batchNormEx(
                T,
                x,
                self.fields.gamma.?,
                self.fields.beta.?,
                .{
                    .context = self.context,
                    .running_mean = self.fields.avg_mean.?,
                    .running_variance = self.fields.avg_var.?,
                    .train = train,
                },
                chain,
            );
        }
    };
}

pub fn LayerNorm(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{
                "gamma",
                "beta",
            },
            &.{},
        ),
        context: *Context,
        chain: *Chain,
        const Self = @This();

        pub fn init(
            context: *Context,
            chain: *Chain,
        ) Self {
            return .{
                .fields = .{
                    .gamma = null,
                    .beta = null,
                },
                .context = context,
                .chain = chain,
            };
        }

        pub fn initParams(self: *Self, x: *TaggedVar) !void {
            const allocator = self.context.allocator;
            const x_shape = x.getShape();
            var axes: []const isize = undefined;
            if (x_shape.len == 4) {
                // 4D input: [batch, channels, height, width]
                axes = &.{ 0, 2, 3 };
            } else if (x_shape.len == 3) {
                // 3D input: [batch, sequence_length, features]
                axes = &.{ 0, 1 };
            } else if (x_shape.len == 2) {
                // 2D input: [batch, features]
                axes = &.{0};
            } else {
                return error.UnsupportedInputShape;
            }
            const keepdims = try GPUTensor(T).computeOutShape(allocator, x_shape, axes, true);
            defer allocator.free(keepdims);

            if (self.fields.gamma == null) {
                var one: GPUTensor(T) = try .initAsync(keepdims, self.context.stream);
                errdefer one.deinitAsync(self.context.stream);
                try one.fill(1.0, self.context.stream);

                self.fields.gamma = try self.chain.createVariable(T, one.move(), "gamma");
            }
            if (self.fields.beta == null) {
                var zero: GPUTensor(T) = try .initAsync(keepdims, self.context.stream);
                errdefer zero.deinitAsync(self.context.stream);
                try zero.fill(0.0, self.context.stream);

                self.fields.beta = try self.chain.createVariable(T, zero.move(), "beta");
            }
        }

        pub fn forward(self: *Self, x: *TaggedVar, eps: if (T != BF16) T else f32, chain: *Chain) !*TaggedVar {
            if (self.fields.gamma == null) {
                try self.initParams(x);
            }

            return try layerNormEx(
                T,
                x,
                self.fields.gamma.?,
                self.fields.beta.?,
                .{
                    .eps = eps,
                    .context = self.context,
                },
                chain,
            );
        }
    };
}

// quick and dirty impl
pub fn LayerNormMixed(comptime T: type, comptime U: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{
                "gamma",
                "beta",
            },
            &.{},
        ),
        context: *Context,
        chain: *Chain,
        const Self = @This();

        pub fn init(
            context: *Context,
            chain: *Chain,
        ) Self {
            return .{
                .fields = .{
                    .gamma = null,
                    .beta = null,
                },
                .context = context,
                .chain = chain,
            };
        }

        pub fn initParams(self: *Self, x: *TaggedVar) !void {
            const allocator = self.context.allocator;
            const x_shape = x.getShape();
            var axes: []const isize = undefined;
            if (x_shape.len == 4) {
                // 4D input: [batch, channels, height, width]
                axes = &.{ 0, 2, 3 };
            } else if (x_shape.len == 3) {
                // 3D input: [batch, sequence_length, features]
                axes = &.{ 0, 1 };
            } else if (x_shape.len == 2) {
                // 2D input: [batch, features]
                axes = &.{0};
            } else {
                return error.UnsupportedInputShape;
            }
            const keepdims = try GPUTensor(T).computeOutShape(allocator, x_shape, axes, true);
            defer allocator.free(keepdims);

            if (self.fields.gamma == null) {
                var one: GPUTensor(T) = try .initAsync(keepdims, self.context.stream);
                errdefer one.deinitAsync(self.context.stream);
                try one.fill(1.0, self.context.stream);

                self.fields.gamma = try self.chain.createVariable(T, one.move(), "gamma");
            }
            if (self.fields.beta == null) {
                var zero: GPUTensor(T) = try .initAsync(keepdims, self.context.stream);
                errdefer zero.deinitAsync(self.context.stream);
                try zero.fill(0.0, self.context.stream);

                self.fields.beta = try self.chain.createVariable(T, zero.move(), "beta");
            }
        }

        pub fn forward(self: *Self, x: *TaggedVar, eps: if (T != BF16) T else f32, chain: *Chain) !*TaggedVar {
            if (self.fields.gamma == null) {
                try self.initParams(x);
            }

            const x_cast = try castEx(T, U, x, chain);
            const gamma_cast = try castEx(T, U, self.fields.gamma.?, chain);
            const beta_cast = try castEx(T, U, self.fields.beta.?, chain);

            var res = try layerNormEx(
                U,
                x_cast,
                gamma_cast,
                beta_cast,
                .{
                    .eps = eps,
                    .context = self.context,
                },
                chain,
            );
            res = try castEx(U, T, res, chain);

            return res;
        }
    };
}

pub fn Dropout(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &.{},
        ),
        dropout_ratio: if (T != BF16) T else f32,
        const Self = @This();

        pub fn init(dropout_ratio: if (T != BF16) T else f32) Self {
            return .{
                .fields = .{},
                .dropout_ratio = dropout_ratio,
            };
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            return try dropoutEx(
                T,
                x,
                self.dropout_ratio,
                train,
                chain,
            );
        }
    };
}

pub fn ResNet(comptime T: type, comptime layers_count: comptime_int) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &.{
                .{ "conv1", Conv2d(T) },
                .{ "bn1", BatchNorm(T) },
                .{ "res2", BuildingBlock(T, block[0]) },
                .{ "res3", BuildingBlock(T, block[1]) },
                .{ "res4", BuildingBlock(T, block[2]) },
                .{ "res5", BuildingBlock(T, block[3]) },
                .{ "fc6", Linear(T) },
                .{ "out", Linear(T) },
            },
        ),
        const Self = @This();

        const block: [4]comptime_int = switch (layers_count) {
            50 => .{ 3, 4, 6, 3 },
            101 => .{ 3, 4, 23, 3 },
            152 => .{ 3, 8, 36, 3 },
            else => unreachable,
        };

        pub fn init(
            out_num: usize,
            context: *Context,
            chain: *Chain,
        ) !Self {
            var conv1: Conv2d(T) = try .init(
                null,
                64,
                .{ 7, 7 },
                .{ 2, 2 },
                .{ 3, 3 },
                .{ 1, 1 },
                false,
                .he_normal,
                context,
                chain,
            );
            errdefer conv1.destroy();

            var bn1: BatchNorm(T) = .init(context, chain);
            errdefer bn1.destroy();

            var res2: BuildingBlock(T, block[0]) = try .init(64, 64, 256, .{ 1, 1 }, false, context, chain);
            errdefer res2.destroy();

            var res3: BuildingBlock(T, block[1]) = try .init(256, 128, 512, .{ 2, 2 }, false, context, chain);
            errdefer res3.destroy();

            var res4: BuildingBlock(T, block[2]) = try .init(512, 256, 1024, .{ 2, 2 }, false, context, chain);
            errdefer res4.destroy();

            var res5: BuildingBlock(T, block[3]) = try .init(1024, 512, 2048, .{ 2, 2 }, false, context, chain);
            errdefer res5.destroy();

            var fc6: Linear(T) = try .init(
                null,
                1000,
                false,
                .he_uniform,
                context,
                chain,
            );
            errdefer fc6.destroy();

            var out: Linear(T) = try .init(
                null,
                out_num,
                false,
                .he_uniform,
                context,
                chain,
            );
            errdefer out.destroy();

            return .{
                .fields = .{
                    .conv1 = conv1,
                    .bn1 = bn1,
                    .res2 = res2,
                    .res3 = res3,
                    .res4 = res4,
                    .res5 = res5,
                    .fc6 = fc6,
                    .out = out,
                },
            };
        }
        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            var y = try reluEx(T, try self.fields.bn1.forward(try self.fields.conv1.forward(x, chain), train, chain), chain);
            y = try maxPoolingEx(T, y, .{
                .kernel_size = .{ 3, 3 },
                .stride = .{ 2, 2 },
                .padding = .{ 0, 0 },
            }, chain);
            y = try self.fields.res2.forward(y, train, chain);
            y = try self.fields.res3.forward(y, train, chain);
            y = try self.fields.res4.forward(y, train, chain);
            y = try self.fields.res5.forward(y, train, chain);
            y = try globalAvgPooling(T, y, chain);
            y = try self.fields.fc6.forward(y, chain);
            y = try reluEx(T, y, chain);
            y = try self.fields.out.forward(y, chain);

            return y;
        }
    };
}

pub fn ResNet152(comptime T: type) type {
    return ResNet(T, 152);
}

pub fn ResNet101(comptime T: type) type {
    return ResNet(T, 101);
}

pub fn ResNet50(comptime T: type) type {
    return ResNet(T, 50);
}

fn globalAvgPooling(comptime T: type, x: *TaggedVar, chain: *Chain) !*TaggedVar {
    const x_shape = x.getShape();
    const n = x_shape[0];
    const c = x_shape[1];
    const h = x_shape[2];
    const w = x_shape[3];

    var t = try averagePoolingEx(T, x, .{
        .kernel_size = .{ h, w },
        .stride = .{ 1, 1 },
        .padding = .{ 0, 0 },
    }, chain);
    t = try reshapeEx(T, t, &.{ n, c }, chain);

    return t;
}

fn BuildingBlock(comptime T: type, comptime layers_count: comptime_int) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &makeFields(),
        ),
        const Self = @This();

        fn makeFields() [layers_count + 1]std.meta.Tuple(&.{ [:0]const u8, type }) {
            var fields: [layers_count + 1]std.meta.Tuple(&.{ [:0]const u8, type }) = undefined;
            fields[0] = .{ "a", BottleneckA(T) };
            for (fields[1..], 1..) |*field, i| {
                field.* = .{ std.fmt.comptimePrint("b{}", .{i}), BottleneckB(T) };
            }
            return fields;
        }

        pub fn init(
            in_channels: usize,
            mid_channels: usize,
            out_channels: usize,
            stride: [2]usize,
            downsample_fb: bool,
            context: *Context,
            chain: *Chain,
        ) !Self {
            var self: Self = undefined;
            self.fields.a = try BottleneckA(T).init(
                in_channels,
                mid_channels,
                out_channels,
                stride,
                downsample_fb,
                context,
                chain,
            );
            errdefer self.fields.a.destroy();

            inline for (1..layers_count + 1) |i| {
                errdefer {
                    inline for (1..i) |j| {
                        @field(self.fields, std.fmt.comptimePrint("b{}", .{j})).destroy();
                    }
                }
                @field(self.fields, std.fmt.comptimePrint("b{}", .{i})) = try BottleneckB(T).init(
                    out_channels,
                    mid_channels,
                    context,
                    chain,
                );
            }

            return self;
        }

        pub fn forward(
            self: *Self,
            x: *TaggedVar,
            train: bool,
            chain: *Chain,
        ) !*TaggedVar {
            var y: *TaggedVar = try self.fields.a.forward(x, train, chain);
            inline for (1..layers_count + 1) |i| {
                y = try @field(self.fields, std.fmt.comptimePrint("b{}", .{i})).forward(y, train, chain);
            }
            return y;
        }
    };
}

fn BottleneckA(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &.{
                .{ "conv1", Conv2d(T) },
                .{ "bn1", BatchNorm(T) },
                .{ "conv2", Conv2d(T) },
                .{ "bn2", BatchNorm(T) },
                .{ "conv3", Conv2d(T) },
                .{ "bn3", BatchNorm(T) },
                .{ "conv4", Conv2d(T) },
                .{ "bn4", BatchNorm(T) },
            },
        ),

        const Self = @This();

        pub fn init(
            _: usize,
            mid_channels: usize,
            out_channels: usize,
            stride: [2]usize,
            downsample_fb: bool,
            context: *Context,
            chain: *Chain,
        ) !Self {
            const stride_1x1, const stride_3x3 = if (downsample_fb) .{ [2]usize{ 1, 1 }, stride } else .{ stride, [2]usize{ 1, 1 } };

            var conv1: Conv2d(T) = try .init(
                null,
                mid_channels,
                .{ 1, 1 },
                stride_1x1,
                .{ 0, 0 },
                .{ 1, 1 },
                true,
                .he_normal,
                context,
                chain,
            );
            errdefer conv1.destroy();
            var bn1: BatchNorm(T) = .init(context, chain);
            errdefer bn1.destroy();

            var conv2: Conv2d(T) = try .init(
                null,
                mid_channels,
                .{ 3, 3 },
                stride_3x3,
                .{ 1, 1 },
                .{ 1, 1 },
                true,
                .he_normal,
                context,
                chain,
            );
            errdefer conv2.destroy();
            var bn2: BatchNorm(T) = .init(context, chain);
            errdefer bn2.destroy();

            var conv3: Conv2d(T) = try .init(
                null,
                out_channels,
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 0, 0 },
                .{ 1, 1 },
                true,
                .he_normal,
                context,
                chain,
            );
            errdefer conv3.destroy();
            var bn3: BatchNorm(T) = .init(context, chain);
            errdefer bn3.destroy();

            var conv4: Conv2d(T) = try .init(
                null,
                out_channels,
                .{ 1, 1 },
                stride,
                .{ 0, 0 },
                .{ 1, 1 },
                true,
                .he_normal,
                context,
                chain,
            );
            errdefer conv4.destroy();
            var bn4: BatchNorm(T) = .init(context, chain);
            errdefer bn4.destroy();

            return .{
                .fields = .{
                    .conv1 = conv1,
                    .bn1 = bn1,
                    .conv2 = conv2,
                    .bn2 = bn2,
                    .conv3 = conv3,
                    .bn3 = bn3,
                    .conv4 = conv4,
                    .bn4 = bn4,
                },
            };
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            var h1 = try self.fields.conv1.forward(x, chain);
            h1 = try self.fields.bn1.forward(h1, train, chain);
            h1 = try reluEx(T, h1, chain);
            h1 = try self.fields.conv2.forward(h1, chain);
            h1 = try self.fields.bn2.forward(h1, train, chain);
            h1 = try reluEx(T, h1, chain);
            h1 = try self.fields.conv3.forward(h1, chain);
            h1 = try self.fields.bn3.forward(h1, train, chain);
            const h2 = try self.fields.bn4.forward(try self.fields.conv4.forward(x, chain), train, chain);
            //std.debug.print("{any} {any}", .{ h1.getShape(), h2.getShape() });
            return try reluEx(T, try addEx(T, h1, h2, chain), chain);
        }
    };
}

fn BottleneckB(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{},
            &.{
                .{ "conv1", Conv2d(T) },
                .{ "bn1", BatchNorm(T) },
                .{ "conv2", Conv2d(T) },
                .{ "bn2", BatchNorm(T) },
                .{ "conv3", Conv2d(T) },
                .{ "bn3", BatchNorm(T) },
            },
        ),

        const Self = @This();

        pub fn init(
            out_channels: usize,
            mid_channels: usize,
            context: *Context,
            chain: *Chain,
        ) !Self {
            var conv1: Conv2d(T) = try .init(
                null,
                mid_channels,
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 0, 0 },
                .{ 1, 1 },
                true,
                .he_normal,
                context,
                chain,
            );
            errdefer conv1.destroy();
            var bn1: BatchNorm(T) = .init(context, chain);
            errdefer bn1.destroy();

            var conv2: Conv2d(T) = try .init(
                null,
                mid_channels,
                .{ 3, 3 },
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 1, 1 },
                true,
                .he_normal,
                context,
                chain,
            );
            errdefer conv2.destroy();
            var bn2: BatchNorm(T) = .init(context, chain);
            errdefer bn2.destroy();

            var conv3: Conv2d(T) = try .init(
                null,
                out_channels,
                .{ 1, 1 },
                .{ 1, 1 },
                .{ 0, 0 },
                .{ 1, 1 },
                true,
                .he_normal,
                context,
                chain,
            );
            errdefer conv3.destroy();
            var bn3: BatchNorm(T) = .init(context, chain);
            errdefer bn3.destroy();

            return .{
                .fields = .{
                    .conv1 = conv1,
                    .bn1 = bn1,
                    .conv2 = conv2,
                    .bn2 = bn2,
                    .conv3 = conv3,
                    .bn3 = bn3,
                },
            };
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            var h = try self.fields.conv1.forward(x, chain);
            h = try self.fields.bn1.forward(h, train, chain);
            h = try reluEx(T, h, chain);
            h = try self.fields.conv2.forward(h, chain);
            h = try self.fields.bn2.forward(h, train, chain);
            h = try reluEx(T, h, chain);
            h = try self.fields.conv3.forward(h, chain);
            h = try self.fields.bn3.forward(h, train, chain);
            return try reluEx(T, try addEx(T, h, x, chain), chain);
        }
    };
}

pub fn Embedding(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);
        fields: LayerFieldsFactory(
            &.{"w"}, // Only one parameter: the embedding weight matrix
            &.{},
        ),
        num_embeddings: usize,
        embedding_dim: usize,
        winit: WInit,
        context: *Context,
        chain: *Chain,

        const Self = @This();

        // Weight initialization options, similar to Linear and Conv2d
        pub const WInit = enum {
            normal, // N(0, 1)
            xavier, // Xavier uniform
            he_normal, // He normal
            he_uniform, // He uniform
        };

        pub fn init(
            num_embeddings: usize,
            embedding_dim: usize,
            winit: WInit,
            context: *Context,
            chain: *Chain,
        ) !Self {
            // Create the weight tensor [num_embeddings, embedding_dim]
            var w_tensor: GPUTensor(T) = try .initAsync(&.{ num_embeddings, embedding_dim }, context.stream);
            errdefer w_tensor.deinitAsync(context.stream);

            // Initialize the weight tensor based on winit
            switch (winit) {
                .normal => try w_tensor.fillNormalDistribution(0.0, 1.0, context.cuda_context, context.stream),
                .xavier => try w_tensor.fillXavierUniform(context.cuda_context, context.stream),
                .he_normal => try w_tensor.fillHeNormal(context.cuda_context, context.stream),
                .he_uniform => try w_tensor.fillHeUniform(context.cuda_context, context.stream),
            }

            // Register the weight as a learnable variable
            const w = try chain.createVariable(T, w_tensor.move(), "w");

            return .{
                .fields = .{
                    .w = w,
                },
                .num_embeddings = num_embeddings,
                .embedding_dim = embedding_dim,
                .winit = winit,
                .context = context,
                .chain = chain,
            };
        }

        pub fn forward(self: *Self, indices: *TaggedVar, chain: *Chain) !*TaggedVar {
            return try embedEx(T, self.fields.w.?, indices, chain);
        }
    };
}

pub fn CausalSelfAttention(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);

        fields: LayerFieldsFactory(
            &.{},
            &.{
                .{ "c_attn", Linear(T) },
                .{ "c_proj", Linear(T) },
                .{ "attn_dropout", Dropout(T) },
                .{ "resid_dropout", Dropout(T) },
            },
        ),
        n_head: usize,
        n_embd: usize,
        // dropout_ratio: T,
        bias: GPUTensor(T),
        context: *Context,

        const Config = struct {
            n_embd: usize,
            n_head: usize,
            block_size: usize,
            nobias: bool,
            dropout_ratio: if (T != tomo.BF16) T else f32,
        };

        const Self = @This();

        pub fn init(
            config: Config,
            context: *Context,
            chain: *Chain,
        ) !Self {
            std.debug.assert(config.n_embd % config.n_head == 0);
            var c_attn: Linear(T) = try .init(config.n_embd, 3 * config.n_embd, config.nobias, .xavier, context, chain);
            errdefer c_attn.destroy();
            var c_proj: Linear(T) = try .init(config.n_embd, config.n_embd, config.nobias, .xavier, context, chain);
            errdefer c_proj.destroy();
            const attn_dropout: Dropout(T) = .init(config.dropout_ratio);
            const resid_dropout: Dropout(T) = .init(config.dropout_ratio);

            var bias: GPUTensor(T) = try .initAsync(&.{ config.block_size, config.block_size }, context.stream);
            defer bias.deinitAsync(context.stream);
            try bias.fill(0.0, context.stream);
            try bias.triu(1.0, context.stream);
            try bias.reshape(&.{ 1, 1, config.block_size, config.block_size });

            return .{
                .fields = .{
                    .c_attn = c_attn,
                    .c_proj = c_proj,
                    .attn_dropout = attn_dropout,
                    .resid_dropout = resid_dropout,
                },
                .n_head = config.n_head,
                .n_embd = config.n_embd,
                .bias = bias.move(),
                .context = context,
            };
        }

        pub fn predestroy(self: *Self) void {
            self.bias.deinitAsync(self.context.stream);
        }

        pub fn forward(self: *Self, x: *TaggedVar, train: bool, chain: *Chain) !*TaggedVar {
            const x_shape = x.getShape();
            const b = x_shape[0];
            const t = x_shape[1];
            const c = x_shape[2];

            if (t > self.bias.base.getShape()[2]) return error.SequenceLengthExceedsBlockSize;

            const qkv = try self.fields.c_attn.forward(x, chain);

            // var q, var k, var v = try splitEx(3, T, qkv, 2, chain);

            const split_size = c; // c = x_shape[2], size of each split along axis 2
            var q = try getItemEx(T, 3, qkv, .{
                .all,
                .all,
                .{ .start = 0, .stop = @intCast(split_size), .step = 1 },
            }, chain);
            var k = try getItemEx(T, 3, qkv, .{
                .all,
                .all,
                .{ .start = @intCast(split_size), .stop = @intCast(2 * split_size), .step = 1 },
            }, chain);
            var v = try getItemEx(T, 3, qkv, .{
                .all,
                .all,
                .{ .start = @intCast(2 * split_size), .stop = @intCast(3 * split_size), .step = 1 },
            }, chain);

            q = try reshapeEx(T, q, &.{ b, t, self.n_head, c / self.n_head }, chain);
            k = try reshapeEx(T, k, &.{ b, t, self.n_head, c / self.n_head }, chain);

            v = try reshapeEx(T, v, &.{ b, t, self.n_head, c / self.n_head }, chain);
            q = try transposeExEx(T, q, &.{ 0, 2, 1, 3 }, chain); // (B, nh, T, hs)
            k = try transposeExEx(T, k, &.{ 0, 2, 1, 3 }, chain); // (B, nh, T, hs)
            v = try transposeExEx(T, v, &.{ 0, 2, 1, 3 }, chain); // (B, nh, T, hs)

            const k_transpose = try transposeExEx(T, k, &.{ 0, 1, 3, 2 }, chain);

            var att = try matmulEx(T, q, k_transpose, chain);

            att = try scaleEx(T, att, 1.0 / @sqrt(@as(if (T != BF16) T else f32, @floatFromInt(k_transpose.getShape()[k_transpose.getShape().len - 1]))), chain);

            var mask = try self.bias.getItem(
                self.context.allocator,
                &.{
                    .all,
                    .all,
                    .{
                        .start = 0,
                        .stop = @intCast(t),
                        .step = 1,
                    },
                    .{
                        .start = 0,
                        .stop = @intCast(t),
                        .step = 1,
                    },
                },
                self.context.stream,
            );
            defer mask.deinitAsync(self.context.stream);
            const mask_var = try chain.createVariable(T, mask.move(), null);
            const mask_broad = try broadcastToEx(T, mask_var, att.getShape(), chain);

            // try dbg(BF16, &mask_broad.asUntagged(BF16).data, self.context);
            // std.time.sleep(5 * std.time.ns_per_s);

            att = try maskedFillEx(
                T,
                att,
                .{
                    .mask = mask_broad,
                    //.val = -std.math.inf(T),
                    .val = -1e9,
                },
                chain,
            );

            // try dbg(BF16, &att.asUntagged(BF16).data, self.context);
            // std.time.sleep(5 * std.time.ns_per_s);

            att = try softmaxEx(T, att, &.{3}, chain);

            att = try self.fields.attn_dropout.forward(att, train, chain);
            var y = try matmulEx(T, att, v, chain);
            y = try transposeExEx(T, y, &.{ 0, 2, 1, 3 }, chain);
            y = try reshapeEx(T, y, &.{ b, t, c }, chain);
            y = try self.fields.c_proj.forward(y, chain);
            y = try self.fields.resid_dropout.forward(y, train, chain);

            return y;
        }
    };
}

pub fn Gelu(comptime T: type) type {
    return struct {
        pub usingnamespace LayerDecorator(Self);

        fields: LayerFieldsFactory(
            &.{},
            &.{},
        ),
        const Self = @This();
        pub const init: Self = .{ .fields = .{} };

        pub fn forward(_: *Self, x: *TaggedVar, chain: *Chain) !*TaggedVar {
            return try geluEx(T, x, chain);
        }
    };
}
