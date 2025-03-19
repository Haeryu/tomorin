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
            .alignment = @alignOf(*TaggedVar),
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

            const src = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
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
        base_chain: *Chain,

        const Self = @This();

        pub fn init(
            in_size: ?usize,
            out_size: usize,
            no_bias: bool,
            context: *Context,
            base_chain: *Chain,
        ) !Self {
            const b = if (no_bias) null else blk: {
                var b_tensor: GPUTensor(T) = try .initAsync(&.{ 1, out_size }, context.stream);
                errdefer b_tensor.deinitAsync(context.stream);
                try b_tensor.fill(0.0, context.stream);
                break :blk try base_chain.createVariable(T, b_tensor.move(), "b");
            };

            return .{
                .fields = .{
                    .w = null,
                    .b = b,
                },
                .in_size = in_size,
                .out_size = out_size,
                .context = context,
                .base_chain = base_chain,
            };
        }

        fn initW(
            self: *Self,
        ) !*TaggedVar {
            var w_tensor: GPUTensor(T) = try .initAsync(&.{ self.in_size.?, self.out_size }, self.context.stream);
            errdefer w_tensor.deinitAsync(self.context.stream);
            try w_tensor.fillNormalDistribution(0.0, 1.0, self.context.cuda_context, self.context.stream);
            try w_tensor.scale(1.0 / @as(T, @floatFromInt(self.in_size.?)), self.context.stream);
            const w = try self.base_chain.createVariable(T, w_tensor.move(), "w");
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
                    context,
                    chain,
                );
            }

            return self;
        }

        pub fn forward(
            self: *Self,
            x: *TaggedVar,
            comptime activation: fn (comptime T: type, x: *TaggedVar, chain: *Chain) anyerror!*TaggedVar,
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
