const std = @import("std");
const builtin = @import("builtin");
const TaggedVar = @import("variable.zig").TaggedVar;
const tomo = @import("tomo");
const Context = @import("context.zig").Context;
const Chain = @import("chain.zig").Chain;
const GPUTensor = tomo.tensor.GPUTensor;

pub fn LayerFieldsFactory(
    comptime param_names: []const [:0]const u8,
    comptime layer_names_types: []const std.meta.Tuple(&.{ [:0]const u8, type }),
) type {
    var fields: [param_names.len + layer_names_types.len]std.builtin.Type.StructField = undefined;

    var i: comptime_int = 0;
    for (param_names) |param_name| {
        fields[i] = .{
            .name = param_name,
            .type = *TaggedVar,
            .default_value_ptr = null,
            .is_comptime = false,
            .alignment = @alignOf(*TaggedVar),
        };
        i += 1;
    }
    for (layer_names_types) |layer_name_type| {
        fields[i] = .{
            .name = layer_name_type[0],
            .type = layer_name_type[1],
            .default_value_ptr = null,
            .is_comptime = false,
            .alignment = @alignOf(layer_name_type[1]),
        };
        i += 1;
    }

    const layer: std.builtin.Type = .{ .@"struct" = .{
        .layout = .auto,
        .backing_integer = null,
        .fields = &fields,
        .decls = &.{},
        .is_tuple = false,
    } };

    return @Type(layer);
}

pub fn LayerDecorator(comptime Self: type) type {
    return struct {
        pub fn calcParamNum() comptime_int {
            const fields_info = @typeInfo(@FieldType(Self, "fields")).@"struct";

            var param_num: comptime_int = 0;
            for (fields_info.fields) |field| {
                if (field.type == *TaggedVar) {
                    param_num += 1;
                } else {
                    param_num = @TypeOf(@field(Self, field.name)).calcParamNum();
                }
            }
            return param_num;
        }

        pub fn getParams(self: *Self) [@This().calcParamNum()]*TaggedVar {
            const fields_info = @typeInfo(@FieldType(Self, "fields")).@"struct";

            var params: [@This().calcParamNum()]*TaggedVar = undefined;

            var i: comptime_int = 0;
            for (fields_info.fields) |field| {
                if (field.type == *TaggedVar) {
                    params[i] = @field(self, field.name);
                    i += 1;
                } else {
                    var field_params = @field(self, field.name).getParams();
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

            inline for (fields_info.fields) |field| {
                @field(self.fields, field.name).destroy();
            }
        }
    };
}

pub const Linear = struct {
    pub usingnamespace LayerDecorator(Linear);
    fields: LayerFieldsFactory(
        &.{
            "w",
            "b",
        },
        &.{},
    ),

    pub fn init(
        comptime T: type,
        in_size: usize,
        out_size: usize,
        context: *Context,
        chain: *Chain,
    ) !Linear {
        var w_tensor: GPUTensor(T) = try .initAsync(&.{ in_size, out_size }, context.stream);
        errdefer w_tensor.deinitAsync(context.stream);

        var b_tensor: GPUTensor(T) = try .initAsync(&.{ 1, out_size }, context.stream);
        errdefer b_tensor.deinitAsync(context.stream);

        const w = try chain.createVariable(T, w_tensor.move(), "w");
        const b = try chain.createVariable(T, b_tensor.move(), "b");

        return .{
            .fields = .{
                .w = w,
                .b = b,
            },
        };
    }
};
