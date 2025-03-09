const std = @import("std");
const builtin = @import("builtin");
const TaggedVar = @import("variable.zig").TaggedVar;
const tomo = @import("tomo");
const Context = @import("context.zig").Context;
const Chain = @import("chain.zig").Chain;
const GPUTensor = tomo.tensor.GPUTensor;

pub const LayerBase = struct {
    params: std.StringArrayHashMapUnmanaged(*TaggedVar),
    layers: std.StringArrayHashMapUnmanaged(Layer),
};

pub fn ConcatIterator(comptime T: type, I1: type, I2: type) type {
    return struct {
        iter1: I1,
        iter2: I2,

        const Self = @This();

        pub fn next(self: *Self) ?T {
            if (self.iter1.next()) |value| {
                return value;
            }
            return self.iter2.next();
        }
    };
}

pub fn ConcatIterators(comptime T: type, IteratorType: type) type {
    return struct {
        iterators: []IteratorType,
        current_index: usize = 0,

        allocator: std.mem.Allocator,

        const Self = @This();

        pub fn next(self: *Self) ?T {
            while (self.current_index < self.iterators.len) {
                const maybe = self.iterators[self.current_index].next();
                if (maybe) |value| {
                    return value;
                } else {
                    self.current_index += 1;
                }
            }
            return null;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.iterators);
        }
    };
}

pub const Layer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    const VTable = struct {
        forward: *const fn (ctx: *anyopaque, xs: []*TaggedVar, ys: []*TaggedVar) anyerror!void,
        destroy: *const fn (ctx: *anyopaque) void,
        get_base: *const fn (ctx: *anyopaque) *LayerBase,
    };

    pub fn forward(self: *Layer, xs: []*TaggedVar, ys: []?*TaggedVar) !void {
        try self.vtable.forward(self.ptr, xs, ys);
    }

    pub fn getBase(self: *Layer) *LayerBase {
        return self.vtable.get_base(self.ptr);
    }

    pub fn paramsIter(self: *Layer, allocator: std.mem.Allocator) !ParamsIter {
        var params_iters: std.ArrayList(std.StringArrayHashMapUnmanaged(*TaggedVar).Iterator) = .init(allocator);
        defer params_iters.deinit();

        try params_iters.append(self.getBase().params.iterator());

        const base = self.getBase();
        var layer_iter = base.layers.iterator();
        while (layer_iter.next()) |layer| {
            var iter = try layer.value_ptr.*.paramsIter(allocator);
            defer iter.deinit();

            try params_iters.appendSlice(iter.iterators);
        }

        return .{
            .iters = try params_iters.toOwnedSlice(),
        };
    }

    pub fn clearGrads(self: *Layer) void {
        var iter = self.paramsIter();
        while (iter.next()) |i| {
            i.value_ptr.*.setGrad(null);
        }
    }

    pub fn destroy(self: *Layer) void {
        const base = self.getBase();
        var param_iter = base.params.iterator();
        while (param_iter.next()) |param| {
            param.value_ptr.*.destroy();
        }

        var layer_iter = base.layers.iterator();
        while (layer_iter.next()) |layer| {
            layer.value_ptr.*.destroy();
        }

        self.vtable.destroy(self.ptr);
    }
};

pub fn LayerBaseFactory(
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
            .alignment = @alignOf(Layer),
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

const ParamsIter = ConcatIterators(*TaggedVar, std.StringArrayHashMapUnmanaged(*TaggedVar).Iterator);

pub fn LayerDecorator(comptime Self: type) type {
    return struct {
        pub fn calcParamNum() comptime_int {
            const base_info = @typeInfo(@FieldType(Self, "base")).@"struct";
            var param_num: comptime_int = 0;
            for (base_info.fields) |field| {
                if (field.type == *TaggedVar) {
                    param_num += 1;
                } else {
                    param_num = @TypeOf(@field(Self, field.name)).calcParamNum();
                }
            }
            return param_num;
        }

        pub fn getParams(self: *Self) [@This().calcParamNum()]*TaggedVar {
            const base_info = @typeInfo(@FieldType(Self, "base")).@"struct";

            var params: [@This().calcParamNum()]*TaggedVar = undefined;

            var i: comptime_int = 0;
            for (base_info.fields) |field| {
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
            const base_info = @typeInfo(@FieldType(Self, "base")).@"struct";

            inline for (base_info.fields) |field| {
                @field(self.base, field.name).destroy();
            }
        }
    };
}

pub const Linear = struct {
    pub usingnamespace LayerDecorator(Linear);
    base: LayerBaseFactory(
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
            .base = .{
                .w = w,
                .b = b,
            },
        };
    }
};
