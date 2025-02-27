const std = @import("std");
const tomo = @import("tomo");

const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const LevelStack = @import("stack.zig").LevelStack;
const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;

pub const Context = struct {
    cuda_context: *const CudaContext,
    stream: *const Stream,
    enable_backprop_graph: bool,
    allocator: std.mem.Allocator,
    function_level_stack: LevelStack(Function),
    tagged_var_level_stack: LevelStack(TaggedVar),

    pub fn init(
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
        enable_backprop_graph: bool,
    ) Context {
        return .{
            .allocator = allocator,
            .cuda_context = cuda_context,
            .stream = stream,
            .enable_backprop_graph = enable_backprop_graph,
            .function_level_stack = LevelStack(Function).init(allocator),
            .tagged_var_level_stack = LevelStack(TaggedVar).init(allocator),
        };
    }

    pub fn deinit(self: *Context) void {
        for (self.function_level_stack.levels.items) |level| {
            for (level.items) |*func| {
                func.destroy();
            }
        }
        for (self.tagged_var_level_stack.levels.items) |level| {
            for (level.items) |*variable| {
                variable.deinit();
            }
        }
        self.function_level_stack.deinit();
        self.tagged_var_level_stack.deinit();
    }

    pub fn makeNewLevel(self: *Context) !void {
        try self.function_level_stack.makeNewLevel();
        try self.tagged_var_level_stack.makeNewLevel();
    }

    pub fn getCurrentLevel(self: *Context) usize {
        return self.function_level_stack.levels.items.len - 1;
    }

    pub fn getCurrentLevelTopTaggedVarIndex(self: *Context) usize {
        return self.tagged_var_level_stack.getTopLevelTopItemIndex();
    }

    pub fn getCurrentLevelTopTaggedVar(self: *Context) *TaggedVar {
        return self.tagged_var_level_stack.getTopLevelTopItem();
    }

    pub fn getCurrentLevelTopTaggedVarConst(self: *const Context) *const TaggedVar {
        return self.tagged_var_level_stack.getTopLevelTopItemConst();
    }

    pub fn pushTaggedVarAtCurrentLevelTop(self: *Context, tagged_var: TaggedVar) !void {
        return self.tagged_var_level_stack.pushAtTopLevel(tagged_var);
    }

    pub fn getCurrentLevelTopFunctionIndex(self: *Context) usize {
        return self.function_level_stack.getTopLevelTopItemIndex();
    }

    pub fn getCurrentLevelTopFunction(self: *Context) *Function {
        return self.function_level_stack.getTopLevelTopItem();
    }

    pub fn getCurrentLevelTopFunctionConst(self: *const Context) *const Function {
        return self.function_level_stack.getTopLevelTopItemConst();
    }

    pub fn pushFunctionAtCurrentLevelTop(self: *Context, function: Function) !void {
        return self.function_level_stack.pushAtTopLevel(function);
    }
};
