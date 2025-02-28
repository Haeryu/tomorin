const std = @import("std");
const tomo = @import("tomo");

const GPUTensor = tomo.tensor.GPUTensor;

const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const LevelStack = @import("stack.zig").LevelStack;
const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;

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
    ) !Context {
        var self: Context = .{
            .allocator = allocator,
            .cuda_context = cuda_context,
            .stream = stream,
            .enable_backprop_graph = enable_backprop_graph,
            .function_level_stack = LevelStack(Function).init(allocator),
            .tagged_var_level_stack = LevelStack(TaggedVar).init(allocator),
        };

        try self.makeNewLevel();

        return self;
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

    pub fn createVariable(
        self: *Context,
        comptime T: type,
        data: GPUTensor(T),
        name: ?[]const u8,
    ) !VarKey {
        var variable: Variable(T) = .{
            .data = data,
            .name = name,
            .context = self,
            .self_key = undefined,
        };

        const tagged = TaggedVar.init(T, variable);

        try self.pushTaggedVarAtCurrentLevelTop(tagged);

        variable.self_key = .{
            .level = self.getCurrentLevel(),
            .index = self.getCurrentLevelTopTaggedVarIndex(),
        };

        return variable.self_key;
    }

    pub fn registerFunction(self: *Context, function: Function) !FuncKey {
        try self.pushFunctionAtCurrentLevelTop(function);

        return .{
            .level = self.getCurrentLevel(),
            .index = self.getCurrentLevelTopFunctionIndex(),
        };
    }

    pub fn getVariable(self: *Context, key: VarKey) *TaggedVar {
        return &self.tagged_var_level_stack.levels.items[key.level].items[key.index];
    }

    pub fn getVariableConst(self: *Context, key: VarKey) *const TaggedVar {
        return &self.tagged_var_level_stack.levels.items[key.level].items[key.index];
    }

    pub fn getFunction(self: *Context, key: FuncKey) *Function {
        return &self.function_level_stack.levels.items[key.level].items[key.index];
    }

    pub fn getFunctionConst(self: *Context, key: FuncKey) *const Function {
        return &self.function_level_stack.levels.items[key.level].items[key.index];
    }

    pub fn backward(self: *Context, comptime T: type, key: VarKey) !void {
        if (key.level < self.getCurrentLevel()) {
            return;
        }

        try self.makeNewLevel();

        const variable = self.getVariable(key).asUntagged(T);
        const creator = variable.getCreator() orelse return error.NoCreator;

        var ones = try tomo.tensor.GPUTensor(T).initAsync(variable.data.base.getShape(), self.stream);
        errdefer ones.deinitAsync(self.stream);
        try ones.fill(1.0, self.stream);

        const initial_grad = try self.createVariable(T, ones, null);
        variable.grad = initial_grad;

        var function_queue = Function.Queue.init(self.allocator, {});
        defer function_queue.deinit();

        var seen_set = Function.SeenSet.init(self.allocator);
        defer seen_set.deinit();

        try function_queue.add(creator);
        try seen_set.put(creator, {});

        while (function_queue.removeOrNull()) |function| {
            try function.backward();
            try function.enqueue(&function_queue, &seen_set);
        }
    }
};

pub const VarKey = struct {
    level: usize,
    index: usize,
};

pub const FuncKey = struct {
    level: usize,
    index: usize,
};
