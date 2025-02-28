const std = @import("std");
const tomo = @import("tomo");

const GPUTensor = tomo.tensor.GPUTensor;

const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const LevelStack = @import("stack.zig").LevelStack;
const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;

const function = @import("function.zig");

pub const Context = struct {
    cuda_context: *const CudaContext,
    stream: *const Stream,
    allocator: std.mem.Allocator,
    compact_mode: bool = false,
    function_level_stack: LevelStack(Function),
    tagged_var_level_stack: LevelStack(TaggedVar),

    const func_max_out = 3;
    pub fn init(
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
        compact_mode: bool,
    ) !Context {
        var self: Context = .{
            .allocator = allocator,
            .cuda_context = cuda_context,
            .stream = stream,
            .compact_mode = compact_mode,
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

    pub fn destroyTopLevelFunctions(self: *Context) void {
        const funcs = self.getTopLevelFunctions();
        for (funcs.items) |*func| {
            func.destroy();
        }
        self.function_level_stack.destroyTopLevel();
    }

    pub fn gc(self: *Context) void {
        for (0..self.tagged_var_level_stack.levels.len) |i| {
            self.gcLevel(i);
        }
    }

    pub fn gcLevel(self: *Context, level: usize) void {
        for (self.tagged_var_level_stack.levels.items[level].items) |*variable| {
            if (variable.getRefCount() == 0) {
                variable.deinit();
            }
        }
    }

    pub fn makeNewLevel(self: *Context) !void {
        try self.function_level_stack.makeNewLevel();
        try self.tagged_var_level_stack.makeNewLevel();
    }

    pub fn getCurrentLevel(self: *Context) usize {
        return self.tagged_var_level_stack.levels.items.len - 1;
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

    pub fn getTopLevelVariables(self: *Context) *std.ArrayList(TaggedVar) {
        return self.tagged_var_level_stack.getTopLevel();
    }

    pub fn getLevelVariables(self: *Context, level: usize) *std.ArrayList(TaggedVar) {
        return self.tagged_var_level_stack.getLevel(level);
    }

    pub fn getLevelVariablesConst(self: *const Context, level: usize) *const std.ArrayList(TaggedVar) {
        return self.tagged_var_level_stack.getLevelConst(level);
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

    pub fn pushFunctionAtCurrentLevelTop(self: *Context, func: Function) !void {
        return self.function_level_stack.pushAtTopLevel(func);
    }

    pub fn getTopLevelFunctions(self: *Context) *std.ArrayList(Function) {
        return self.function_level_stack.getTopLevel();
    }

    pub fn createVariable(
        self: *Context,
        comptime T: type,
        data: GPUTensor(T),
        name: ?[]const u8,
    ) !VarKey {
        const variable: Variable(T) = .{
            .data = data,
            .name = name,
            .self_key = undefined,
            .refcount = 1,
        };

        const tagged = TaggedVar.init(T, variable);

        try self.pushTaggedVarAtCurrentLevelTop(tagged);

        const self_key: VarKey = .{
            .level = self.getCurrentLevel(),
            .index = self.getCurrentLevelTopTaggedVarIndex(),
            .context = self,
        };

        self.getCurrentLevelTopTaggedVar().setSelfkey(self_key);

        return self_key;
    }

    pub fn registerFunction(self: *Context, func: Function) !FuncKey {
        try self.pushFunctionAtCurrentLevelTop(func);

        return .{
            .level = self.getCurrentLevel(),
            .index = self.getCurrentLevelTopFunctionIndex(),
            .context = self,
        };
    }

    pub fn refVariable(self: *Context, key: VarKey) *TaggedVar {
        const variable = &self.tagged_var_level_stack.levels.items[key.level].items[key.index];
        return variable;
    }

    pub fn refVariableConst(self: *Context, key: VarKey) *const TaggedVar {
        const variable = &self.tagged_var_level_stack.levels.items[key.level].items[key.index];
        return variable;
    }

    pub fn acquireVariable(self: *Context, key: VarKey) *TaggedVar {
        const variable = self.refVariable(key);
        variable.acquire();
        return variable;
    }

    pub fn acquireVariableConst(self: *Context, key: VarKey) *const TaggedVar {
        const variable = self.refVariable(key);
        variable.acquire();
        return variable;
    }

    pub fn releaseVariable(self: *Context, key: VarKey) void {
        const variable = self.refVariable(key);
        variable.release();
    }

    pub fn refFunction(self: *Context, key: FuncKey) *Function {
        return &self.function_level_stack.levels.items[key.level].items[key.index];
    }

    pub fn refFunctionConst(self: *const Context, key: FuncKey) *const Function {
        return &self.function_level_stack.levels.items[key.level].items[key.index];
    }

    pub fn backward(
        self: *Context,
        comptime T: type,
        key: VarKey,
        grad_catch_vars: []const VarKey,
    ) !void {
        if (key.level < self.getCurrentLevel()) {
            return;
        }

        try self.makeNewLevel();

        const variable = self.acquireVariable(key).asUntagged(T);
        const creator = variable.refCreator() orelse return error.NoCreator;

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

        while (function_queue.removeOrNull()) |func| {
            try func.backward();
            try func.enqueue(&function_queue, &seen_set);
        }

        for (grad_catch_vars) |grad_catch_var| {
            self.refVariable(grad_catch_var).acquireGrad();
        }

        if (self.compact_mode) {
            self.destroyTopLevelFunctions(); // f'
            self.destroyTopLevelFunctions(); // f

            const prime_level = self.getCurrentLevel();
            const func_level = prime_level - 1;

            self.gcLevel(prime_level);
            self.gcLevel(func_level);
        }
    }

    pub fn countAliveVariableAtLevel(self: *const Context, level: usize) usize {
        const var_level = self.getLevelVariablesConst(level);
        var count: usize = 0;
        for (var_level.items) |*variable| {
            // if (variable.getRefCount() != 0) {
            //     count += 1;
            // }
            if (!variable.isDataNull()) {
                count += 1;
            }
        }

        return count;
    }
};

pub const VarKey = struct {
    level: usize,
    index: usize,
    context: *Context,

    pub fn ref(self: *VarKey) *TaggedVar {
        return self.context.refVariable(self.*);
    }

    pub fn refConst(self: *const VarKey) *const TaggedVar {
        return self.context.refVariableConst(self.*);
    }

    pub fn dataToHost(self: *const VarKey, comptime T: type) !tomo.tensor.CPUTensor(T) {
        return try self.refConst().asUntaggedConst(T).data.toHost(self.context.allocator, self.context.stream);
    }

    pub fn gradToHost(self: *const VarKey, comptime T: type) !tomo.tensor.CPUTensor(T) {
        return try self.refConst().refGradConst().?.asUntaggedConst(T).data.toHost(self.context.allocator, self.context.stream);
    }

    pub fn release(self: VarKey) void {
        self.context.releaseVariable(self);
    }
};

// pub fn VarKeyTyped(comptime T: type) type {
//     return struct {
//         varkey: VarKey,

//         const Self = @This();

//         pub fn neg(self: Self) !Self {
//             return Self{ .varkey = try function.neg(T, self.varkey) };
//         }

//         pub fn square(self: Self) !Self {
//             return Self{ .varkey = try function.square(T, self.varkey) };
//         }

//         pub fn exp(self: Self) !Self {
//             return Self{ .varkey = try function.exp(T, self.varkey) };
//         }

//         //

//         pub fn shift(self: Self, scalar: T) !Self {
//             return Self{ .varkey = try function.shift(T, self.varkey, scalar) };
//         }

//         pub fn scale(self: Self, scalar: T) !Self {
//             return Self{ .varkey = try function.scale(T, self.varkey, scalar) };
//         }

//         pub fn powf(self: Self, scalar: T) !Self {
//             return Self{ .varkey = try function.powf(T, self.varkey, scalar) };
//         }

//         pub fn pow(self: Self, scalar: i32) !Self {
//             return Self{ .varkey = try function.pow(T, self.varkey, scalar) };
//         }

//         //
//         pub fn scaleShift(self: Self, scal: T, shif: T) !Self {
//             return Self{ .varkey = try function.scaleShift(T, self.varkey, scal, shif) };
//         }

//         //

//         pub fn add(self: Self, other: Self) !Self {
//             return Self{ .varkey = try function.add(T, self.varkey, other) };
//         }

//         pub fn sub(self: Self, other: Self) !Self {
//             return Self{ .varkey = try function.sub(T, self.varkey, other) };
//         }

//         pub fn mul(self: Self, other: Self) !Self {
//             return Self{ .varkey = try function.mul(T, self.varkey, other) };
//         }

//         pub fn div(self: Self, other: Self) !Self {
//             return Self{ .varkey = try function.div(T, self.varkey, other) };
//         }
//     };
// }

pub const FuncKey = struct {
    level: usize,
    index: usize,
    context: *Context,
};
