const std = @import("std");
const tomo = @import("tomo");

const GPUTensor = tomo.tensor.GPUTensor;

const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const LevelStack = @import("stack.zig").LevelStack;
const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;
const Pool = @import("nina").container.pool.Pool;

const function = @import("function.zig");

pub const ContextOptions = struct {
    aggressive_release: bool = false,
    verbose_dot: bool = false,
    init_var_capacity: usize = 0,
    init_func_capacity: usize = 0,
};

pub const Context = struct {
    cuda_context: *const CudaContext,
    stream: *const Stream,
    allocator: std.mem.Allocator,

    functions: std.ArrayList(Function),
    tagged_vars: Pool(TaggedVar),
    options: ContextOptions,

    const func_max_out = 3;

    pub fn init(
        allocator: std.mem.Allocator,
        cuda_context: *const CudaContext,
        stream: *const Stream,
        options: ContextOptions,
    ) !Context {
        return .{
            .allocator = allocator,
            .cuda_context = cuda_context,
            .stream = stream,
            .functions = try std.ArrayList(Function).initCapacity(allocator, options.init_func_capacity),
            .tagged_vars = try Pool(TaggedVar).initCapacity(allocator, options.init_var_capacity),
            .options = options,
        };
    }

    pub fn deinit(self: *Context) void {
        self.destroyFunctions();
        self.destroyVariables();

        self.functions.deinit();
        self.tagged_vars.deinit();
    }

    pub fn destroyFunctions(self: *Context) void {
        for (self.functions.items) |*f| {
            f.destroy();
        }
    }

    pub fn destroyVariables(self: *Context) void {
        var var_iter = self.tagged_vars.getIter() catch unreachable;
        defer var_iter.deinit();

        while (var_iter.nextPtr()) |v| {
            v.deinit();
        }
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

        const index = try self.tagged_vars.pushItem(tagged);

        const self_key: VarKey = .{
            .index = index,
            .context = self,
        };

        self.tagged_vars.getItem(index).setSelfkey(self_key);

        return self_key;
    }

    pub fn registerFunction(self: *Context, func: Function) !FuncKey {
        try self.functions.append(func);

        return .{
            .index = self.functions.items.len - 1,
            .context = self,
        };
    }

    pub fn refVariable(self: *Context, key: VarKey) *TaggedVar {
        return self.tagged_vars.getItem(key.index);
    }

    pub fn refVariableConst(self: *const Context, key: VarKey) *const TaggedVar {
        return self.tagged_vars.getItemConst(key.index);
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
        if (self.options.aggressive_release) {
            if (variable.getRefCount() == 0) {
                variable.deinit();
                self.tagged_vars.destroyItem(key.index);
            }
        }
    }

    pub fn refFunction(self: *Context, key: FuncKey) *Function {
        return &self.functions.items[key.index];
    }

    pub fn refFunctionConst(self: *const Context, key: FuncKey) *const Function {
        return &self.functions.items[key.index];
    }

    pub fn backward(
        self: *Context,
        comptime T: type,
        key: VarKey,
        grad_catch_vars: []const VarKey,
    ) !void {
        const variable = self.acquireVariable(key).asUntagged(T);
        const creator = variable.creator orelse return error.NoCreator;

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

        if (self.options.aggressive_release) {
            self.destroyFunctions();
            self.functions.clearAndFree(); // retain??
        }
    }

    pub fn countAliveVariable(self: *const Context) !usize {
        var var_iter = try self.tagged_vars.getIter();
        defer var_iter.deinit();

        var count: usize = 0;

        while (var_iter.nextPtr()) |variable| {
            if (!variable.isDataNull()) {
                count += 1;
            }
        }

        return count;
    }

    pub fn createSnapShot(self: *const Context) SnapShot {
        std.debug.assert(!self.options.aggressive_release);

        return .{
            .context = self,
        };
    }
};

pub const VarKey = struct {
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

    pub fn acquire(self: VarKey) void {
        var mut_self = self;
        mut_self.ref().acquire();
    }

    pub fn setGrad(self: VarKey, grad: VarKey) void {
        var mut_self = self;
        mut_self.ref().setGrad(grad);
    }
};

pub const FuncKey = struct {
    index: usize,
    context: *Context,

    pub fn getGeneration(self: FuncKey) usize {
        return self.context.refFunction(self).getGeneration();
    }

    pub fn backward(self: FuncKey) !void {
        try self.context.refFunction(self).backward();
    }

    pub fn enqueue(self: FuncKey, function_queue: *Function.Queue, seen_set: *Function.SeenSet) !void {
        try self.context.refFunction(self).enqueue(function_queue, seen_set);
    }

    pub fn ref(self: FuncKey) *Function {
        return self.context.refFunction(self);
    }

    pub fn refConst(self: FuncKey) *const Function {
        return self.context.refFunctionConst(self);
    }
};

pub const SnapShot = struct {
    start_var_idx: ?usize = null,
    start_func_idx: ?usize = null,
    end_var_idx: ?usize = null,
    end_func_idx: ?usize = null,
    context: *const Context,

    pub fn shotStart(self: *SnapShot) void {
        self.start_var_idx = self.context.tagged_vars.datas.items.len;
        self.start_func_idx = self.context.functions.items.len;
    }

    pub fn shotEnd(self: *SnapShot) void {
        self.end_var_idx = self.context.tagged_vars.datas.items.len;
        self.end_func_idx = self.context.functions.items.len;
    }

    fn getVars(self: *SnapShot) []TaggedVar {
        return self.context.tagged_vars.datas.items[self.start_var_idx.?..self.end_var_idx.?];
    }

    fn getFuncs(self: *SnapShot) []TaggedVar {
        return self.context.functions.items[self.start_func_idx.?..self.end_func_idx.?];
    }

    pub fn writeDot(self: *const SnapShot, writer: anytype) !void {
        std.debug.assert(!self.context.options.aggressive_release);

        try writer.writeAll("digraph g {\n");

        for (self.getVars()) |variable| {
            const dot_var = try variable.getDotAlloc();
            defer self.context.allocator.free(dot_var);

            try writer.writeAll(dot_var);
        }

        for (self.getFuncs()) |func| {
            const dot_func = try func.getDotAlloc();
            defer self.context.allocator.free(dot_func);

            try writer.writeAll(dot_func);
        }

        try writer.writeAll("}\n");
    }

    pub fn saveDot(self: *const SnapShot, filename: []const u8) !void {
        const file = try std.fs.cwd().createFile(filename, .{});
        defer file.close();

        try self.writeDot(file.writer());
    }

    // TODO: snapshot -> add destroy var, func function?
};
