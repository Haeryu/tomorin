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
    front_only: bool = false,
    enable_multi_backprop: bool = true,
};

pub const Context = struct {
    cuda_context: *const CudaContext,
    stream: *const Stream,
    allocator: std.mem.Allocator,

    functions: std.ArrayList(Function),
    tagged_vars: std.heap.MemoryPool(TaggedVar),
    options: ContextOptions,

    var_chain: ?*TaggedVar,

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
            .tagged_vars = try std.heap.MemoryPool(TaggedVar).initPreheated(allocator, options.init_var_capacity),
            .options = options,
            .var_chain = null,
        };
    }

    pub fn deinit(self: *Context) void {
        self.destroyFunctions();
        self.functions.deinit();
        self.destroyVariables();
        self.tagged_vars.deinit();
    }

    pub fn destroyFunctions(self: *Context) void {
        for (self.functions.items) |*f| {
            f.destroy();
        }
    }

    pub fn destroyVariables(self: *Context) void {
        while (self.var_chain) |head| {
            self.var_chain = head.getNext();
            head.destroy();
        }
    }

    pub fn createVariable(
        self: *Context,
        comptime T: type,
        data: GPUTensor(T),
        name: ?[]const u8,
    ) !*TaggedVar {
        const variable: Variable(T) = .{
            .data = data,
            .name = name,
            .context = self,
            .protected = false,
            .prev = null,
            .next = null,
        };

        const tagged = TaggedVar.init(T, variable);

        const ptr = try self.tagged_vars.create();
        ptr.* = tagged;

        if (self.var_chain) |head| {
            ptr.setNext(head);
            head.setPrev(ptr);
        }

        self.var_chain = ptr;

        return ptr;
    }

    pub fn countVariable(self: *const Context) usize {
        var iter = self.var_chain;
        var count: usize = 0;

        while (iter) |variable| : (iter = variable.getNext()) {
            count += 1;
        }

        return count;
    }

    pub fn resetVarChain(self: *Context) void {
        self.var_chain = null;
    }

    pub fn registerFunction(self: *Context, func: Function) !FuncKey {
        try self.functions.append(func);

        return .{
            .index = self.functions.items.len - 1,
            .context = self,
        };
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
        variable: *TaggedVar,
        grad_catch_vars: []const *TaggedVar,
    ) !void {
        std.debug.assert(!self.options.front_only);

        const variable_untagged = variable.asUntagged(T);
        const creator = variable.getCreator() orelse return error.NoCreator;
        self.resetVarChain();

        var ones = try tomo.tensor.GPUTensor(T).initAsync(variable_untagged.data.base.getShape(), self.stream);
        errdefer ones.deinitAsync(self.stream);
        try ones.fill(1.0, self.stream);

        const initial_grad = try self.createVariable(T, ones, null);
        initial_grad.protect();
        defer initial_grad.unprotect();

        variable_untagged.grad = initial_grad;

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

        if (self.options.aggressive_release) {
            for (grad_catch_vars) |grad_catch_var| {
                grad_catch_var.refGrad().?.protect();
            }
            defer for (grad_catch_vars) |grad_catch_var| {
                grad_catch_var.refGrad().?.unprotect();
            };

            self.destroyFunctions();
            self.functions.clearRetainingCapacity();
        }
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
