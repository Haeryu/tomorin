const std = @import("std");
const tomo = @import("tomo");

const GPUTensor = tomo.tensor.GPUTensor;

const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;
const Pool = @import("nina").container.pool.Pool;
const BF16 = tomo.BF16;

const function = @import("function.zig");

const Snapshot = @import("snapshot.zig").Snapshot;

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

    functions: std.heap.MemoryPool(Function),
    tagged_vars: std.heap.MemoryPool(TaggedVar),
    options: ContextOptions,

    func_chain: ?*Function,
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
            .functions = try std.heap.MemoryPool(Function).initPreheated(allocator, options.init_func_capacity),
            .tagged_vars = try std.heap.MemoryPool(TaggedVar).initPreheated(allocator, options.init_var_capacity),
            .options = options,
            .var_chain = null,
            .func_chain = null,
        };
    }

    pub fn deinit(self: *Context) void {
        self.destroyFunctions();
        self.functions.deinit();
        self.destroyVariables();
        self.tagged_vars.deinit();
    }

    pub fn destroyFunctionsChain(chain: ?*Function) void {
        var mut_chain = chain;
        while (mut_chain) |head| {
            mut_chain = head.next;
            head.destroy();
        }
    }
    pub fn destroyVariablesChain(chain: ?*TaggedVar) void {
        var mut_chain = chain;
        while (mut_chain) |head| {
            mut_chain = head.getNext();
            head.destroy();
        }
    }

    pub fn destroyFunctions(self: *Context) void {
        destroyFunctionsChain(self.func_chain);
    }

    pub fn destroyVariables(self: *Context) void {
        destroyVariablesChain(self.var_chain);
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

    pub fn countVariables(self: *const Context) usize {
        var iter = self.var_chain;
        var count: usize = 0;

        while (iter) |variable| : (iter = variable.getNext()) {
            count += 1;
        }

        return count;
    }

    pub fn countFunctions(self: *const Context) usize {
        var iter = self.func_chain;
        var count: usize = 0;

        while (iter) |func| : (iter = func.next) {
            count += 1;
        }

        return count;
    }

    pub fn resetVarChain(self: *Context) void {
        self.var_chain = null;
    }

    pub fn resetFuncChain(self: *Context) void {
        self.func_chain = null;
    }

    pub fn registerFunction(self: *Context, func: Function) !*Function {
        const ptr = try self.functions.create();
        ptr.* = func;

        if (self.func_chain) |head| {
            ptr.next = head;
            head.prev = ptr;
        }

        self.func_chain = ptr;

        return ptr;
    }

    pub fn backward(
        self: *Context,
        comptime T: type,
        variable: *TaggedVar,
    ) !void {
        std.debug.assert(!self.options.front_only);

        const variable_untagged = variable.asUntagged(T);
        const creator = variable.getCreator() orelse return error.NoCreator;

        var ones = try tomo.tensor.GPUTensor(T).initAsync(variable_untagged.data.base.getShape(), self.stream);
        errdefer ones.deinitAsync(self.stream);
        try ones.fill(if (T == BF16) BF16.fromF32(1.0) else 1.0, self.stream);

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
    }

    pub fn takeSnapshot(self: *const Context) Snapshot {
        return .{
            .start_var = null,
            .end_var = self.var_chain,
            .start_func = null,
            .end_func = self.func_chain,
            .context = self,
        };
    }
};
