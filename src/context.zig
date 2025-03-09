const std = @import("std");
const tomo = @import("tomo");

const GPUTensor = tomo.tensor.GPUTensor;

const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;
const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;
const Variable = @import("variable.zig").Variable;
const Chain = @import("chain.zig").Chain;
const Pool = @import("nina").container.pool.Pool;
const BF16 = tomo.BF16;

const function = @import("function.zig");

const Snapshot = @import("snapshot.zig").Snapshot;

pub const ContextOptions = struct {
    aggressive_release: bool = false,
    verbose_dot: bool = false,
    init_var_capacity: usize = 0,
    init_func_capacity: usize = 0,
    init_chain_capacity: usize = 0,
};

pub const Context = struct {
    cuda_context: *const CudaContext,
    stream: *const Stream,
    allocator: std.mem.Allocator,

    functions: std.heap.MemoryPool(Function),
    tagged_vars: std.heap.MemoryPool(TaggedVar),
    chains: std.heap.MemoryPool(Chain),

    chain_head: ?*Chain,
    current_chain: ?*Chain,

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
            .functions = try .initPreheated(allocator, options.init_func_capacity),
            .tagged_vars = try .initPreheated(allocator, options.init_var_capacity),
            .chains = try .initPreheated(allocator, options.init_chain_capacity),
            .options = options,
            .chain_head = null,
            .current_chain = null,
        };
    }

    pub fn deinit(self: *Context) void {
        var iter = self.chain_head;
        while (iter) |chain| {
            iter = chain.next;
            chain.destroy();
        }
        self.functions.deinit();
        self.tagged_vars.deinit();
        self.chains.deinit();
    }

    pub fn createChain(self: *Context) !*Chain {
        const ptr = try self.chains.create();
        ptr.* = .empty;

        if (self.chain_head) |head| {
            ptr.next = head;
            head.prev = ptr;
        }

        self.chain_head = ptr;

        return ptr;
    }

    pub fn destroyChain(self: *Context, chain: *Chain) void {
        if (chain.prev) |prev| {
            prev.next = chain.next;
        }
        if (chain.next) |next| {
            next.prev = chain.prev;
        }
        chain.prev = null;
        chain.next = null;

        chain.destroy();
        self.chains.destroy(chain);
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
            .chain = self.current_chain.?,
        };

        const tagged = TaggedVar.init(T, variable);

        const ptr = try self.tagged_vars.create();
        ptr.* = tagged;

        self.current_chain.?.chainVariable(ptr);

        return ptr;
    }

    pub fn createVariableEx(
        self: *Context,
        comptime T: type,
        data: GPUTensor(T),
        name: ?[]const u8,
        chain: *Chain,
    ) !*TaggedVar {
        const variable: Variable(T) = .{
            .data = data,
            .name = name,
            .context = self,
            .protected = false,
            .prev = null,
            .next = null,
            .chain = chain,
        };

        const tagged = TaggedVar.init(T, variable);

        const ptr = try self.tagged_vars.create();
        ptr.* = tagged;

        chain.chainVariable(ptr);

        return ptr;
    }

    pub fn registerFunction(self: *Context, func: Function, chain: *Chain) !*Function {
        const ptr = try self.functions.create();
        ptr.* = func;

        chain.chainFunction(ptr);

        return ptr;
    }

    pub fn backward(
        self: *Context,
        comptime T: type,
        variable: *TaggedVar,
    ) !void {
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
};
