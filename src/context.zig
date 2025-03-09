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
    verbose_dot: bool = false,
    init_var_capacity: usize = 0,
    init_func_capacity: usize = 0,
    init_chain_capacity: usize = 0,
};

pub const Context = struct {
    cuda_context: *const CudaContext,
    stream: *const Stream,
    allocator: std.mem.Allocator,

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

        self.chains.deinit();
    }

    pub fn createChain(self: *Context) !*Chain {
        const ptr = try self.chains.create();
        ptr.* = try .init(self.allocator, self, self.options.init_func_capacity, self.options.init_var_capacity);

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

    pub fn backward(self: *Context, comptime T: type, variable: *TaggedVar) !void {
        try self.backwardEx(T, variable, self.current_chain.?);
    }

    pub fn backwardEx(self: *Context, comptime T: type, variable: *TaggedVar, chain: *Chain) !void {
        const variable_untagged = variable.asUntagged(T);
        const creator = variable.getCreator() orelse return error.NoCreator;

        var ones: tomo.tensor.GPUTensor(T) = try .initAsync(variable_untagged.data.base.getShape(), self.stream);
        errdefer ones.deinitAsync(self.stream);
        try ones.fill(if (T == BF16) BF16.fromF32(1.0) else 1.0, self.stream);

        const initial_grad = try chain.createVariable(T, ones, null);

        variable_untagged.grad = initial_grad;

        var function_queue: Function.Queue = .init(self.allocator, {});
        defer function_queue.deinit();

        var seen_set: Function.SeenSet = .init(self.allocator);
        defer seen_set.deinit();

        try function_queue.add(creator);
        try seen_set.put(creator, {});

        while (function_queue.removeOrNull()) |func| {
            try func.backward();
            try func.enqueue(&function_queue, &seen_set);
        }
    }
};
