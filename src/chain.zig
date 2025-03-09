const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;
const std = @import("std");
const Variable = @import("variable.zig").Variable;
const Context = @import("context.zig").Context;
const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;

pub const Chain = struct {
    func_chain: ?*Function = null,
    var_chain: ?*TaggedVar = null,
    context: *Context,

    functions: std.heap.MemoryPool(Function),
    tagged_vars: std.heap.MemoryPool(TaggedVar),

    prev: ?*Chain = null,
    next: ?*Chain = null,

    pub const empty: Chain = .{
        .func_chain = null,
        .var_chain = null,
        .prev = null,
        .next = null,
    };

    pub fn init(allocator: std.mem.Allocator, context: *Context, init_func_capacity: usize, init_var_capacity: usize) !Chain {
        return .{
            .func_chain = null,
            .var_chain = null,
            .context = context,
            .functions = try .initPreheated(allocator, init_func_capacity),
            .tagged_vars = try .initPreheated(allocator, init_var_capacity),
            .prev = null,
            .next = null,
        };
    }

    pub fn createVariable(
        self: *Chain,
        comptime T: type,
        data: GPUTensor(T),
        name: ?[]const u8,
    ) !*TaggedVar {
        const variable: Variable(T) = .{
            .data = data,
            .name = name,
            .context = self.context,
            .protected = false,
            .prev = null,
            .next = null,
            .chain = self,
        };

        const tagged: TaggedVar = .init(T, variable);

        const ptr = try self.tagged_vars.create();
        ptr.* = tagged;

        self.chainVariable(ptr);

        return ptr;
    }

    pub fn registerFunction(self: *Chain, func: Function) !*Function {
        const ptr = try self.functions.create();
        ptr.* = func;

        self.chainFunction(ptr);

        return ptr;
    }

    pub fn chainVariable(self: *Chain, variable: *TaggedVar) void {
        if (self.var_chain) |head| {
            variable.setNext(head);
            head.setPrev(variable);
        }

        self.var_chain = variable;
    }

    pub fn chainFunction(self: *Chain, function: *Function) void {
        if (self.func_chain) |head| {
            function.next = head;
            head.prev = function;
        }

        self.func_chain = function;
    }

    pub fn destroyFunctions(self: *Chain) void {
        while (self.func_chain) |head| {
            //  self.func_chain = head.next;

            head.destroy();
        }
    }

    pub fn destroyVariables(self: *Chain) void {
        while (self.var_chain) |head| {
            //  self.var_chain = head.getNext();
            head.destroy();
        }
    }

    pub fn destroy(self: *Chain) void {
        self.destroyFunctions();
        self.destroyVariables();
        self.functions.deinit();
        self.tagged_vars.deinit();
    }

    pub fn clear(self: *Chain) void {
        self.destroyFunctions();
        self.destroyVariables();
    }

    pub fn countVariables(self: *const Chain) usize {
        var iter = self.var_chain;
        var count: usize = 0;

        while (iter) |variable| : (iter = variable.getNext()) {
            count += 1;
        }

        return count;
    }

    pub fn countFunctions(self: *const Chain) usize {
        var iter = self.func_chain;
        var count: usize = 0;

        while (iter) |func| : (iter = func.next) {
            count += 1;
        }

        return count;
    }
};
