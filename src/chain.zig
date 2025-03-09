const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;
const std = @import("std");

pub const Chain = struct {
    func_chain: ?*Function = null,
    var_chain: ?*TaggedVar = null,

    prev: ?*Chain = null,
    next: ?*Chain = null,

    pub const empty: Chain = .{
        .func_chain = null,
        .var_chain = null,
        .prev = null,
        .next = null,
    };

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
