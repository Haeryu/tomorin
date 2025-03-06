const Function = @import("function.zig").Function;
const TaggedVar = @import("variable.zig").TaggedVar;
const Context = @import("context.zig").Context;
const std = @import("std");

pub const Snapshot = struct {
    start_var: ?*TaggedVar,
    end_var: ?*TaggedVar,

    start_func: ?*Function,
    end_func: ?*Function,

    context: *const Context,

    pub fn record(self: *Snapshot) void {
        self.start_func = self.context.func_chain;
        self.start_var = self.context.var_chain;
    }

    pub fn countVariables(self: *const Snapshot) usize {
        var count: usize = 0;
        var iter = self.start_var;
        while (iter != self.end_var) : (iter = iter.?.getNext()) {
            count += 1;
        }

        return count;
    }

    pub fn countFunctions(self: *const Snapshot) usize {
        var count: usize = 0;
        var iter = self.start_func;
        while (iter != self.end_func) : (iter = iter.?.next) {
            count += 1;
        }

        return count;
    }

    pub fn destroyVariables(self: *const Snapshot) void {
        var chain = self.start_var;
        while (chain) |head| {
            if (head == self.end_var) break;
            chain = head.getNext();
            head.destroy();
        }
    }

    pub fn releaseVariables(self: *const Snapshot) void {
        var chain = self.start_var;
        while (chain) |head| {
            if (head == self.end_var) break;
            chain = head.getNext();
            head.release();
        }
    }

    pub fn destroyFunctions(self: *const Snapshot) void {
        var chain = self.start_func;
        while (chain) |head| {
            if (head == self.end_func) break;
            chain = head.next;
            head.destroy();
        }
    }
};
