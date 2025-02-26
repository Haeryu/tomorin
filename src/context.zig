const std = @import("std");
const tomo = @import("tomo");

const Stream = tomo.stream.Stream;
const CudaContext = tomo.cuda_context.CudaContext;

pub const Context = struct {
    variable_allocator: std.mem.Allocator,
    function_allocator: std.mem.Allocator,
    multi_purpose_allocator: std.mem.Allocator,
    cuda_context: *const CudaContext,
    stream: *const Stream,
    enable_backprop_graph: bool, // TODO: this name sucks
};
