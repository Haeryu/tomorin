const tomo = @import("tomo");
const GPUTensor = tomo.tensor.GPUTensor;

pub fn Variable(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        data: *GPUTensor(T, rank),
    };
}
