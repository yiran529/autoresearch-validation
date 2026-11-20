"""One native-valued coordinate per group of four with FP32 local EF.

Worker supports differ: gather byte packets, decode each worker, then average.
Four two-bit support positions share a byte. The native values start at a
four-byte aligned offset. Padding groups have zero values and position zero.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from comm_hooks.default_hooks import _allreduce_fut
from comm_hooks.utils import HookState


def packet_layout(n, itemsize):
    groups = ((n + 15) // 16) * 4
    support_bytes = groups // 4
    offset = (support_bytes + 3) // 4 * 4
    return groups, offset, offset + groups * itemsize


def encode_reference(gradient, residual):
    n = gradient.numel()
    groups, offset, size = packet_layout(n, gradient.element_size())
    h = gradient.reshape(-1).float() + residual
    blocks = F.pad(h, (0, groups * 4 - n)).reshape(groups, 4)
    positions = blocks.abs().argmax(1)
    values = blocks.gather(1, positions[:, None]).flatten().to(gradient.dtype)
    packet = torch.zeros(size, dtype=torch.uint8, device=gradient.device)
    codes = positions.to(torch.uint8).reshape(-1, 4)
    packet[:groups//4] = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    packet[offset:] = values.contiguous().view(torch.uint8)
    local = torch.zeros_like(blocks).scatter_(1, positions[:, None], values.float()[:, None]).flatten()[:n]
    return packet, h - local


def decode_reference(packet, n, dtype):
    groups, offset, size = packet_layout(n, torch.empty((), dtype=dtype).element_size())
    if packet.numel() != size:
        raise ValueError('invalid sparse packet size')
    code = packet[:groups//4]
    positions = torch.stack([(code >> shift) & 3 for shift in (0, 2, 4, 6)], 1).flatten().long()
    values = packet[offset:].contiguous().view(dtype).float()
    return torch.zeros(groups, 4, dtype=torch.float32, device=packet.device).scatter_(1, positions[:, None], values[:, None]).flatten()[:n]


# Explicit non-FTZ operations preserve subnormal FP32 feedback, as in N011.
_CUDA = r'''
__device__ __forceinline__ float add_ieee(float a, float b) {
    float r; asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float sub_ieee(float a, float b) {
    float r; asm volatile("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float div_ieee(float a, float b) {
    float r; asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ unsigned short to_bf16(float value) {
    unsigned int b = __float_as_uint(value);
    b += 0x7fff + ((b >> 16) & 1);
    return (unsigned short)(b >> 16);
}
extern "C" __global__ void encode(const void* G, float* E, unsigned char* P,
                                 long long N, long long GROUPS, long long OFFSET, int BF16) {
    long long group = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    float h[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned int best = 0;
    int position = 0;
    for (int j = 0; j < 4; ++j) {
        long long i = group * 4 + j;
        if (i < N) {
            float g = BF16 ? __uint_as_float((unsigned int)((const unsigned short*)G)[i] << 16)
                           : ((const float*)G)[i];
            h[j] = add_ieee(g, E[i]);
        }
        unsigned int magnitude = __float_as_uint(h[j]) & 0x7fffffff;
        if (magnitude > best) { best = magnitude; position = j; }
    }
    int code = position;
    int p1 = __shfl_down_sync(0xffffffff, code, 1);
    int p2 = __shfl_down_sync(0xffffffff, code, 2);
    int p3 = __shfl_down_sync(0xffffffff, code, 3);
    if (group >= GROUPS) return;
    if ((threadIdx.x & 3) == 0) P[group / 4] = code | (p1 << 2) | (p2 << 4) | (p3 << 6);
    // Write alignment padding deterministically, even though it is never decoded.
    if (group == 0) for (long long k = GROUPS / 4; k < OFFSET; ++k) P[k] = 0;
    float local = h[position];
    if (BF16) {
        unsigned short v = to_bf16(local);
        ((unsigned short*)(P + OFFSET))[group] = v;
        local = __uint_as_float((unsigned int)v << 16);
    } else ((float*)(P + OFFSET))[group] = local;
    for (int j = 0; j < 4; ++j) {
        long long i = group * 4 + j;
        if (i < N) E[i] = sub_ieee(h[j], j == position ? local : 0.f);
    }
}
extern "C" __global__ void decode(const unsigned char* P, void* G, long long N,
                                 long long OFFSET, long long STRIDE, int WORLD, int BF16) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    long long group = i / 4;
    float total = 0.f;
    for (int rank = 0; rank < WORLD; ++rank) {
        const unsigned char* p = P + (long long)rank * STRIDE;
        int position = (p[group / 4] >> (2 * (group % 4))) & 3;
        float value = 0.f;
        if ((i % 4) == position) {
            value = BF16 ? __uint_as_float((unsigned int)((const unsigned short*)(p + OFFSET))[group] << 16)
                         : ((const float*)(p + OFFSET))[group];
        }
        total = add_ieee(total, value);
    }
    float mean = div_ieee(total, (float)WORLD);
    if (BF16) ((unsigned short*)G)[i] = to_bf16(mean);
    else ((float*)G)[i] = mean;
}
'''
_KERNELS = None


def _cuda_kernels():
    global _KERNELS
    if _KERNELS is None:
        import cupy
        _KERNELS = (cupy.RawKernel(_CUDA, 'encode', options=('--fmad=false',)),
                    cupy.RawKernel(_CUDA, 'decode', options=('--fmad=false',)))
    return _KERNELS


def encode_cuda(gradient, residual):
    import cupy
    import numpy as np
    if gradient.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('BlockTop1 CUDA requires BF16 or FP32')
    n = gradient.numel()
    groups, offset, size = packet_layout(n, gradient.element_size())
    packet = torch.empty(size, device=gradient.device, dtype=torch.uint8)
    kernel, _ = _cuda_kernels()
    with cupy.cuda.Device(gradient.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(gradient.device).cuda_stream):
        kernel(((groups + 255)//256,), (256,),
               (cupy.from_dlpack(gradient.view(torch.uint16) if gradient.dtype == torch.bfloat16 else gradient),
                cupy.from_dlpack(residual), cupy.from_dlpack(packet), np.int64(n), np.int64(groups),
                np.int64(offset), np.int32(gradient.dtype == torch.bfloat16)))
    return packet


def decode_cuda(received, gradient, world):
    import cupy
    import numpy as np
    _, offset, _ = packet_layout(gradient.numel(), gradient.element_size())
    _, kernel = _cuda_kernels()
    with cupy.cuda.Device(gradient.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(gradient.device).cuda_stream):
        kernel(((gradient.numel() + 255)//256,), (256,),
               (cupy.from_dlpack(received),
                cupy.from_dlpack(gradient.view(torch.uint16) if gradient.dtype == torch.bfloat16 else gradient),
                np.int64(gradient.numel()), np.int64(offset), np.int64(received.numel()//world),
                np.int32(world), np.int32(gradient.dtype == torch.bfloat16)))


class BlockTop1State(HookState):
    def __init__(self, process_group, start_compress_iter=0):
        super().__init__(process_group)
        self.start_compress_iter = start_compress_iter
        self.compressor_name = 'block_top1_ef'
        self.residuals = {}
        self.bucket_sizes = {}
        self.timing_period_steps = 1
        self.timing_refresh_steps = []

    @property
    def dense_bits(self):
        return self.total_bit_before_compression

    @property
    def payload_bits(self):
        return self.total_bit_after_compression

    @property
    def steps(self):
        return self.iter


def block_top1_hook(state: BlockTop1State, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    gradient = bucket.buffer()
    n = gradient.numel()
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world = group.size()
    state.total_bit_before_compression += n * gradient.element_size() * 8
    if state.iter < state.start_compress_iter:
        state.total_bit_after_compression += n * gradient.element_size() * 8
        state.maybe_increase_iter(bucket)
        return _allreduce_fut(group, gradient)
    index = bucket.index()
    if index not in state.residuals:
        state.residuals[index] = torch.zeros(n, device=gradient.device, dtype=torch.float32)
        state.bucket_sizes[index] = n
    if state.bucket_sizes[index] != n:
        raise ValueError('BlockTop1 bucket layout changed')
    residual = state.residuals[index]
    if gradient.is_cuda:
        packet = encode_cuda(gradient, residual)
    else:
        packet, updated = encode_reference(gradient, residual)
        residual.copy_(updated)
    state.total_bit_after_compression += packet.numel() * 8
    if bucket.is_last():
        state.timing_refresh_steps.append(state.iter + 1)
    state.maybe_increase_iter(bucket)
    if gradient.is_cuda:
        received = torch.empty(packet.numel() * world, device=gradient.device, dtype=torch.uint8)
        future = dist.all_gather_into_tensor(received, packet, group=group, async_op=True).get_future()
    else:
        received = [torch.empty_like(packet) for _ in range(world)]
        future = dist.all_gather(received, packet, group=group, async_op=True).get_future()

    def finish(fut):
        fut.value()  # propagate collective errors before touching output
        if gradient.is_cuda:
            decode_cuda(received, gradient, world)
        else:
            total = torch.zeros(n, device=gradient.device, dtype=torch.float32)
            for message in received:
                total.add_(decode_reference(message, n, gradient.dtype))
            gradient.copy_(total.div_(world))
        return gradient

    return future.then(finish)
