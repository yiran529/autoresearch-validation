"""Worker-local top1-of4 sparsity, four-bit values and one composed FP32 EF.

Every 128 input coordinates send 8 support bytes, 16 nibble bytes and one
FP32 scale. Scales cover 32 selected maxima; no native-value cast precedes
quantization. AllGather preserves worker-local supports and scales.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from comm_hooks.default_hooks import _allreduce_fut
from comm_hooks.utils import HookState

INPUT_BLOCK = 128
SELECTED = 32
PACKET = 28


def encode_reference(gradient, residual):
    n = gradient.numel()
    h = gradient.reshape(-1).float() + residual
    blocks = F.pad(h, (0, (-n) % INPUT_BLOCK)).reshape(-1, SELECTED, 4)
    positions = blocks.abs().argmax(2)
    values = blocks.gather(2, positions[:, :, None]).squeeze(2)
    maximum = values.abs().amax(1, keepdim=True)
    scale = maximum / 7.
    minimum = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
    scale = torch.where((maximum != 0) & (scale == 0), torch.full_like(scale, minimum), scale)
    q = (values / torch.where(scale == 0, torch.ones_like(scale), scale)).round().clamp(-7, 7)
    codes = positions.to(torch.uint8).reshape(-1, 8, 4)
    values_code = (q + 7).to(torch.uint8)
    packet = torch.empty((blocks.shape[0], PACKET), dtype=torch.uint8, device=h.device)
    packet[:, :8] = codes[:, :, 0] | (codes[:, :, 1] << 2) | (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)
    packet[:, 8:24] = values_code[:, 0::2] | (values_code[:, 1::2] << 4)
    packet[:, 24:] = scale.contiguous().view(torch.uint8).reshape(-1, 4)
    local = torch.zeros_like(blocks).scatter_(2, positions[:, :, None], (q * scale)[:, :, None]).flatten()[:n]
    return packet.flatten(), h - local


def decode_reference(packet, n):
    if packet.numel() != ((n + INPUT_BLOCK - 1)//INPUT_BLOCK)*PACKET:
        raise ValueError('invalid SparseQ4 packet size')
    blocks = packet.reshape(-1, PACKET)
    code = blocks[:, :8]
    positions = torch.stack([(code >> shift) & 3 for shift in (0, 2, 4, 6)], -1).reshape(-1, SELECTED).long()
    value_code = blocks[:, 8:24]
    q = torch.stack([value_code & 15, value_code >> 4], -1).reshape(-1, SELECTED).float() - 7.
    scale = blocks[:, 24:].contiguous().view(torch.float32).reshape(-1, 1)
    out = torch.zeros((blocks.shape[0], SELECTED, 4), dtype=torch.float32, device=packet.device)
    return out.scatter_(2, positions[:, :, None], (q * scale)[:, :, None]).flatten()[:n]


# CuPy defaults include FTZ. Explicit IEEE instructions preserve subnormal EF,
# scales and reconstruction, as in the parent's codec and N011 block4 codec.
_CUDA = r'''
__device__ __forceinline__ float qadd(float a, float b) {
    float r; asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float qsub(float a, float b) {
    float r; asm volatile("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float qmul(float a, float b) {
    float r; asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float qdiv(float a, float b) {
    float r; asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ unsigned short to_bf16(float value) {
    unsigned int b = __float_as_uint(value);
    b += 0x7fff + ((b >> 16) & 1);
    return (unsigned short)(b >> 16);
}
extern "C" __global__ void encode(const void* G, float* E, unsigned char* P,
                                 long long N, int BF16) {
    int t = threadIdx.x;
    long long group = (long long)blockIdx.x * 32 + t;
    float h[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned int best = 0;
    int position = 0;
    for (int j = 0; j < 4; ++j) {
        long long i = group * 4 + j;
        if (i < N) {
            float g = BF16 ? __uint_as_float((unsigned int)((const unsigned short*)G)[i] << 16)
                           : ((const float*)G)[i];
            h[j] = qadd(g, E[i]);
        }
        unsigned int magnitude = __float_as_uint(h[j]) & 0x7fffffff;
        if (magnitude > best) { best = magnitude; position = j; }
    }
    unsigned int m = best;
    for (int d = 16; d; d >>= 1) m = max(m, __shfl_down_sync(0xffffffff, m, d));
    float scale = qdiv(__uint_as_float(__shfl_sync(0xffffffff, m, 0)), 7.f);
    if (__shfl_sync(0xffffffff, m, 0) != 0 && __float_as_uint(scale) == 0) scale = __uint_as_float(1);
    unsigned char* p = P + (long long)blockIdx.x * 28;
    if (t == 0) *((float*)(p + 24)) = scale;
    int q = __float_as_uint(scale) == 0 ? 0 : __float2int_rn(qdiv(h[position], scale));
    q = max(-7, min(7, q));
    int code = q + 7;
    int next = __shfl_down_sync(0xffffffff, code, 1);
    if ((t & 1) == 0) p[8 + (t >> 1)] = code | (next << 4);
    int p1 = __shfl_down_sync(0xffffffff, position, 1);
    int p2 = __shfl_down_sync(0xffffffff, position, 2);
    int p3 = __shfl_down_sync(0xffffffff, position, 3);
    if ((t & 3) == 0) p[t >> 2] = position | (p1 << 2) | (p2 << 4) | (p3 << 6);
    float local = qmul((float)q, scale);
    for (int j = 0; j < 4; ++j) {
        long long i = group * 4 + j;
        if (i < N) E[i] = qsub(h[j], j == position ? local : 0.f);
    }
}
extern "C" __global__ void decode(const unsigned char* P, void* G, long long N,
                                 long long STRIDE, int WORLD, int BF16) {
    long long group = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (group * 4 >= N) return;
    int t = group % 32;
    float total[4] = {0.f, 0.f, 0.f, 0.f};
    for (int rank = 0; rank < WORLD; ++rank) {
        const unsigned char* p = P + (long long)rank * STRIDE + (group / 32) * 28;
        int position = (p[t >> 2] >> (2 * (t & 3))) & 3;
        int code = (p[8 + (t >> 1)] >> (4 * (t & 1))) & 15;
        float local = qmul((float)(code - 7), *((const float*)(p + 24)));
        for (int j = 0; j < 4; ++j) total[j] = qadd(total[j], j == position ? local : 0.f);
    }
    for (int j = 0; j < 4; ++j) {
        long long i = group * 4 + j;
        if (i < N) {
            float mean = qdiv(total[j], (float)WORLD);
            if (BF16) ((unsigned short*)G)[i] = to_bf16(mean);
            else ((float*)G)[i] = mean;
        }
    }
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
        raise ValueError('SparseQ4 CUDA requires BF16 or FP32')
    n = gradient.numel()
    blocks = (n + INPUT_BLOCK - 1)//INPUT_BLOCK
    packet = torch.empty(blocks * PACKET, device=gradient.device, dtype=torch.uint8)
    kernel, _ = _cuda_kernels()
    with cupy.cuda.Device(gradient.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(gradient.device).cuda_stream):
        kernel((blocks,), (32,),
               (cupy.from_dlpack(gradient.view(torch.uint16) if gradient.dtype == torch.bfloat16 else gradient),
                cupy.from_dlpack(residual), cupy.from_dlpack(packet), np.int64(n),
                np.int32(gradient.dtype == torch.bfloat16)))
    return packet


def decode_cuda(received, gradient, world):
    import cupy
    import numpy as np
    groups = (gradient.numel() + 3)//4
    _, kernel = _cuda_kernels()
    with cupy.cuda.Device(gradient.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(gradient.device).cuda_stream):
        kernel(((groups + 255)//256,), (256,),
               (cupy.from_dlpack(received),
                cupy.from_dlpack(gradient.view(torch.uint16) if gradient.dtype == torch.bfloat16 else gradient),
                np.int64(gradient.numel()), np.int64(received.numel()//world),
                np.int32(world), np.int32(gradient.dtype == torch.bfloat16)))


class SparseQ4State(HookState):
    def __init__(self, process_group, start_compress_iter=0):
        super().__init__(process_group)
        self.start_compress_iter = start_compress_iter
        self.compressor_name = 'sparse_q4_ef'
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


def sparse_q4_hook(state: SparseQ4State, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
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
        raise ValueError('SparseQ4 bucket layout changed')
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
                total.add_(decode_reference(message, n))
            gradient.copy_(total.div_(world))
        return gradient

    return future.then(finish)
