"""Guarded coordinate/Hadamard top4-of16 at the parent's 32-byte budget.

Byte 11 carries eight basis flags. All scales, support ranks, coefficient
slots, rank-order aggregation and continuous local error feedback retain the
N024 definitions. The guard scores actual FP32 residuals in explicit FP64.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from comm_hooks.default_hooks import _allreduce_fut
from comm_hooks.utils import HookState

INPUT_BLOCK = 128
SELECTED = 32
PACKET = 32


# Colexicographic rank is a bijection on all C(16,4)=1820 supports.
from itertools import combinations
from math import comb
_SUPPORTS = [None] * comb(16, 4)
for _positions in combinations(range(16), 4):
    _SUPPORTS[sum(comb(p, j + 1) for j, p in enumerate(_positions))] = _positions


def hadamard16(values):
    """Ordered normalized Sylvester butterfly on the last dimension."""
    x = values.clone()
    for stride in (1, 2, 4, 8):
        chunks = x.reshape(*x.shape[:-1], -1, 2, stride)
        left, right = chunks[..., 0, :], chunks[..., 1, :]
        x = torch.stack((left + right, left - right), -2).reshape_as(x)
    return x * .25


def guard_score(residual):
    r = residual.double().reshape(*residual.shape[:-1], 4, 4)
    lane = torch.zeros_like(r[..., 0])
    for j in range(4):
        lane = lane + r[..., j] * r[..., j]
    return (lane[..., 0] + lane[..., 1]) + (lane[..., 2] + lane[..., 3])


def encode_reference(gradient, residual):
    n = gradient.numel()
    h = gradient.reshape(-1).float() + residual
    blocks = F.pad(h, (0, (-n) % INPUT_BLOCK)).reshape(-1, 8, 16)
    maximum = blocks.abs().amax((1, 2), keepdim=True)
    scale = maximum / 7.
    minimum = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
    scale = torch.where((maximum != 0) & (scale == 0), torch.full_like(scale, minimum), scale)
    options = []
    for transformed in (False, True):
        x = hadamard16(blocks) if transformed else blocks
        positions = x.abs().argsort(dim=2, descending=True, stable=True)[:, :, :4].sort(2).values
        values = x.gather(2, positions)
        q = (values / torch.where(scale == 0, torch.ones_like(scale), scale)).round().clamp(-7, 7)
        decoded = torch.zeros_like(blocks).scatter_(2, positions, q * scale)
        if transformed:
            decoded = hadamard16(decoded)
        options.append((positions, q, decoded, guard_score(blocks - decoded)))
    use = options[1][3] < options[0][3]
    positions, q, local = [torch.where(use[..., None], options[1][j], options[0][j]) for j in range(3)]
    a, b, c, d = positions.unbind(2)
    ranks = a + b*(b-1)//2 + c*(c-1)*(c-2)//6 + d*(d-1)*(d-2)*(d-3)//24
    packet = torch.zeros((blocks.shape[0], PACKET), dtype=torch.uint8, device=h.device)
    for byte in range(11):
        group, shift = divmod(8*byte, 11)
        word = ranks[:, group]
        if group < 7:
            word = word | (ranks[:, group + 1] << 11)
        packet[:, byte] = ((word >> shift) & 255).to(torch.uint8)
    for group in range(8):
        packet[:, 11] |= use[:, group].to(torch.uint8) << group
    codes = (q.reshape(-1, SELECTED) + 7).to(torch.uint8)
    packet[:, 12:28] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    packet[:, 28:] = scale.contiguous().view(torch.uint8).reshape(-1, 4)
    return packet.flatten(), h - local.flatten()[:n]


def decode_reference(packet, n):
    if packet.numel() != ((n + INPUT_BLOCK - 1)//INPUT_BLOCK)*PACKET:
        raise ValueError('invalid SparseQ4 packet size')
    blocks = packet.reshape(-1, PACKET)
    ranks = []
    for group in range(8):
        byte, shift = divmod(11*group, 8)
        word = blocks[:, byte].long() | (blocks[:, byte+1].long() << 8)
        if shift > 5:
            word |= blocks[:, byte+2].long() << 16
        ranks.append((word >> shift) & 2047)
    positions = torch.tensor(_SUPPORTS, device=packet.device)[torch.stack(ranks, 1)]
    codes = blocks[:, 12:28]
    q = torch.stack([codes & 15, codes >> 4], -1).reshape(-1, 8, 4).float() - 7.
    scale = blocks[:, 28:].contiguous().view(torch.float32).reshape(-1, 1, 1)
    out = torch.zeros((blocks.shape[0], 8, 16), dtype=torch.float32, device=packet.device)
    out.scatter_(2, positions, q * scale)
    flags = (blocks[:, 11, None].long() >> torch.arange(8, device=packet.device)) & 1
    return torch.where(flags[..., None].bool(), hadamard16(out), out).flatten()[:n]


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
__device__ __forceinline__ int choose(int n, int k) {
    if (k == 1) return n;
    if (k == 2) return n*(n-1)/2;
    if (k == 3) return n*(n-1)*(n-2)/6;
    return n*(n-1)*(n-2)*(n-3)/24;
}
__device__ const unsigned short SUPPORT_MASK[1820] = {SUPPORT_MASK_VALUES};
__device__ __forceinline__ void transform(float* v, int lane) {
    float w[4];
    w[0]=qadd(v[0],v[1]); w[1]=qsub(v[0],v[1]);
    w[2]=qadd(v[2],v[3]); w[3]=qsub(v[2],v[3]);
    v[0]=qadd(w[0],w[2]); v[1]=qadd(w[1],w[3]);
    v[2]=qsub(w[0],w[2]); v[3]=qsub(w[1],w[3]);
    for(int bit=1; bit<=2; bit*=2) {
        for(int j=0;j<4;++j) {
            float other=__shfl_xor_sync(0xffffffff,v[j],bit);
            v[j]=(lane&bit) ? qsub(other,v[j]) : qadd(v[j],other);
        }
    }
    for(int j=0;j<4;++j) v[j]=qmul(v[j],.25f);
}
__device__ __forceinline__ int encode_group(float* input, float scale, int lane,
                                           int* selected, float* local, int* code) {
    float selected_values[4]; unsigned int used = 0;
    for (int k = 0; k < 4; ++k) {
        int best = 16; unsigned int best_mag = 0; float best_value = 0.f;
        for (int j = 0; j < 4; ++j) {
            int pos = 4*lane+j;
            unsigned int mag = __float_as_uint(input[j]) & 0x7fffffff;
            if (!(used & (1u << pos)) && (best == 16 || mag > best_mag)) {
                best = pos; best_mag = mag; best_value = input[j];
            }
        }
        for (int offset = 1; offset <= 2; offset *= 2) {
            int other_pos = __shfl_xor_sync(0xffffffff, best, offset);
            unsigned int other_mag = __shfl_xor_sync(0xffffffff, best_mag, offset);
            float other_value = __shfl_xor_sync(0xffffffff, best_value, offset);
            if (other_pos < 16 && (best == 16 || other_mag > best_mag || (other_mag == best_mag && other_pos < best))) {
                best = other_pos; best_mag = other_mag; best_value = other_value;
            }
        }
        selected[k] = best; selected_values[k] = best_value; used |= 1u << best;
    }
    // Canonical spatial order removes the 4! transmitted permutations.
    for (int j = 1; j < 4; ++j) {
        int pos = selected[j]; float v = selected_values[j]; int i = j;
        while (i > 0 && selected[i-1] > pos) {
            selected[i] = selected[i-1]; selected_values[i] = selected_values[i-1]; --i;
        }
        selected[i] = pos; selected_values[i] = v;
    }

    int rank=0;
    for(int j=0;j<4;++j) rank+=choose(selected[j],j+1);
    int q=__float_as_uint(scale)==0 ? 0 : __float2int_rn(qdiv(selected_values[lane],scale));
    q=max(-7,min(7,q)); *code=q+7;
    float quant=qmul((float)q,scale), delivered[4];
    int base=threadIdx.x&~3;
    for(int k=0;k<4;++k) delivered[k]=__shfl_sync(0xffffffff,quant,base+k);
    for(int j=0;j<4;++j) {
        local[j]=0.f;
        for(int k=0;k<4;++k) if(selected[k]==4*lane+j) local[j]=delivered[k];
    }
    return rank;
}
__device__ __forceinline__ double score(float* h, float* decoded) {
    double error=0.;
    for(int j=0;j<4;++j) {
        float residual=qsub(h[j],decoded[j]);
        double r; asm volatile("cvt.f64.f32 %0, %1;" : "=d"(r) : "f"(residual));
        error=__fma_rn(r,r,error);
    }
    error+=__shfl_xor_sync(0xffffffff,error,1);
    error+=__shfl_xor_sync(0xffffffff,error,2);
    return error;
}
extern "C" __global__ void encode(const void* G, float* E, unsigned char* P,
                                 long long N, int BF16) {
    int t=threadIdx.x,lane=t&3;
    long long group4=(long long)blockIdx.x*32+t;
    float h[4]={0.f,0.f,0.f,0.f}; unsigned int maximum=0;
    for(int j=0;j<4;++j) {
        long long i=group4*4+j;
        if(i<N) {
            float g=BF16 ? __uint_as_float((unsigned int)((const unsigned short*)G)[i]<<16) : ((const float*)G)[i];
            h[j]=qadd(g,E[i]);
        }
        maximum=max(maximum,__float_as_uint(h[j])&0x7fffffff);
    }
    unsigned int m=maximum;
    for(int step=16;step;step>>=1) m=max(m,__shfl_down_sync(0xffffffff,m,step));
    maximum=__shfl_sync(0xffffffff,m,0);
    float scale=qdiv(__uint_as_float(maximum),7.f);
    if(maximum!=0 && __float_as_uint(scale)==0) scale=__uint_as_float(1);
    float z[4];for(int j=0;j<4;++j)z[j]=h[j];
    transform(z,lane);
    int positions0[4],positions1[4],code0,code1;float local0[4],local1[4];
    int rank0=encode_group(h,scale,lane,positions0,local0,&code0);
    int rank1=encode_group(z,scale,lane,positions1,local1,&code1);
    transform(local1,lane);
    double error0=score(h,local0),error1=score(h,local1);
    bool use=error1<error0;
    int rank=use?rank1:rank0,code=use?code1:code0;
    unsigned char* p=P+(long long)blockIdx.x*32;
    unsigned int flags=__ballot_sync(0xffffffff,use);
    if(t==0) {
        *((float*)(p+28))=scale; unsigned int packed=0;
        for(int g=0;g<8;++g)packed|=((flags>>(4*g))&1)<<g;
        p[11]=packed;
    }
    int byte_group=min((8*t)/11,7),shift=(8*t)%11;
    unsigned int word=(unsigned int)__shfl_sync(0xffffffff,rank,4*byte_group);
    unsigned int next_rank=(unsigned int)__shfl_sync(0xffffffff,rank,4*min(byte_group+1,7));
    if(byte_group<7)word|=next_rank<<11;
    if(t<11)p[t]=(word>>shift)&255;
    int next=__shfl_down_sync(0xffffffff,code,1);
    if((t&1)==0)p[12+(t>>1)]=code|(next<<4);
    for(int j=0;j<4;++j) {
        long long i=group4*4+j;
        if(i<N)E[i]=qsub(h[j],use?local1[j]:local0[j]);
    }
}
extern "C" __global__ void decode(const unsigned char* P, void* G, long long N,
                                 long long STRIDE, int WORLD, int BF16) {
    long long group4 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (group4/32 >= (N+127)/128) return;
    int group = (group4 % 32) / 4, quarter = group4 & 3;
    float total[4] = {0.f, 0.f, 0.f, 0.f};
    for (int rank = 0; rank < WORLD; ++rank) {
        const unsigned char* p = P + (long long)rank*STRIDE + (group4/32)*32;
        int bit = 11*group, byte = bit/8, shift = bit%8;
        unsigned int word = (unsigned int)p[byte] | ((unsigned int)p[byte+1]<<8);
        if (shift > 5) word |= (unsigned int)p[byte+2]<<16;
        unsigned int mask = SUPPORT_MASK[(word>>shift)&2047];
        float scale = *((const float*)(p+28));
        float local[4];
        for (int j = 0; j < 4; ++j) {
            int pos = 4*quarter+j;
            float value = 0.f;
            if (mask & (1u<<pos)) {
                int index = 4*group+__popc(mask & ((1u<<pos)-1));
                int code = (p[12+index/2] >> (4*(index&1))) & 15;
                value = qmul((float)(code-7), scale);
            }
            local[j] = value;
        }
        float rotated[4]; for(int j=0;j<4;++j) rotated[j]=local[j];
        transform(rotated,quarter);
        if((p[11]>>group)&1) for(int j=0;j<4;++j) local[j]=rotated[j];
        for(int j=0;j<4;++j) total[j]=qadd(total[j],local[j]);
    }
    for (int j = 0; j < 4; ++j) {
        long long i = group4*4+j;
        if (i < N) {
            float mean = (WORLD > 0 && (WORLD & (WORLD-1)) == 0)
                         ? qmul(total[j], qdiv(1.f, (float)WORLD)) : qdiv(total[j], (float)WORLD);
            if (BF16) ((unsigned short*)G)[i] = to_bf16(mean);
            else ((float*)G)[i] = mean;
        }
    }
}
'''
_CUDA = _CUDA.replace('SUPPORT_MASK_VALUES', ','.join(str(sum(1 << p for p in positions)) for positions in _SUPPORTS))
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
    groups = ((gradient.numel() + 127)//128)*32
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
        self.compressor_name = 'basis_pooled_sparse_q4_ef'
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
