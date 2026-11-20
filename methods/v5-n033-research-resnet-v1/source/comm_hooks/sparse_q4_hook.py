"""Guarded top4-Q4/top5-Q3 equal-wire gradient codec with composed FP32 EF.

Each128-coordinate input sends32bytes. Scale sign tags the selected format.
The guard compares actual decoded FP32 residuals using ordered FP64 scores.
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


def _support_table(k):
    table = [None] * comb(16, k)
    for positions in combinations(range(16), k):
        table[sum(comb(p, j+1) for j, p in enumerate(positions))] = positions
    return table


_SUPPORTS5 = _support_table(5)


def _option(blocks, k, bits):
    positions = blocks.abs().argsort(dim=2, descending=True, stable=True)[:, :, :k].sort(2).values
    values = blocks.gather(2, positions)
    limit = (1 << (bits-1)) - 1
    maximum = blocks.abs().flatten(1).amax(1, keepdim=True)
    scale = maximum / limit
    minimum = torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps
    scale = torch.where((maximum != 0) & (scale == 0), torch.full_like(scale, minimum), scale)
    q = (values / torch.where(scale == 0, torch.ones_like(scale), scale).unsqueeze(2)).round().clamp(-limit, limit)
    local = torch.zeros_like(blocks).scatter_(2, positions, q * scale.unsqueeze(2))
    residual = blocks - local
    squares = residual.double().square()
    group_score = torch.zeros_like(squares[:, :, 0])
    for j in range(16):
        group_score = group_score + squares[:, :, j]
    score = torch.zeros_like(group_score[:, 0])
    for j in range(8):
        score = score + group_score[:, j]
    ranks = torch.zeros_like(positions[:, :, 0])
    for j in range(k):
        term = torch.ones_like(ranks)
        for v in range(j+1):
            term = term * (positions[:, :, j] - v)
        from math import factorial
        ranks = ranks + term // factorial(j+1)
    support_bits = 11 if k == 4 else 13
    support_bytes = 12 if k == 4 else 13
    packet = torch.zeros((blocks.shape[0], PACKET), dtype=torch.uint8, device=blocks.device)
    for byte in range(11 if k == 4 else 13):
        group, shift = divmod(8*byte, support_bits)
        word = ranks[:, group]
        if group < 7:
            word = word | (ranks[:, group+1] << support_bits)
        packet[:, byte] = ((word >> shift) & 255).byte()
    codes = (q + limit).long().flatten(1)
    for byte in range(16 if k == 4 else 15):
        index, shift = divmod(8*byte, bits)
        word = codes[:, index]
        for extra in range(1, 4):
            if index+extra < codes.shape[1]:
                word = word | (codes[:, index+extra] << (bits*extra))
        packet[:, support_bytes+byte] = ((word >> shift) & 255).byte()
    signed_scale = scale if k == 4 else -scale
    packet[:, 28:] = signed_scale.contiguous().view(torch.uint8).reshape(-1, 4)
    return packet, residual, score


def encode_reference(gradient, residual):
    n = gradient.numel()
    h = gradient.reshape(-1).float() + residual
    blocks = F.pad(h, (0, (-n) % INPUT_BLOCK)).reshape(-1, 8, 16)
    a, ea, sa = _option(blocks, 4, 4)
    b, eb, sb = _option(blocks, 5, 3)
    mode = sb < sa
    return torch.where(mode[:, None], b, a).flatten(), torch.where(mode[:, None, None], eb, ea).flatten()[:n]


def decode_reference(packet, n):
    if packet.numel() != ((n+127)//128)*32:
        raise ValueError('invalid SparseQ4 packet size')
    blocks = packet.reshape(-1, 32)
    signed = blocks[:, 28:].contiguous().view(torch.float32).reshape(-1, 1)
    mode = torch.signbit(signed).flatten()
    out = torch.zeros((blocks.shape[0], 8, 16), dtype=torch.float32, device=packet.device)
    for k, bits, table in [(4, 4, _SUPPORTS), (5, 3, _SUPPORTS5)]:
        chosen = ~mode if k == 4 else mode
        data = blocks[chosen]
        if not data.shape[0]:
            continue
        ranks = []
        support_bits = 11 if k == 4 else 13
        for group in range(8):
            byte, shift = divmod(support_bits*group, 8)
            word = data[:, byte].long() | (data[:, byte+1].long() << 8)
            if shift+support_bits > 16:
                word |= data[:, byte+2].long() << 16
            ranks.append((word >> shift) & ((1 << support_bits)-1))
        positions = torch.tensor(table, device=packet.device)[torch.stack(ranks, 1)]
        codes = []
        start = 12 if k == 4 else 13
        for index in range(8*k):
            byte, shift = divmod(bits*index, 8)
            word = data[:, start+byte].long()
            if shift+bits > 8:
                word |= data[:, start+byte+1].long() << 8
            codes.append((word >> shift) & ((1 << bits)-1))
        q = torch.stack(codes, 1).float() - ((1 << (bits-1))-1)
        local = torch.zeros((data.shape[0], 8, 16), dtype=torch.float32, device=packet.device)
        out[chosen] = local.scatter_(2, positions, (q*signed[chosen].abs()).reshape(-1, 8, k))
    return out.flatten()[:n]


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
    if (k == 4) return n*(n-1)*(n-2)*(n-3)/24;
    return n*(n-1)*(n-2)*(n-3)*(n-4)/120;
}
__device__ unsigned short SUPPORT_MASK[1820] = {SUPPORT_MASK_VALUES};
__device__ unsigned short SUPPORT_MASK5[4368] = {SUPPORT_MASK5_VALUES};
__device__ __forceinline__ int code_at(int index, int* codes) {
    int v = 0;
    for (int k=0; k<5; ++k) {
        int candidate = __shfl_sync(0xffffffff, codes[k], 4*min(index/5,7));
        if (index%5 == k) v = candidate;
    }
    return index < 40 ? v : 0;
}
__device__ __forceinline__ double qdouble(float a) {
    // Explicit conversion avoids NVRTC FTZ on finite subnormal residuals.
    double r; asm volatile("cvt.f64.f32 %0, %1;" : "=d"(r) : "f"(a)); return r;
}
extern "C" __global__ void encode(const void* G, float* E, unsigned char* P,
                                 long long N, int BF16) {
    int t=threadIdx.x, lane=t&3, base=t&~3;
    long long group4=(long long)blockIdx.x*32+t;
    float h[4]={0.f,0.f,0.f,0.f}; unsigned int maximum=0;
    for (int j=0; j<4; ++j) {
        long long i=group4*4+j;
        if (i<N) {
            float g=BF16 ? __uint_as_float((unsigned int)((const unsigned short*)G)[i]<<16) : ((const float*)G)[i];
            h[j]=qadd(g,E[i]);
        }
        maximum=max(maximum,__float_as_uint(h[j])&0x7fffffff);
    }
    unsigned int m=maximum;
    for (int step=16;step;step>>=1) m=max(m,__shfl_down_sync(0xffffffff,m,step));
    maximum=__shfl_sync(0xffffffff,m,0);
    int selected[5]; float values[5]; unsigned int used=0;
    for (int k=0;k<5;++k) {
        int best=16; unsigned int best_mag=0; float best_value=0.f;
        for (int j=0;j<4;++j) {
            int pos=4*lane+j; unsigned int mag=__float_as_uint(h[j])&0x7fffffff;
            if (!(used&(1u<<pos)) && (best==16 || mag>best_mag)) {
                best=pos; best_mag=mag; best_value=h[j];
            }
        }
        for(int offset=1;offset<=2;offset*=2) {
            int other=__shfl_xor_sync(0xffffffff,best,offset);
            unsigned int mag=__shfl_xor_sync(0xffffffff,best_mag,offset);
            float v=__shfl_xor_sync(0xffffffff,best_value,offset);
            if(other<16 && (best==16 || mag>best_mag || (mag==best_mag && other<best))) {
                best=other;best_mag=mag;best_value=v;
            }
        }
        selected[k]=best;values[k]=best_value;used|=1u<<best;
    }
    int positions[2][5], codes[2][5]; float scales[2], decoded[2][5]; double scores[2];
    for(int option=0;option<2;++option) {
        int count=4+option, limit=option ? 3 : 7;
        float scale=qdiv(__uint_as_float(maximum),(float)limit);
        if(maximum && __float_as_uint(scale)==0) scale=__uint_as_float(1);
        scales[option]=scale;
        for(int k=0;k<count;++k) positions[option][k]=selected[k];
        for(int j=1;j<count;++j) {
            int pos=positions[option][j], i=j;
            while(i>0 && positions[option][i-1]>pos) {positions[option][i]=positions[option][i-1];--i;}
            positions[option][i]=pos;
        }
        for(int k=0;k<count;++k) {
            float v=0.f;
            for(int j=0;j<count;++j) if(selected[j]==positions[option][k]) v=values[j];
            int q=__float_as_uint(scale)==0 ? 0 : __float2int_rn(qdiv(v,scale));
            q=max(-limit,min(limit,q)); codes[option][k]=q+limit;
            decoded[option][k]=qmul((float)q,scale);
        }
        if(!option) codes[option][4]=0;
        double squares[4];
        for(int j=0;j<4;++j) {
            float v=0.f;
            for(int k=0;k<count;++k) if(positions[option][k]==4*lane+j) v=decoded[option][k];
            double r=qdouble(qsub(h[j],v)); squares[j]=r*r;
        }
        double score=0.;
        for(int pos=0;pos<16;++pos) {
            double v=__shfl_sync(0xffffffff,squares[pos%4],base+pos/4);
            score=score+v;
        }
        double total=0.;
        for(int g=0;g<8;++g) total=total+__shfl_sync(0xffffffff,score,4*g);
        scores[option]=total;
    }
    int mode=scores[1]<scores[0], count=4+mode, support_bits=mode?13:11;
    int rank=0;
    for(int k=0;k<count;++k) rank+=choose(positions[mode][k],k+1);
    unsigned char* p=P+(long long)blockIdx.x*32;
    if(t==0) { *((float*)(p+28))=__uint_as_float(__float_as_uint(scales[mode]) | ((unsigned int)mode<<31)); if(!mode) p[11]=0; }
    int bg=min((8*t)/support_bits,7), shift=(8*t)%support_bits;
    unsigned int word=__shfl_sync(0xffffffff,rank,4*bg);
    unsigned int next=__shfl_sync(0xffffffff,rank,4*min(bg+1,7));
    if(bg<7) word|=next<<support_bits;
    if(t<(mode?13:11)) p[t]=(word>>shift)&255;
    if(mode) {
        int index=(8*t)/3, off=(8*t)%3;
        // All lanes participate: helper source lanes may be nonwriters.
        unsigned int v=code_at(index,codes[1]) | (code_at(index+1,codes[1])<<3) | (code_at(index+2,codes[1])<<6) | (code_at(index+3,codes[1])<<9);
        if(t<15) p[13+t]=(v>>off)&255;
    } else {
        int code=codes[0][lane], other=__shfl_down_sync(0xffffffff,code,1);
        if(!(t&1)) p[12+(t>>1)]=code|(other<<4);
    }
    for(int j=0;j<4;++j) {
        float v=0.f;
        for(int k=0;k<count;++k) if(positions[mode][k]==4*lane+j) v=decoded[mode][k];
        long long i=group4*4+j;
        if(i<N) E[i]=qsub(h[j],v);
    }
}
extern "C" __global__ void decode(const unsigned char* P, void* G, long long N,
                                 long long STRIDE, int WORLD, int BF16) {
    long long group4 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (group4*4 >= N) return;
    int group = (group4 % 32) / 4, quarter = group4 & 3;
    float total[4] = {0.f, 0.f, 0.f, 0.f};
    for (int rank = 0; rank < WORLD; ++rank) {
        const unsigned char* p = P + (long long)rank*STRIDE + (group4/32)*32;
        float signed_scale = *((const float*)(p+28));
        int mode = (__float_as_uint(signed_scale)>>31), width=mode?13:11;
        float scale = __uint_as_float(__float_as_uint(signed_scale)&0x7fffffff);
        int bit=width*group, byte=bit/8, shift=bit%8;
        unsigned int word=(unsigned int)p[byte] | ((unsigned int)p[byte+1]<<8);
        if(shift+width>16) word|=(unsigned int)p[byte+2]<<16;
        unsigned int support=(word>>shift)&((1u<<width)-1);
        unsigned int mask=mode ? SUPPORT_MASK5[support] : SUPPORT_MASK[support];
        for(int j=0;j<4;++j) {
            int pos=4*quarter+j; float value=0.f;
            if(mask&(1u<<pos)) {
                int index=(4+mode)*group+__popc(mask&((1u<<pos)-1));
                int bits=mode?3:4, b=bits*index/8, off=(bits*index)%8, start=mode?13:12;
                unsigned int w=p[start+b];
                if(off+bits>8) w|=(unsigned int)p[start+b+1]<<8;
                int code=(w>>off)&((1<<bits)-1);
                value=qmul((float)(code-(mode?3:7)),scale);
            }
            total[j] = qadd(total[j], value);
        }
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
_CUDA = _CUDA.replace('SUPPORT_MASK5_VALUES', ','.join(str(sum(1 << p for p in positions)) for positions in _SUPPORTS5))
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
        self.compressor_name = 'adaptive_density_precision_ef'
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
