"""Shared EMA prediction with locally guarded SparseQ4 innovations and FP32 EF.

The mode minimizes actual decoded remainder SSE for each 128-coordinate block.
The predictor is shared from decoded means; EF remains worker-local. This is a
local same-input safeguard, not a guarantee about global error or training loss.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from comm_hooks.default_hooks import _allreduce_fut
from comm_hooks.utils import HookState
from comm_hooks.sparse_q4_hook import (
    encode_reference as raw_encode, decode_reference as raw_decode,
    _CUDA as _RAW_CUDA,
)

INPUT_BLOCK = 128
PACKET = 32


def block_sse(remainder):
    """FP64: four coordinates/lane sequentially, then a 32-lane tree."""
    x = F.pad(remainder, (0, (-remainder.numel()) % 128)).double().reshape(-1, 32, 4)
    lane = x[:, :, 0].square()
    for j in range(1, 4):
        lane = lane + x[:, :, j].square()
    for d in (16, 8, 4, 2, 1):
        lane = lane[:, :d] + lane[:, d:2*d]
    return lane[:, 0]


def encode_reference(gradient, residual, predictor):
    n = gradient.numel()
    h = gradient.flatten().float() + residual
    raw, e0 = raw_encode(gradient, residual)
    innovation, _ = raw_encode(h - predictor, torch.zeros_like(residual))
    z1 = predictor + raw_decode(innovation, n)
    e1 = h - z1
    mode = block_sse(e1) < block_sse(e0)
    packet = torch.zeros((mode.numel(), PACKET), dtype=torch.uint8, device=h.device)
    packet[:, :28] = torch.where(mode[:, None], innovation.reshape(-1, 28), raw.reshape(-1, 28))
    packet[:, 28] = mode.to(torch.uint8)
    updated = torch.where(mode.repeat_interleave(128)[:n], e1, e0)
    return packet.flatten(), updated


def decode_reference(packet, n, predictor):
    if packet.numel() != ((n + 127)//128)*PACKET:
        raise ValueError('invalid guarded predictor packet size')
    blocks = packet.reshape(-1, PACKET)
    correction = raw_decode(blocks[:, :28].contiguous().flatten(), n)
    mode = blocks[:, 28].bool().repeat_interleave(128)[:n]
    return torch.where(mode, predictor + correction, correction)


# Reuse parent's explicit non-FTZ FP32 primitives and BF16 cast definition.
# Widening must preserve subnormals before the FP64 SSE accumulation too.
_CUDA = _RAW_CUDA.split('extern "C"')[0] + r'''
extern "C" __global__ void encode(const void* G, float* E, const float* A,
                                 unsigned char* P, long long N, int BF16) {
    int t = threadIdx.x;
    long long group = (long long)blockIdx.x * 32 + t;
    float h[4] = {0.f,0.f,0.f,0.f};
    float a[4] = {0.f,0.f,0.f,0.f};
    float v[2][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
    for (int j=0; j<4; ++j) {
        long long i = group*4+j;
        if (i<N) {
            float g = BF16 ? __uint_as_float((unsigned int)((const unsigned short*)G)[i]<<16)
                           : ((const float*)G)[i];
            h[j]=qadd(g,E[i]); a[j]=A[i];
            v[0][j]=h[j]; v[1][j]=qsub(h[j],a[j]);
        }
    }
    int pos[2]; int codes[2]; float scales[2]; float errors[2][4]; double sse[2];
    for (int b=0; b<2; ++b) {
        unsigned int best=0; pos[b]=0;
        for (int j=0;j<4;++j) {
            unsigned int magnitude=__float_as_uint(v[b][j]) & 0x7fffffff;
            if (magnitude>best) {best=magnitude;pos[b]=j;}
        }
        unsigned int m=best;
        for (int d=16;d;d>>=1) m=max(m,__shfl_down_sync(0xffffffff,m,d));
        unsigned int maximum=__shfl_sync(0xffffffff,m,0);
        scales[b]=qdiv(__uint_as_float(maximum),7.f);
        if (maximum!=0 && __float_as_uint(scales[b])==0) scales[b]=__uint_as_float(1);
        int q=__float_as_uint(scales[b])==0 ? 0 : __float2int_rn(qdiv(v[b][pos[b]],scales[b]));
        q=max(-7,min(7,q)); codes[b]=q+7;
        float local=qmul((float)q,scales[b]);
        sse[b]=0.;
        for (int j=0;j<4;++j) {
            float reconstructed=j==pos[b]?local:0.f;
            if (b==1) reconstructed=qadd(a[j],reconstructed);
            errors[b][j]=qsub(h[j],reconstructed);
            double e; asm volatile("cvt.f64.f32 %0, %1;" : "=d"(e) : "f"(errors[b][j]));
            sse[b] += e*e;
        }
        for (int d=16;d;d>>=1) sse[b]+=__shfl_down_sync(0xffffffff,sse[b],d);
        sse[b]=__shfl_sync(0xffffffff,sse[b],0);
    }
    int mode=sse[1]<sse[0];
    unsigned char* p=P+(long long)blockIdx.x*32;
    int position=pos[mode], code=codes[mode];
    if (t==0) {
        *((float*)(p+24))=scales[mode];
        p[28]=mode;p[29]=0;p[30]=0;p[31]=0;
    }
    int next=__shfl_down_sync(0xffffffff,code,1);
    if ((t&1)==0) p[8+(t>>1)]=code|(next<<4);
    int p1=__shfl_down_sync(0xffffffff,position,1);
    int p2=__shfl_down_sync(0xffffffff,position,2);
    int p3=__shfl_down_sync(0xffffffff,position,3);
    if ((t&3)==0) p[t>>2]=position|(p1<<2)|(p2<<4)|(p3<<6);
    for (int j=0;j<4;++j) {
        long long i=group*4+j;
        if (i<N) E[i]=errors[mode][j];
    }
}
extern "C" __global__ void decode(const unsigned char* P, void* G, float* A,
                                 long long N, long long STRIDE, int WORLD, int BF16) {
    long long group=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (group*4>=N) return;
    int t=group%32;
    float a[4]={0.f,0.f,0.f,0.f}, total[4]={0.f,0.f,0.f,0.f};
    for (int j=0;j<4;++j) if (group*4+j<N) a[j]=A[group*4+j];
    for (int rank=0;rank<WORLD;++rank) {
        const unsigned char* p=P+(long long)rank*STRIDE+(group/32)*32;
        int position=(p[t>>2]>>(2*(t&3)))&3;
        int code=(p[8+(t>>1)]>>(4*(t&1)))&15;
        float local=qmul((float)(code-7),*((const float*)(p+24)));
        int mode=p[28];
        for (int j=0;j<4;++j) {
            float z=j==position?local:0.f;
            if (mode) z=qadd(a[j],z);
            total[j]=qadd(total[j],z);
        }
    }
    for (int j=0;j<4;++j) {
        long long i=group*4+j;
        if (i<N) {
            float mean=(WORLD>0 && (WORLD&(WORLD-1))==0)
                ? qmul(total[j],1.f/(float)WORLD) : qdiv(total[j],(float)WORLD);
            A[i]=qadd(qmul(0.9f,a[j]),qmul(0.1f,mean));
            if (BF16) ((unsigned short*)G)[i]=to_bf16(mean);
            else ((float*)G)[i]=mean;
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


def encode_cuda(gradient, residual, predictor):
    import cupy
    import numpy as np
    if gradient.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Guarded predictor CUDA requires BF16 or FP32')
    n = gradient.numel()
    blocks = (n+127)//128
    packet = torch.empty(blocks*PACKET, device=gradient.device, dtype=torch.uint8)
    kernel, _ = _cuda_kernels()
    with cupy.cuda.Device(gradient.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(gradient.device).cuda_stream):
        kernel((blocks,), (32,),
               (cupy.from_dlpack(gradient.view(torch.uint16) if gradient.dtype == torch.bfloat16 else gradient),
                cupy.from_dlpack(residual), cupy.from_dlpack(predictor), cupy.from_dlpack(packet),
                np.int64(n), np.int32(gradient.dtype == torch.bfloat16)))
    return packet


def decode_cuda(received, gradient, predictor, world):
    import cupy
    import numpy as np
    _, kernel = _cuda_kernels()
    groups = (gradient.numel()+3)//4
    with cupy.cuda.Device(gradient.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(gradient.device).cuda_stream):
        kernel(((groups+255)//256,), (256,),
               (cupy.from_dlpack(received),
                cupy.from_dlpack(gradient.view(torch.uint16) if gradient.dtype == torch.bfloat16 else gradient),
                cupy.from_dlpack(predictor), np.int64(gradient.numel()),
                np.int64(received.numel()//world), np.int32(world),
                np.int32(gradient.dtype == torch.bfloat16)))


class GuardedPredictorState(HookState):
    def __init__(self, process_group, start_compress_iter=0):
        super().__init__(process_group)
        self.start_compress_iter = start_compress_iter
        self.compressor_name = 'guarded_predictor_q4_ef'
        self.residuals = {}
        self.predictors = {}
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


def guarded_predictor_hook(state: GuardedPredictorState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    gradient = bucket.buffer()
    n = gradient.numel()
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world = group.size()
    state.total_bit_before_compression += n*gradient.element_size()*8
    if state.iter < state.start_compress_iter:
        state.total_bit_after_compression += n*gradient.element_size()*8
        state.maybe_increase_iter(bucket)
        return _allreduce_fut(group, gradient)
    index = bucket.index()
    if index not in state.residuals:
        state.residuals[index] = torch.zeros(n, device=gradient.device, dtype=torch.float32)
        state.predictors[index] = torch.zeros(n, device=gradient.device, dtype=torch.float32)
        state.bucket_sizes[index] = n
    if state.bucket_sizes[index] != n:
        raise ValueError('Guarded predictor bucket layout changed')
    residual, predictor = state.residuals[index], state.predictors[index]
    if gradient.is_cuda:
        packet = encode_cuda(gradient, residual, predictor)
    else:
        packet, updated = encode_reference(gradient, residual, predictor)
        residual.copy_(updated)
    state.total_bit_after_compression += packet.numel()*8
    if bucket.is_last():
        state.timing_refresh_steps.append(state.iter+1)
    state.maybe_increase_iter(bucket)
    if gradient.is_cuda:
        received = torch.empty(packet.numel()*world, device=gradient.device, dtype=torch.uint8)
        future = dist.all_gather_into_tensor(received, packet, group=group, async_op=True).get_future()
    else:
        received = [torch.empty_like(packet) for _ in range(world)]
        future = dist.all_gather(received, packet, group=group, async_op=True).get_future()

    def finish(fut):
        fut.value()
        if gradient.is_cuda:
            decode_cuda(received, gradient, predictor, world)
        else:
            total = torch.zeros(n, device=gradient.device, dtype=torch.float32)
            for message in received:
                total.add_(decode_reference(message, n, predictor))
            mean = total.div_(world)
            predictor.copy_(predictor*0.9 + mean*0.1)
            gradient.copy_(mean)
        return gradient

    return future.then(finish)
