"""Exact rank-ordered sparse mean; sparse offsets are unique within blocks."""
from functools import lru_cache
import os
import torch
import numpy as np

@lru_cache(None)
def _kernel(world, k, bf16):
    os.environ.setdefault('CUPY_CACHE_DIR', '/tmp/greedy-lore-cupy-cache')
    import cupy
    typ = 'unsigned short' if bf16 else 'float'
    args = ', '.join(f'const unsigned char* i{r}, const {typ}* v{r}' for r in range(world))
    scatter = '\n'.join(f'''if (t < {k}) {{
        long long p = (long long)blockIdx.x * {k} + t;
        decoded[{r} * 256 + i{r}[p]] = {'__uint_as_float((unsigned int)v'+str(r)+'[p] << 16)' if bf16 else 'v'+str(r)+'[p]'};
    }}''' for r in range(world))
    add = '\n'.join(f'asm("add.rn.f32 %0, %1, %2;" : "=f"(acc) : "f"(acc), "f"(decoded[{r}*256+t]));' for r in range(world))
    source = f'''extern "C" __global__ void sparse_mean({args}, float* out) {{
        int t = threadIdx.x;
        __shared__ float decoded[{world}*256];
        for(int r=0;r<{world};++r) decoded[r*256+t]=0.0f;
        __syncthreads();
        {scatter}
        __syncthreads();
        float acc=0.0f;
        {add}
        float result;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(acc), "f"({1.0/world:.17e}f));
        out[(long long)blockIdx.x*256+t]=result;
    }}'''
    return cupy.RawKernel(source,'sparse_mean',options=('--fmad=false',))

def aggregate(indices, values):
    import cupy
    blocks,k=indices[0].shape
    out=torch.empty((blocks,256),device=values[0].device,dtype=torch.float32)
    with cupy.cuda.ExternalStream(torch.cuda.current_stream(out.device).cuda_stream):
        args=[]
        for i,v in zip(indices,values): args.extend((np.uint64(i.data_ptr()), np.uint64(v.data_ptr())))
        args.append(np.uint64(out.data_ptr()))
        _kernel(len(indices),k,values[0].dtype==torch.bfloat16)((blocks,),(256,),tuple(args))
    return out

@lru_cache(None)
def _residual_kernel(k, bf16):
    os.environ.setdefault('CUPY_CACHE_DIR', '/tmp/greedy-lore-cupy-cache')
    import cupy
    typ = 'unsigned short' if bf16 else 'float'
    val = '__uint_as_float((unsigned int)v[p] << 16)' if bf16 else 'v[p]'
    source = f'''extern "C" __global__ void local_residual(float* h, const unsigned char* indices, const {typ}* v) {{
        int t=threadIdx.x;
        __shared__ float decoded[256];
        decoded[t]=0.0f;
        __syncthreads();
        if(t<{k}) {{
            long long p=(long long)blockIdx.x*{k}+t;
            decoded[indices[p]]={val};
        }}
        __syncthreads();
        long long p=(long long)blockIdx.x*256+t;
        float result;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(h[p]), "f"(decoded[t]));
        h[p]=result;
    }}'''
    return cupy.RawKernel(source,'local_residual',options=('--fmad=false',))

def residual_(h, indices, values):
    import cupy
    with cupy.cuda.ExternalStream(torch.cuda.current_stream(h.device).cuda_stream):
        # RawKernel copies these pointer values into CUDA's argument buffer.
        # PyTorch owns every tensor through the hook's final synchronization.
        args=tuple(np.uint64(t.data_ptr()) for t in (h,indices,values))
        _residual_kernel(indices.shape[1],values.dtype==torch.bfloat16)((h.shape[0],),(256,),args)
    return h
