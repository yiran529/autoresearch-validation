from functools import lru_cache
import os
import numpy as np
import torch

@lru_cache(None)
def _kernel(world,k,B,bf16,residual=False):
    os.environ.setdefault('CUPY_CACHE_DIR','/tmp/greedy-lore-cupy-cache')
    import cupy
    T=min(B,1024)
    typ='unsigned short' if bf16 else 'float'
    args=', '.join(f'const short* i{r}, const {typ}* v{r}' for r in range(world))
    scatter='\n'.join(f'''for(int j=t;j<{k};j+=256) {{
        long long p=(long long)group*{k}+j;
        int ix=i{r}[p]-tile*{T};
        if(ix>=0 && ix<{T}) decoded[{r}*{T}+ix]={'__uint_as_float((unsigned int)v'+str(r)+'[p]<<16)' if bf16 else 'v'+str(r)+'[p]'};
    }}''' for r in range(world))
    add='\n'.join(f'asm("add.rn.f32 %0, %1, %2;" : "=f"(acc) : "f"(acc), "f"(decoded[{r}*{T}+j]));' for r in range(world))
    calc=f'''float acc=0.0f; {add}
        float result; asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(acc), "f"({1/world:.17e}f));'''
    if residual:
        calc='float result; asm("sub.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(out[p]), "f"(decoded[j]));'
    src=f'''extern "C" __global__ void run({args}, float* out) {{
        int t=threadIdx.x; int group=blockIdx.x/{B//T}; int tile=blockIdx.x%{B//T};
        __shared__ float decoded[{world*T}];
        for(int j=t;j<{world*T};j+=256) decoded[j]=0.0f;
        __syncthreads(); {scatter} __syncthreads();
        for(int j=t;j<{T};j+=256) {{
            long long p=(long long)group*{B}+tile*{T}+j;
            {calc} out[p]=result;
        }}
    }}'''
    return cupy.RawKernel(src,'run',options=('--fmad=false',))

def aggregate(indices,values,B):
    import cupy
    groups,k=indices[0].shape
    out=torch.empty((groups,B),device=values[0].device,dtype=torch.float32)
    args=[]
    for i,v in zip(indices,values):args.extend((np.uint64(i.data_ptr()),np.uint64(v.data_ptr())))
    args.append(np.uint64(out.data_ptr()))
    with cupy.cuda.ExternalStream(torch.cuda.current_stream(out.device).cuda_stream):
        _kernel(len(indices),k,B,values[0].dtype==torch.bfloat16)((groups*(B//min(B,1024)),),(256,),tuple(args))
    return out

def residual_(h,indices,values):
    import cupy
    B=h.shape[1]
    with cupy.cuda.ExternalStream(torch.cuda.current_stream(h.device).cuda_stream):
        _kernel(1,indices.shape[1],B,values.dtype==torch.bfloat16,True)((h.shape[0]*(B//min(B,1024)),),(256,),tuple(np.uint64(t.data_ptr()) for t in (indices,values,h)))
    return h
