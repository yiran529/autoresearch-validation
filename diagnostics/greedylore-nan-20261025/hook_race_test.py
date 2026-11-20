"""4-rank hook test. Delayed producer and NaN-poisoned R expose missing ordering."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist

HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/wyr/greedylore-validation/methods/greedylore-engineer/source')
sys.path.insert(0,str(SOURCE))

class Bucket:
    def __init__(self,buffer,count,rows,cols):
        self.tensor=buffer
        self.views=[buffer[i*rows*cols:(i+1)*rows*cols].view(rows,cols) for i in range(count)]
        self.views.append(buffer[count*rows*cols:])
    def buffer(self):return self.tensor
    def gradients(self):return self.views
    def parameters(self):return self.views
    def index(self):return 0
    def is_last(self):return True

def load(name,poison):
    path=HERE/'variants'/(name+'.py')
    text=path.read_text()
    if poison:
        target='r_memory = torch.empty(total_Rs_size, device=device, dtype=dtype)'
        assert text.count(target)==2
        text=text.replace(target,'r_memory = _poisoned_allocation(total_Rs_size, device, dtype)',1)
    spec=importlib.util.spec_from_loader('diagnostic_'+name,loader=None)
    module=importlib.util.module_from_spec(spec)
    def poisoned(size,device,dtype):
        tensor=torch.full((size,),float('nan'),device=device,dtype=dtype)
        torch.cuda.synchronize(device)
        torch.cuda._sleep(100_000_000)
        return tensor
    module._poisoned_allocation=poisoned
    exec(compile(text,str(path),'exec'),module.__dict__)
    return module

def run_case(name,poison,initial):
    module=load(name,poison)
    buffer=initial.clone()
    count,rows,cols=8,256,688
    bucket=Bucket(buffer,count,rows,cols)
    state=module.SubspaceState(dist.group.WORLD,matrix_approximation_rank=32,
                              update_proj_gap=200,start_compress_iter=0,
                              min_compression_rate=1,use_error_feedback='ef14')
    state.iter=1
    state.param_to_name={p:f'layer.{i}.weight' for i,p in enumerate(bucket.parameters())}
    state.error_dict[0]=torch.zeros_like(buffer)
    state.ms_dict[0]=[torch.eye(rows,device=buffer.device,dtype=buffer.dtype).expand(count,rows,rows).contiguous()]
    dist.barrier();torch.cuda.synchronize()
    future=module.subspace_hook_ef14(state,bucket)
    output=future.wait().clone()
    torch.cuda.synchronize()
    return output,state.error_dict[0].clone()

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required')
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.manual_seed(1243+rank)
    initial=torch.randn(8*256*688+16,device='cuda',dtype=torch.bfloat16)*.01
    reference,reference_error=run_case('serial',True,initial)
    rows=[]
    for name,poison in [('engineer',False),('eventfix',False),('research',False),
                        ('engineer',True),('eventfix',True),('research',True)]:
        actual,error=run_case(name,poison,initial)
        finite=bool(torch.isfinite(actual).all().item())
        same=bool(torch.allclose(actual,reference,rtol=.01,atol=1e-4))
        ef_same=bool(torch.allclose(error,reference_error,rtol=.01,atol=1e-4))
        item={'name':name,'delayed_poisoned_R':poison,'rank':rank,'finite':finite,
              'matches_serial_reference':same,'local_error_matches':ef_same,
              'max_abs_diff':float((actual-reference).abs().max().item()) if finite else None}
        rows.append(item)
        dist.barrier()
    all_rows=[None]*dist.get_world_size()
    dist.all_gather_object(all_rows,rows)
    if rank==0:
        result={'test':'synthetic_hook_stream_ordering','ranks':all_rows,
                'scope':'Not a real-model convergence test; delay and NaN poisoning are test instrumentation only.'}
        (HERE/'hook-race-results.json').write_text(json.dumps(result,indent=2)+'\n')
        for item in rows:print(json.dumps(item),flush=True)
        assert all(r[4]['finite'] and r[4]['matches_serial_reference'] and r[4]['local_error_matches'] for r in all_rows)
        assert any(not r[3]['matches_serial_reference'] for r in all_rows),'Stress case did not reproduce race'
    dist.destroy_process_group()

if __name__=='__main__':main()
