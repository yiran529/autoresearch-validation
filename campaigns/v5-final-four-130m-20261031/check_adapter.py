
# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import argparse,copy,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from glval.bridge import register_comm_hook_for_ddp_model as attach,finish_runtime
from comm_hooks.utils import add_comm_hook_args,register_comm_hook_for_ddp_model as direct
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');torch.manual_seed(1243)
root=Path.cwd();recipe=json.loads((root/'recipe.json').read_text());method=json.loads((root/'method.json').read_text())
p=argparse.ArgumentParser(add_help=False);add_comm_hook_args(p);args=p.parse_args([])
for k,v in recipe['args'].items():setattr(args,k,v)
args.seed=1243;args.start_compress_iter=recipe['compression_start_step']
for k,v in method['parameters'].items():setattr(args,k,v)
dtype=torch.bfloat16 if os.environ['CHECK_DTYPE']=='bf16' else torch.float32
net=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,16)).cuda().to(dtype)
a=nn.parallel.DistributedDataParallel(net,device_ids=[rank]);other=copy.deepcopy(net)
b=nn.parallel.DistributedDataParallel(other,device_ids=[rank])
oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
sa=attach(a,dist.group.WORLD,args,oa)
sb=direct(b,dist.group.WORLD,args,optimizer=ob)
def exact(x,y):
 if torch.is_tensor(x):assert torch.is_tensor(y) and torch.equal(x.reshape(-1).view(torch.uint8),y.reshape(-1).view(torch.uint8))
 elif isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in x:exact(x[k],y[k])
 elif isinstance(x,(list,tuple)):
  assert len(x)==len(y)
  for u,v in zip(x,y):exact(u,v)
 else:assert x==y
for step in range(1010):
 torch.manual_seed(90000+rank*2000+step);x=torch.randn(4,128,device='cuda',dtype=dtype);target=torch.randn(4,16,device='cuda',dtype=dtype)
 for m,o in [(a,oa),(b,ob)]:
  o.zero_grad(set_to_none=True);loss=(m(x).float()-target.float()).square().mean();assert torch.isfinite(loss);loss.backward();o.step()
 if step>=999:
  for u,v in zip(a.parameters(),b.parameters()):exact(u.data,v.data);exact(u.grad,v.grad)
  exact(oa.state_dict(),ob.state_dict())
  for key in ['residuals','predictors','bucket_sizes']: 
   if hasattr(sa,key):exact(getattr(sa,key),getattr(sb,key))
assert sa.iter==sb.iter==1010 and sa.residuals
finish_runtime();torch.cuda.synchronize()
out=Path(os.environ['GLVAL_OUTPUT']);(out/f'equivalence-rank-{rank}.json').write_text(json.dumps({'passed':True,'dtype':str(dtype),'updates':1010,'dense_prefix':1000,'compressed_updates':10,'gradient_parameters_adamw_EF_predictor':'bit_exact_adapter_vs_direct','scope':'Adapter identity and prefix integration; not independent algorithm correctness or full Muon learning evidence'}))
dist.destroy_process_group()
