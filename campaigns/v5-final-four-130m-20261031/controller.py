
# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import os,json,subprocess,sys,time
from pathlib import Path
from glval.common import write_json,read_json
from glval.plans import verify_plan
from glval.runner import gpu_lease,run_plan,stop_group
ROOT=Path('/home/wyr/greedylore-validation');C=ROOT/'campaigns/v5-final-four-130m-20261031';FULL=ROOT/'runs/v5-final-four-130m-20261031';SMOKE=ROOT/'runs/v5-four-130m-smoke-20261031'
def status(stage,**kw):write_json(C/'status.json',{'stage':stage,'at':time.time(),**kw})
try:
 plan=verify_plan(FULL)
 with gpu_lease('4,5,6,7',4) as hardware:
  write_json(C/'allocation.json',{'gpus':hardware,'scope':'4-7 only; all checks and full jobs serial'})
  for job in plan['jobs']:
   runtime=FULL/job['runtime']
   for dtype in ['fp32','bf16']:
    status('adapter_equivalence',method=job['method_id'],dtype=dtype)
    out=C/'checks'/job['method_id']/dtype;out.mkdir(parents=True)
    env=os.environ.copy();env.pop('PYTHONHOME',None);env.update(plan['execution']['environment']);cache=ROOT/'cache'
    env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),PYTHONPATH=str(runtime),GLVAL_OUTPUT=str(out),CHECK_DTYPE=dtype,WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1',CUPY_CACHE_DIR=str(cache/'cupy'),CUDA_CACHE_PATH=str(cache/'cuda'),XDG_CACHE_HOME=str(cache))
    cmd=[plan['execution']['python'],'-m','torch.distributed.run','--nnodes','1','--rdzv_backend','c10d','--rdzv_endpoint','127.0.0.1:0','--rdzv_id',job['id']+'-'+dtype,'--nproc_per_node','4',str(C/'check_adapter.py')]
    process=None
    try:
     with (out/'stdout.log').open('w') as so,(out/'stderr.log').open('w') as se:
      process=subprocess.Popen(cmd,cwd=runtime,env=env,stdout=so,stderr=se,start_new_session=True);code=process.wait(timeout=600)
     if code:raise RuntimeError(f'adapter check failed {job["method_id"]}/{dtype}: {code}')
     for rank in range(4):assert read_json(out/f'equivalence-rank-{rank}.json')['passed']
    except BaseException:
     if process:stop_group(process)
     raise
  status('real_130m_smoke',note='8 updates, compression starts2; independent short recipe, no full-quality claim')
  run_plan(SMOKE,'4,5,6,7',True,hardware_allocation=hardware)
  smoke=read_json(SMOKE/'state.json')
  if not all(x['status']=='completed' for x in smoke['jobs'].values()):raise RuntimeError('real130M smoke failed; full queue blocked')
  for job in verify_plan(SMOKE)['jobs']:
   out=SMOKE/job['output']
   for rank in range(4):
    z=read_json(out/f'runtime-rank-{rank}.json');assert z['iter']>=8 and z['hook_calls']>=8
  status('full_training',order=[j['method_id'] for j in plan['jobs']],seed=1243,steps=20000)
  run_plan(FULL,'4,5,6,7',True,hardware_allocation=hardware)
  state=read_json(FULL/'state.json');status(state['status'],jobs=state['jobs'])
except BaseException as e:
 status('failed',error=str(e));raise
