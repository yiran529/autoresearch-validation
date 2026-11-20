import argparse
import os
from pathlib import Path
import subprocess
from glval.common import write_json
from glval.runner import stop_group
from allocation import reserve

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    if not p.parse_args().execute:p.error('Explicit --execute required')
    hardware,lease=reserve()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),
               PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',WANDB_MODE='disabled',
               NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_DEBUG='WARN')
    command=['/home/wyr/greedy_lore/.venv/bin/python','-B','-m','torch.distributed.run',
             '--nnodes','1','--rdzv_backend','c10d','--rdzv_endpoint','127.0.0.1:0',
             '--rdzv_id','greedylore-nan-hooktest','--nproc_per_node','4',str(HERE/'hook_race_test.py'),'--execute']
    child=None
    try:
        with (HERE/'hook-test.stdout.log').open('w') as out,(HERE/'hook-test.stderr.log').open('w') as err:
            child=subprocess.Popen(command,env=env,stdout=out,stderr=err,start_new_session=True)
            code=child.wait(timeout=240)
    except BaseException:
        if child:stop_group(child)
        raise
    write_json(HERE/'hook-test-process.json',{'returncode':code,'gpus':hardware,'lease':lease,'command':command})
    print({'returncode':code,'results':str(HERE/'hook-race-results.json')},flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
