"""Bounded setting sweep, exclusively leases all eight idle GPUs."""
import argparse
import os
from pathlib import Path
import socket
import random
import subprocess
import time
from glval.common import ROOT,read_json,write_json,inventory
from glval.runner import gpu_lease,stop_group

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--only');args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    cases=read_json(HERE/'cases.json')['cases']
    if args.only:cases=[c for c in cases if c['id'] in args.only.split(',')]
    records=[]
    with gpu_lease('0,1,2,3,4,5,6,7',8) as allocation:
        write_json(HERE/'hardware.json',{'gpus':allocation,'reservation':'exclusive sequential four/eight-card comparisons'})
        for index,case in enumerate(cases):
            output=Path(case['output']);runtime=Path(case['runtime'])
            assert inventory(runtime)==case['runtime_files']
            output.mkdir(exist_ok=False)
            hardware=allocation if case['world_size']==8 else allocation[4:]
            env=os.environ.copy();cache=ROOT/'cache'
            env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),PYTHONPATH=str(runtime),
                       GLVAL_OUTPUT=str(output),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',WANDB_MODE='disabled',
                       HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUPY_CACHE_DIR=str(cache/'cupy'),
                       TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'),
                       XDG_CACHE_HOME=str(cache),NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_DEBUG='WARN')
            ephemeral_low,ephemeral_high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
            for attempt in range(100):
                port=random.randrange(15000,30000)
                if ephemeral_low<=port<=ephemeral_high:continue
                try:
                    with socket.socket() as sock:sock.bind(('127.0.0.1',port))
                    break
                except OSError:continue
            else:raise RuntimeError('No free non-ephemeral master port')
            command=['/home/wyr/greedy_lore/.venv/bin/python','-B','-m','torch.distributed.run',
                     '--nnodes','1','--rdzv_backend','static','--master_addr','127.0.0.1',
                     '--master_port',str(port),'--nproc_per_node',str(case['world_size']),
                     str(HERE/'worker.py'),str(output.parent/'case.json'),'--execute']
            started=time.time()
            write_json(HERE/'status.json',{'status':'running','case':case['id'],'index':index+1,'total':len(cases),
                                         'started_at':started,'completed':records})
            child=None
            try:
                with (output/'stdout.log').open('w') as stdout,(output/'stderr.log').open('w') as stderr:
                    child=subprocess.Popen(command,cwd=runtime,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                    code=child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                if child:stop_group(child)
                code='timeout'
            except BaseException:
                if child:stop_group(child)
                raise
            record={'case':case['id'],'returncode':code,'elapsed_seconds':time.time()-started}
            records.append(record)
            write_json(output/'process.json',{**record,'command':command,'gpus':hardware})
            print(record,flush=True)
            # Ordinary method failure is preserved and does not suppress other comparisons.
        write_json(HERE/'status.json',{'status':'finished','completed':records,'total':len(cases)})

if __name__=='__main__':main()
