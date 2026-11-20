"""Bounded four-GPU performance diagnostics; no changes to active full training."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import socket
from glval.common import ROOT,read_json,write_json,inventory
from glval.runner import gpu_lease,stop_group

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--only')
    args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required')
    cases=read_json(HERE/'cases.json')['cases']
    if args.only:cases=[c for c in cases if c['id'] in args.only.split(',')]
    with gpu_lease('4,5,6,7',4) as hardware:
        for case in cases:
            output=Path(case['output']);runtime=Path(case['runtime'])
            assert inventory(runtime)==case['runtime_files']
            output.mkdir(exist_ok=False)
            env=os.environ.copy();cache=ROOT/'cache'
            env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),PYTHONPATH=str(runtime),
                       GLVAL_OUTPUT=str(output),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',WANDB_MODE='disabled',
                       HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUPY_CACHE_DIR=str(cache/'cupy'),
                       TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'),
                       XDG_CACHE_HOME=str(cache),NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_DEBUG='WARN')
            with socket.socket() as port_socket:
                port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1]
            command=['/home/wyr/greedy_lore/.venv/bin/python','-B','-m','torch.distributed.run',
                     '--nnodes','1','--rdzv_backend','static','--master_addr','127.0.0.1',
                     '--master_port',str(port),'--nproc_per_node','4',str(HERE/'worker.py'),
                     str(output.parent/'case.json'),'--execute']
            write_json(HERE/'status.json',{'status':'running','case':case['id'],'started_at':time.time()})
            child=None
            try:
                with (output/'stdout.log').open('w') as stdout,(output/'stderr.log').open('w') as stderr:
                    child=subprocess.Popen(command,cwd=runtime,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                    code=child.wait(timeout=180)
            except BaseException:
                if child:stop_group(child)
                raise
            write_json(output/'process.json',{'returncode':code,'command':command,'gpus':hardware})
            if code!=0:
                write_json(HERE/'status.json',{'status':'failed','case':case['id'],'returncode':code})
                raise RuntimeError('Diagnostic case failed: '+case['id'])
        write_json(HERE/'status.json',{'status':'finished','cases':[c['id'] for c in cases]})

if __name__=='__main__':main()
