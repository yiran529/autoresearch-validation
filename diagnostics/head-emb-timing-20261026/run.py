"""Bounded setting sweep, exclusively leases all eight idle GPUs."""
import argparse
import os
from pathlib import Path
import socket
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
    manifest=read_json(HERE/'cases.json')
    from glval.common import file_hash
    assert file_hash(HERE/'worker.py')==manifest['worker_hash']
    assert file_hash(HERE/'run.py')==manifest['runner_hash']
    with gpu_lease('4,5,6,7',4) as allocation:
        write_json(HERE/'hardware.json',{'gpus':allocation,'reservation':'exclusive sequential four-card head/embedding comparisons'})
        for index,case in enumerate(cases):
            output=Path(case['output']);runtime=Path(case['runtime'])
            assert inventory(runtime)==case['runtime_files']
            output.mkdir(exist_ok=False)
            hardware=allocation
            env=os.environ.copy();cache=ROOT/'cache'
            env.update(TMPDIR=str(cache/'tmp'),CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),PYTHONPATH=str(runtime),
                       GLVAL_OUTPUT=str(output),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',WANDB_MODE='disabled',
                       HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUPY_CACHE_DIR=str(cache/'cupy'),
                       TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'),
                       XDG_CACHE_HOME=str(cache),NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_DEBUG='WARN')
            with socket.socket() as sock:
                import random
                port=random.randrange(15000,28000)
                sock.bind(('127.0.0.1',port))
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
            valid=False;errors=[]
            if code==0:
                try:
                    result=read_json(output/'result.json')
                    assert len(result['ranks'])==4
                    for rank in range(4):
                        evidence=read_json(output/f'runtime-rank-{rank}.json')
                        assert evidence['method_id']==case['method']
                        assert evidence['last_bucket_calls']==case['warmup']+case['measured']
                        assert evidence['hook_sha256']==case['runtime_files'][evidence['hook_source']]
                    valid=True
                except Exception as error:errors.append(str(error))
            record={'valid':valid,'errors':errors,'case':case['id'],'returncode':code,'elapsed_seconds':time.time()-started}
            records.append(record)
            write_json(output/'process.json',{**record,'command':command,'gpus':hardware})
            print(record,flush=True)
            # Ordinary method failure is preserved and does not suppress other comparisons.
        write_json(HERE/'status.json',{'status':'completed' if all(r['valid'] for r in records) else 'finished_with_failures','completed':records,'total':len(cases)})

if __name__=='__main__':main()
