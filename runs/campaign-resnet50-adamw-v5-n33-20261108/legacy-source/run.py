"""Four dedicated GPUs only: check all adapters, then train five methods serially."""
import argparse
import csv
import io
import os
from pathlib import Path
import random
import socket
import subprocess
import time
from glval.common import ROOT,read_json,write_json,inventory,file_hash
from glval.runner import gpu_lease,stop_group,query_gpus
from glval.results import inspect_job

HERE=Path(__file__).resolve().parent
PYTHON='/home/wyr/greedy_lore/.venv/bin/python'

def master_port():
    low,high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    for _ in range(100):
        port=random.randrange(15000,30000)
        if low<=port<=high:continue
        try:
            with socket.socket() as sock:sock.bind(('127.0.0.1',port))
            return port
        except OSError:continue
    raise RuntimeError('No free non-ephemeral master port')

def assert_no_foreign_gpu_process(hardware):
    rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
    selected={g['uuid'] for g in hardware}
    if any(row and row[0].strip() in selected for row in rows):
        raise RuntimeError('Selected GPUs have a compute process; queue stopped without interrupting it')

SELECTION = '4,5,6,7'

def publish(stage, **details):
    write_json(HERE / "status.json", {
        "stage": stage, "pid": os.getpid(), "updated_at": time.time(), **details
    })

def wait_for_idle(method):
    idle_since = None
    while True:
        selected = [gpu for gpu in query_gpus() if gpu["index"] in SELECTION.split(",")]
        if len(selected) != 4:
            raise RuntimeError("Cannot resolve all four assigned GPUs")
        selected_ids = {gpu["uuid"] for gpu in selected}
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"
        ], text=True)
        busy_process = any(
            row and row[0].strip() in selected_ids
            for row in csv.reader(io.StringIO(output))
        )
        idle = not busy_process and all(
            int(gpu["memory_mib"]) <= 512 and int(gpu["utilization"]) <= 5
            for gpu in selected
        )
        now = time.monotonic()
        if idle:
            if idle_since is None:
                idle_since = now
        else:
            idle_since = None
        elapsed = now - idle_since if idle_since is not None else 0
        publish("waiting_for_idle", method=method, idle_seconds=round(elapsed, 1),
                required_idle_seconds=600, poll_seconds=10, gpus=selected)
        if elapsed >= 600:
            return
        time.sleep(10)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--gpus',required=True)
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    manifest=read_json(HERE/'cases.json');records=[]
    assert args.gpus == SELECTION, 'Use only the assigned four GPUs'
    for case in manifest['cases']:
        wait_for_idle(case['id'])
        with gpu_lease(args.gpus,4) as hardware:
            write_json(HERE/'hardware.json', {'gpus': hardware, 'selection': args.gpus,
                'scope': 'One case per lease, after ten minutes idle'})
            if case['phase']=='full' and any(r['phase']=='synthetic_check' and not r['valid'] for r in records):
                write_json(HERE/'status.json',{'status':'checks_failed','records':records,'full_training_started':False})
                raise RuntimeError('Synthetic adapter check failed; full training not started')
            runtime=Path(case['runtime']);output=Path(case['output'])
            assert inventory(runtime)==case['runtime_files'],'Frozen runtime modified'
            assert_no_foreign_gpu_process(hardware)
            output.mkdir(exist_ok=False)
            cache=ROOT/'cache';tracking=case['recipe']['tracking']
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),PYTHONPATH=str(runtime),
                GLVAL_OUTPUT=str(output),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
                WANDB_MODE=tracking['mode'],WANDB_PROJECT=tracking['project'],WANDB_RUN_GROUP=tracking['group'],
                WANDB_NAME='resnet50-adamw-'+case['method_id']+'-s1243',WANDB_RUN_ID=file_hash(output.parent/'case.json')[:16],
                WANDB_RESUME='never',WANDB_DIR=str(output),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                CUPY_CACHE_DIR=str(cache/'cupy'),TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),
                CUDA_CACHE_PATH=str(cache/'cuda'),XDG_CACHE_HOME=str(cache),TMPDIR=str(cache/'tmp'),
                NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_DEBUG='WARN',
                HTTP_PROXY='http://127.0.0.1:28731',HTTPS_PROXY='http://127.0.0.1:28731',
                http_proxy='http://127.0.0.1:28731',https_proxy='http://127.0.0.1:28731',
                NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1')
            command=[PYTHON,'-B','-m','torch.distributed.run','--nnodes','1','--rdzv_backend','static',
                '--master_addr','127.0.0.1','--master_port',str(master_port()),'--nproc_per_node','4',
                '--module','glval.worker','--driver',case['driver'],'--',*case['args']]
            started=time.time();child=None
            write_json(HERE/'status.json',{'status':'running','phase':case['phase'],'case':case['id'],
                'started_at':started,'records':records,'selection':args.gpus,'full_training_started':case['phase']=='full'})
            try:
                with (output/'stdout.log').open('w') as stdout,(output/'stderr.log').open('w') as stderr:
                    child=subprocess.Popen(command,cwd=runtime,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                    write_json(HERE/'active-process.json',{'case':case['id'],'pid':child.pid,'started_at':started,'phase':case['phase']})
                    code=child.wait(timeout=900 if case['phase']=='synthetic_check' else 21600)
            except BaseException as error:
                if child:stop_group(child)
                write_json(HERE/'status.json',{'status':'interrupted','case':case['id'],'error':str(error),'records':records})
                raise
            job={'id':case['id'],'method_id':case['method_id'],'output':case['output'],'runtime_files':case['runtime_files']}
            validation=inspect_job(HERE,job,case['recipe'])
            config_path=output/'config.json'
            if config_path.exists():
                actual=read_json(config_path)
                if actual.get('optimizer_class')!='torch.optim.AdamW' or actual.get('optimizer')!='adamw' or actual.get('lr')!=case['recipe']['args']['lr'] or actual.get('optimizer_betas')!=[0.9,0.999] or actual.get('optimizer_eps')!=1e-8 or actual.get('weight_decay')!=0.0005 or actual.get('model')!='ResNet50' or actual.get('parameter_count')!=23520842 or actual.get('resnet_blocks')!=[3,4,6,3] or actual.get('resnet_block_type')!='Bottleneck':
                    validation['status']='invalid';validation['errors'].append('ResNet50 architecture/parameter identity mismatch')
            else:
                validation['status']='invalid';validation['errors'].append('Missing model config')
            if validation['status']=='verified':
                for rank in range(4):
                    evidence=read_json(output/f'runtime-rank-{rank}.json')
                    if evidence['last_bucket_calls']!=case['recipe']['completion']['update_step']:
                        validation['status']='invalid';validation['errors'].append('Hook update budget mismatch')
                    if case['method_id'] == 'v5-n033-research-resnet-v1':
                        first = case['recipe']['compression_start_step'] + 1
                        last = case['recipe']['completion']['update_step']
                        if evidence.get('timing_refresh_steps') != list(range(first, last + 1)):
                            validation['status'] = 'invalid'
                            validation['errors'].append('N33 compression steps do not match budget')
                metrics=read_json(output/'all_results.json')
                if case['phase']=='full' and len((output/'metrics.jsonl').read_text().splitlines())!=40:
                    validation['status']='invalid';validation['errors'].append('Expected40epoch rows')
            record={'case':case['id'],'method_id':case['method_id'],'phase':case['phase'],'returncode':code,
                'elapsed_seconds':time.time()-started,'valid':code==0 and validation['status']=='verified','validation':validation}
            write_json(output/'process.json',{**record,'command':command,'gpus':hardware})
            records.append(record);print(record,flush=True)
            time.sleep(1)
    write_json(HERE/'status.json',{'status':'completed' if all(r['valid'] for r in records) else 'finished_with_failures',
            'records':records,'full_training_started':True,'selection':args.gpus})

if __name__=='__main__':main()
