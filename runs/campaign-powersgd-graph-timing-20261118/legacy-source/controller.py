"""Execute only the four authorized graph timing cases, using frozen local runtimes."""
import argparse
import csv
import fcntl
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from glval.common import digest,file_hash,inventory,read_json,write_json
from glval.results import inspect_job
from glval.runner import gpu_lease,stop_group


def publish(stage,**details):
    write_json(HERE/'status.json',{'stage':stage,'pid':os.getpid(),'updated_at':time.time(),**details})


def verify(manifest,case):
    body={k:v for k,v in manifest.items() if k!='manifest_hash'}
    if digest(body)!=manifest['manifest_hash']: raise ValueError('Frozen manifest changed')
    if manifest['method_id']!='powersgd-r32-graph-v1' or len(manifest['cases'])!=4:
        raise ValueError('Only four optimized-PowerSGD cases are authorized')
    if any(file_hash(HERE/n)!=h for n,h in manifest['campaign_files'].items()):
        raise ValueError('Campaign code changed after freeze')
    if inventory(case['runtime'])!=case['runtime_files']: raise ValueError('Frozen runtime changed')
    if case['method_id']!=manifest['method_id']: raise ValueError('Unauthorized method')
    if case['recipe']['tracking']['mode']!='disabled' or case['recipe']['checkpoint_policy']!='none':
        raise ValueError('W&B/checkpoint settings changed')


def validate_result(case):
    inspected=inspect_job(HERE,case,case['recipe'])
    out=Path(case['output']); data=read_json(out/'all_results.json')
    if data.get('observed_bucket_counts')!=case['expected_bucket_counts'] or data.get('observed_bucket_bytes')!=case['expected_bucket_bytes']:
        inspected['errors'].append('Observed bucket layout differs from AutoResearch reference')
    if data.get('measured_steps')!=400 or data.get('parameter_count')!=367969280:
        inspected['errors'].append('Wrong measurement budget or model size')
    for rank in range(4):
        evidence=read_json(out/f'runtime-rank-{rank}.json')
        if evidence.get('last_bucket_calls')!=501:
            inspected['errors'].append(f'Rank{rank} did not complete501 backward cycles')
    inspected['status']='verified' if not inspected['errors'] else 'invalid'
    return inspected,data


def summarize(records):
    cells={}
    for precision in ('bf16','fp32'):
        rows=[r for r in records if r['precision']==precision]
        values=[r['mean_iteration_seconds'] for r in rows]
        cells[precision]={'repetitions':rows,'mean_iteration_seconds':statistics.mean(values),
                          'relative_spread':(max(values)-min(values))/statistics.mean(values)}
    write_json(HERE/'summary.json',{'status':'completed','method_id':'powersgd-r32-graph-v1',
        'cells':cells,'scope':'AutoResearch350M AdamW batch1; native passive bucket recording; no contemporaneous Dense baseline.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    options=parser.parse_args()
    if not options.execute: parser.error('Explicit --execute required')
    with (HERE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=read_json(HERE/'manifest.json');records=[]
        review=read_json(HERE/'review.json')
        if review['status']!='approved' or review['manifest_hash']!=manifest['manifest_hash']:
            raise ValueError('Independent review must match the frozen manifest')
        try:
            for case in manifest['cases']:
                verify(manifest,case)
                output=Path(case['output']);runtime=Path(case['runtime'])
                if output.exists():
                    prior=read_json(output/'process.json')
                    if not prior['valid']: raise ValueError('No implicit retry: '+case['id'])
                    checked,_=validate_result(case)
                    if checked['status']!='verified': raise ValueError('Prior result is invalid')
                    records.append(prior);continue
                while True:
                    from contextlib import ExitStack
                    with ExitStack() as resources:
                        try: hardware=resources.enter_context(gpu_lease(manifest['gpus'],4))
                        except (ValueError,BlockingIOError) as error:
                            publish('waiting_for_idle',case=case['id'],gpus=manifest['gpus'],reason=str(error))
                        else:
                            output.mkdir(parents=True,exist_ok=False)
                            topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
                            write_json(output/'hardware.json',{'gpus':hardware,'topology':topology})
                            cache=ROOT/'cache'; env=os.environ.copy()
                            env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),PYTHONPATH=str(runtime),
                                GLVAL_OUTPUT=str(output),PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',WANDB_DIR=str(output),
                                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_EVALUATE_OFFLINE='1',
                                HF_HOME=str(cache/'huggingface'),HF_DATASETS_CACHE=str(cache/'datasets'),
                                XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),
                                CUDA_CACHE_PATH=str(cache/'cuda'),CUPY_CACHE_DIR=str(cache/'cupy'),TOKENIZERS_PARALLELISM='false',
                                OMP_NUM_THREADS='4',NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_DEBUG='INFO')
                            command=[manifest['python'],'-m','torch.distributed.run','--nnodes','1',
                                '--rdzv_backend','c10d','--rdzv_endpoint','127.0.0.1:0','--rdzv_id',case['id'],
                                '--nproc_per_node','4','--module','glval.worker','--driver',case['driver'],'--',*case['args']]
                            publish('running',case=case['id'],precision=case['precision'],repetition=case['repetition'],completed=len(records))
                            print('START',case['id'],flush=True)
                            begin=time.monotonic(); child=None
                            try:
                                with (output/'stdout.log').open('w') as stdout,(output/'stderr.log').open('w') as stderr:
                                    child=subprocess.Popen(command,cwd=runtime,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                                    write_json(HERE/'active-process.json',{'case':case['id'],'pid':child.pid,'command':command})
                                    code=child.wait(timeout=1800)
                            except BaseException:
                                if child:stop_group(child)
                                raise
                            if code: raise RuntimeError('Timing process failed: '+case['id'])
                            validation,data=validate_result(case)
                            record={'case':case['id'],'precision':case['precision'],'repetition':case['repetition'],
                                'returncode':code,'elapsed_seconds':time.monotonic()-begin,
                                'valid':validation['status']=='verified','validation':validation,
                                'mean_iteration_seconds':data['mean_iteration_seconds']}
                            write_json(output/'process.json',record);records.append(record)
                            if not record['valid']: raise RuntimeError('Timing result invalid: '+case['id'])
                            print('COMPLETED',record,flush=True)
                            break
                    time.sleep(30)
            summarize(records);publish('completed',records=records)
        except BaseException as error:
            publish('stopped',error=str(error),records=records);raise


if __name__=='__main__':main()
