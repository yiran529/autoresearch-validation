"""Run bounded diagnostic cases in the background; never poll through agent turns."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from allocation import reserve
from glval.common import ROOT,read_json,write_json,inventory
from glval.runner import stop_group

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--phase',choices=['early','formal'],required=True)
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    hardware,lease=reserve()
    manifest=read_json(HERE/'model-checks.json')
    cases=[c for c in manifest['checks'] if ('early' in c['id'])==(args.phase=='early')]
    env=os.environ.copy()
    cache=ROOT/'cache'
    env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),
               PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',WANDB_MODE='disabled',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
               HF_HOME=str(cache/'huggingface'),HF_DATASETS_CACHE=str(cache/'datasets'),
               XDG_CACHE_HOME=str(cache),TORCH_HOME=str(cache/'torch'),
               TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'),
               CUPY_CACHE_DIR=str(cache/'cupy'),NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_DEBUG='WARN')
    outcomes=[]
    for case in cases:
        runtime=Path(case['runtime']);output=Path(case['output'])
        assert inventory(runtime)==case['runtime_files'],'Diagnostic runtime changed'
        output.mkdir(exist_ok=False)
        env.update(PYTHONPATH=str(runtime),GLVAL_OUTPUT=str(output),
                   GLVAL_TOKENIZER_PATH=read_json(runtime/'recipe.json')['tokenizer_path'])
        child=None;started=time.time()
        write_json(HERE/('model-'+args.phase+'-status.json'),{'status':'running','check':case['id'],'started_at':started})
        try:
            with (output/'stdout.log').open('w') as out,(output/'stderr.log').open('w') as err:
                child=subprocess.Popen(case['command'],cwd=runtime,env=env,stdout=out,stderr=err,start_new_session=True)
                write_json(output/'process.json',{'pid':child.pid,'started_at':started,'gpus':hardware,'lease':lease})
                code=child.wait(timeout=3600)
        except BaseException:
            if child:stop_group(child)
            raise
        ranks=[read_json(output/f'diagnostic-result-rank-{i}.json') for i in range(4) if (output/f'diagnostic-result-rank-{i}.json').exists()]
        verified=code==0 and len(ranks)==4 and all(r['completed_steps']==case['target_steps'] and r['last_loss_finite'] for r in ranks)
        outcome={'id':case['id'],'returncode':code,'status':'diagnostic_pass' if verified else 'diagnostic_failed',
                 'elapsed_seconds':time.time()-started,'ranks':ranks,'output':str(output)}
        write_json(output/'process-result.json',outcome);outcomes.append(outcome)
        if 'fixed' in case['id'] and not verified:break
    result={'status':'finished','scope':'Short diagnostics, not full pretraining','checks':outcomes}
    write_json(HERE/('model-'+args.phase+'-results.json'),result)
    write_json(HERE/('model-'+args.phase+'-status.json'),{'status':'finished','checks':[{k:c[k] for k in ['id','status','returncode']} for c in outcomes]})
    print(result,flush=True)

if __name__=='__main__':main()
