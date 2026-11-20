"""One independent supervisor per model; wait on exact dependency PID handles."""
import argparse
import fcntl
import os
from pathlib import Path
import select
import sys
import time

from glval.common import ROOT,read_json,write_json
from glval.plans import verify_plan
from glval.runner import run_plan

HERE=Path(__file__).resolve().parent

def start_ticks(pid):
    try:text=Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:return None
    return text[text.rfind(')')+2:].split()[19]

def wait_for_dependency(dependency):
    pid=dependency['pid']
    if start_ticks(pid)==dependency['start_ticks']:
        try:handle=os.pidfd_open(pid)
        except ProcessLookupError:handle=None
        if handle is not None:
            if start_ticks(pid)==dependency['start_ticks']:select.select([handle],[],[])
            os.close(handle)
    for run in dependency['runs']:
        verify_plan(run)
        jobs=read_json(Path(run)/'state.json')['jobs']
        if any(j['status'] in {'planned','running'} for j in jobs.values()):
            raise RuntimeError('Timing controller exited with pending/running jobs; inspect before training')
    time.sleep(5)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['60m','130m'],required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    model=args.model
    status=HERE/(model+'-status.json')
    with (HERE/(model+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/(model+'-started.json')).exists():raise RuntimeError('Supervisor already started; no retries')
        config=read_json(HERE/'campaign.json');run=config['runs'][model]
        verify_plan(run)
        write_json(HERE/(model+'-started.json'),{'pid':os.getpid(),'start_ticks':start_ticks(os.getpid()),'time':time.time()})
        if model=='60m':
            write_json(status,{'status':'waiting_for_timing','run':run,'dependency':config['timing_dependency']})
            wait_for_dependency(config['timing_dependency'])
            hardware=None
        else:
            # This campaign's timing is fixed on4-7; borrow its idle0-3 reservation until those locks transfer.
            sys.path.insert(0,str(ROOT/'diagnostics/greedylore-nan-20261025'))
            from allocation import reserve
            hardware,lease=reserve()
            write_json(HERE/'130m-allocation.json',{'gpus':hardware,'lease':lease})
        write_json(status,{'status':'running','run':run,'gpus':config['allocations'][model]})
        result=run_plan(run,config['allocations'][model],True,hardware_allocation=hardware)
        state=read_json(Path(run)/'state.json')
        write_json(status,{'status':state['status'],'run':run,'report':result})

if __name__=='__main__':
    try:main()
    except BaseException as error:
        # Keep the model-specific error record without guessing a successful training outcome.
        import json
        if '--model' in sys.argv:
            model=sys.argv[sys.argv.index('--model')+1]
            write_json(HERE/(model+'-status.json'),{'status':'controller_error','error':str(error),'time':time.time()})
        print(json.dumps({'controller_error':str(error)}),flush=True)
        raise
