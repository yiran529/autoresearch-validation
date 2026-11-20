"""Explicitly authorized local continuation. Wait on kernel process handles, never kill/retry."""
import argparse
import fcntl
import multiprocessing
import os
from pathlib import Path
import select
import time

from glval.common import ROOT, read_json, write_json
from glval.plans import verify_plan
from glval.runner import gpu_lease, run_plan
from glval.results import report

HERE = Path(__file__).resolve().parent
OLD = ROOT / 'campaigns/llama60m-engineer-maxu-20261024'

def identity(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text[text.rfind(')')+2:].split()[19]
    except FileNotFoundError:
        return None

def remaining_methods(run, dense):
    plan = verify_plan(run)
    state = read_json(Path(run)/'state.json')
    methods = []
    for job in plan['jobs']:
        mid = job['method_id']
        if (mid == 'dense') != dense or mid in methods:
            continue
        statuses = [state['jobs'][j['id']]['status'] for j in plan['jobs'] if j['method_id'] == mid]
        if 'running' in statuses:
            raise RuntimeError(f'Existing running job after dependency exit: {run}/{mid}')
        if 'planned' in statuses:
            if any(s != 'planned' for s in statuses):
                raise RuntimeError(f'Mixed started/planned repetitions require explicit recovery: {run}/{mid}')
            methods.append(mid)
    return methods

def worker(run, hardware, methods, output):
    try:
        run_plan(run, ','.join(g['uuid'] for g in hardware), True,
                 only_methods=methods, hardware_allocation=hardware)
        state = read_json(Path(run)/'state.json')
        write_json(output, {'status':'finished','jobs':state['jobs']})
    except BaseException as error:
        write_json(output, {'status':'controller_error','error':str(error)})
        raise

def execute_campaign(config, directory):
    context = multiprocessing.get_context('fork')
    def phase(name, tasks):
        write_json(HERE/'status.json', {'status':'running','campaign':str(directory),'phase':name})
        children = []
        for index,(run,hardware,dense) in enumerate(tasks):
            methods = remaining_methods(run, dense)
            if not methods:
                continue
            child = context.Process(target=worker, args=(run,hardware,methods,directory/f'continuation-{name}-{index}.json'))
            child.start()
            children.append(child)
        for child in children:
            child.join()
        if any(c.exitcode != 0 for c in children):
            raise RuntimeError(f'Controller error during {name}; see phase records')
    with gpu_lease(config['training_queues'][0]['gpus'],4) as h0:
        with gpu_lease(config['training_queues'][1]['gpus'],4) as h1:
            phase('pretraining_non_dense',[(q['run'],h,False) for q,h in zip(config['training_queues'],[h0,h1])])
        phase('pretraining_dense_last',[(config['training_queues'][0]['run'],h0,True)])
        phase('timing_non_dense',[(config['timing_run'],h0,False)])
        phase('timing_dense_last',[(config['timing_run'],h0,True)])
    states = {}
    for run in [q['run'] for q in config['training_queues']]+[config['timing_run']]:
        plan = verify_plan(run)
        states[run] = read_json(Path(run)/'state.json')
        write_json(Path(run)/'continuation-report.json',report(run,plan))
    complete = all(j['status']=='completed' for s in states.values() for j in s['jobs'].values())
    write_json(directory/'continuation-result.json',
               {'status':'completed' if complete else 'finished_with_failures','runs':states})

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute',action='store_true')
    args = p.parse_args()
    if not args.execute:
        p.error('Explicit --execute required')
    with (HERE/'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'started.json').exists():
            raise RuntimeError('Continuation already started; no implicit duplicate launch')
        dependencies = read_json(HERE/'dependencies.json')
        handles = []
        for item in dependencies['processes']:
            if identity(item['pid']) != item['start_ticks']:
                continue
            try:
                handle = os.pidfd_open(item['pid'])
            except ProcessLookupError:
                continue
            if identity(item['pid']) == item['start_ticks']:
                handles.append(handle)
            else:
                os.close(handle)
        write_json(HERE/'started.json',{'pid':os.getpid(),'start_ticks':identity(os.getpid()),'time':time.time()})
        write_json(HERE/'status.json',{'status':'waiting_for_60m','dependency_count':len(handles),'processes':dependencies['processes']})
        while handles:
            ready,_,_ = select.select(handles,[],[])
            for handle in ready:
                handles.remove(handle)
                os.close(handle)
        time.sleep(5)  # Let the last process release its CUDA context and utilization sampling settle.
        # The old controller can stop at its failure barrier. Finish only previously authorized, never-started jobs.
        execute_campaign(read_json(OLD/'campaign.json'),OLD)
        time.sleep(5)
        execute_campaign(read_json(HERE/'campaign.json'),HERE)
        result = read_json(HERE/'continuation-result.json')
        write_json(HERE/'status.json',{'status':result['status'],'phase':'finished'})

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_json(HERE/'status.json',{'status':'controller_error','error':str(error)})
        raise
