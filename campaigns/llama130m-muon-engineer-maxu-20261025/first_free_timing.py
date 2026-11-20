"""Authorized timing on the first completed four-GPU group, retaining live training."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import argparse
from contextlib import contextmanager, ExitStack
import csv
import fcntl
import io
import os
from pathlib import Path
import select
import signal
import subprocess
import threading
import time

from glval.common import ROOT, read_json, write_json
from glval.plans import verify_plan
from glval.runner import query_gpus, run_plan
from glval.results import report, inspect_job

HERE = Path(__file__).resolve().parent
STATUS = HERE/'first-free-timing-status.json'

def proc(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    parts = text[text.rfind(')')+2:].split()
    return {'state':parts[0],'ppid':int(parts[1]),'ticks':parts[19]}

def open_handle(pid,ticks):
    current = proc(pid)
    if current is None or current['ticks'] != ticks:
        return None
    try: handle = os.pidfd_open(pid)
    except ProcessLookupError: return None
    current = proc(pid)
    if current is None or current['ticks'] != ticks:
        os.close(handle)
        return None
    return handle

def wait_first(handles):
    # A missing handle means the exact tracked process already exited.
    if any(h is None for h in handles):
        return next(i for i,h in enumerate(handles) if h is None)
    ready,_,_ = select.select(handles,[],[])
    return next(i for i,h in enumerate(handles) if h in ready)

def check_idle(hardware):
    uuids = {g['uuid'] for g in hardware}
    current = {g['uuid']:g for g in query_gpus()}
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'],text=True)
    if any(row and row[0].strip() in uuids for row in csv.reader(io.StringIO(active))):
        raise RuntimeError('Selected completed group still has a compute process')
    if len(uuids)!=4 or any(int(current[u]['memory_mib'])>512 or int(current[u]['utilization'])>5 for u in uuids):
        raise RuntimeError('Selected group is not idle')

@contextmanager
def campaign_lease(hardware,previous_owner):
    """Retain the existing campaign reservation until its inherited locks transfer."""
    with ExitStack() as stack:
        files=[]
        pending=[]
        for gpu in sorted(hardware,key=lambda g:g['uuid']):
            path=ROOT/'locks'/(gpu['uuid']+'.lock')
            file=stack.enter_context(path.open('a'))
            files.append(file)
            try: fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                owner=proc(previous_owner['pid'])
                if owner is None or owner['ticks']!=previous_owner['ticks']:
                    # The known owner just exited; wait for its inherited reservation to close.
                    fcntl.flock(file,fcntl.LOCK_EX)
                else:
                    descriptors=Path(f"/proc/{previous_owner['pid']}/fd")
                    owned=set()
                    for fd in descriptors.iterdir():
                        try: owned.add(os.readlink(fd))
                        except FileNotFoundError: pass
                    if str(path) not in owned:
                        raise RuntimeError('GPU lock belongs to another allocation')
                    pending.append(file)
        errors=[]
        acquired=threading.Event()
        def transfer():
            try:
                for file in pending: fcntl.flock(file,fcntl.LOCK_EX)
            except BaseException as error: errors.append(error)
            finally: acquired.set()
        thread=threading.Thread(target=transfer,daemon=True)
        thread.start()
        # Until transfer, the verified live N20 worker holds the same campaign's locks.
        yield {'inherited_lock_bridge':bool(pending),'previous_owner':previous_owner}
        acquired.wait()
        thread.join()
        if errors: raise errors[0]

def run_timing(config,hardware,metadata):
    run=Path(config['timing_run'])
    plan=verify_plan(run)
    methods=list(dict.fromkeys(j['method_id'] for j in plan['jobs'] if j['method_id']!='dense'))
    for phase,selected in [('timing_non_dense',methods),('timing_dense_last',['dense'])]:
        write_json(STATUS,{'status':'running','phase':phase,'active_timing_run':str(run),**metadata})
        run_plan(run,','.join(g['uuid'] for g in hardware),True,
                 only_methods=selected,hardware_allocation=hardware)

def finalize_orphan(run,job_id):
    """Verify saved evidence without inventing the lost parent's process exit code."""
    run=Path(run)
    plan=verify_plan(run)
    state=read_json(run/'state.json')
    item=state['jobs'][job_id]
    if item['status']!='running': return
    job=next(j for j in plan['jobs'] if j['id']==job_id)
    inspected=inspect_job(run,job,plan['recipe'])
    item.update(status='finished_exit_unknown' if inspected['status']=='verified' else 'failed',
                returncode=None,process_exit_verified=False,validation=inspected,
                completion_observed_at=time.time(),
                elapsed_seconds=None,
                monitor_note='Original monitor exited during scheduler handoff. Exact torchrun process exit observed via pidfd; exit code unavailable. Artifact verification is separate.')
    state['status']='finished_exit_unknown' if all(j['status'] not in {'running','planned'} for j in state['jobs'].values()) else 'partial'
    write_json(run/'state.json',state)

def resume_handoff():
    config=read_json(HERE/'campaign.json')
    previous=read_json(HERE/'first-free-timing-started.json')
    assert proc(previous['replaced_controller']['pid']) is None, 'Old coordinator still alive'
    q0=Path(config['training_queues'][0]['run'])
    q1=Path(config['training_queues'][1]['run'])
    timing_runs=[config['timing_run'],*config.get('additional_timing_runs',[])]
    for run in [q0,q1,*map(Path,timing_runs)]: verify_plan(run)
    for run in timing_runs:
        assert all(j['status']=='planned' for j in read_json(Path(run)/'state.json')['jobs'].values())
    dense_pid=read_json(HERE/'advance-dense.json')['supervisor_pid']
    dense_info=proc(dense_pid)
    n20=read_json(q1/'state.json')['jobs']['n020-engineer-maxu-s1243']
    assert dense_info and n20['status']=='running'
    train_info=proc(n20['pid'])
    assert train_info, 'N20 exited before handoff; inspect evidence explicitly'
    groups=[{'pid':dense_pid,'ticks':dense_info['ticks'],'queue':str(q0)},
            {'pid':n20['pid'],'ticks':train_info['ticks'],'queue':str(q1)}]
    handles=[open_handle(g['pid'],g['ticks']) for g in groups]
    assert all(h is not None for h in handles)
    with ExitStack() as locks:
        # Previous CPU worker has exited; reserve both existing allocations without touching training.
        hardware_groups=[read_json(run/'hardware.json')['gpus'] for run in [q0,q1]]
        for gpu in sorted([g for h in hardware_groups for g in h],key=lambda g:g['uuid']):
            file=locks.enter_context((ROOT/'locks'/(gpu['uuid']+'.lock')).open('a'))
            fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(HERE/'first-free-timing-resumed.json',{'pid':os.getpid(),'groups':groups,'time':time.time(),
                   'all_eight_gpu_locks_reacquired':True,'previous_monitor_exit_recorded':True})
        write_json(STATUS,{'status':'waiting_first_group','groups':groups,'all_eight_gpu_locks_held':True})
        first=wait_first(handles)
        if first==1: finalize_orphan(q1,'n020-engineer-maxu-s1243')
        time.sleep(5)
        state=read_json(Path(groups[first]['queue'])/'state.json')
        assert all(j['status'] in {'completed','failed','finished_exit_unknown'} for j in state['jobs'].values()), state
        hardware=hardware_groups[first]
        check_idle(hardware)
        metadata={'selected_group':first,'gpus':[g['index'] for g in hardware],
                  'gpu_uuids':[g['uuid'] for g in hardware],
                  'concurrent_pretraining_possible':True,
                  'comparison_note':'Timing may overlap pretraining on the other four GPUs; shared host/CPU/IO interference is possible.'}
        write_json(HERE/'first-free-timing-allocation.json',metadata)
        for timing_run in timing_runs:
            run_timing({**config,'timing_run':timing_run},hardware,metadata)
        write_json(STATUS,{'status':'timing_finished_waiting_full_training',**metadata})
        for handle in handles:
            select.select([handle],[],[])
            os.close(handle)
        finalize_orphan(q1,'n020-engineer-maxu-s1243')
        states={}
        for run in [str(q0),str(q1),*timing_runs]:
            plan=verify_plan(run)
            states[run]=read_json(Path(run)/'state.json')
            result=report(run,plan)
            if run==str(q1):
                for item in result['jobs']:
                    if item['id']=='n020-engineer-maxu-s1243' and states[run]['jobs'][item['id']]['status']=='finished_exit_unknown':
                        item['status']='artifact_verified_exit_unknown'
                        item['process_exit_verified']=False
            write_json(Path(run)/'continuation-report.json',result)
        complete=all(j['status']=='completed' for s in states.values() for j in s['jobs'].values())
        status='completed' if complete else 'finished_with_failures_or_unverified_exit'
        write_json(HERE/'continuation-result.json',{'status':status,'runs':states,'timing_allocation':metadata})
        write_json(STATUS,{'status':status,'phase':'finished',**metadata})
        write_json(HERE/'status.json',{'status':status,'phase':'finished','timing_status_file':str(STATUS)})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume-handoff',action='store_true')
    parser.add_argument('--replace-waiter',action='store_true')
    args=parser.parse_args()
    if not args.execute: parser.error('Explicit --execute required')
    if args.replace_waiter:
        assert args.resume_handoff
        with (HERE/'replace-waiter.lock').open('a') as replacement:
            fcntl.flock(replacement,fcntl.LOCK_EX|fcntl.LOCK_NB)
            previous=read_json(HERE/'first-free-timing-resumed.json')
            pid=previous['pid']
            current=proc(pid)
            assert current and b'first_free_timing.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
            handle=open_handle(pid,current['ticks'])
            assert handle is not None
            paused=False
            try:
                signal.pidfd_send_signal(handle,signal.SIGSTOP)
                paused=True
                for _ in range(100):
                    if proc(pid)['state']=='T': break
                    time.sleep(.01)
                else: raise RuntimeError('Waiting coordinator did not pause')
                assert read_json(STATUS)['status']=='waiting_first_group', 'Cannot replace an active timing worker'
                signal.pidfd_send_signal(handle,signal.SIGTERM)
                signal.pidfd_send_signal(handle,signal.SIGCONT)
                paused=False
                if not select.select([handle],[],[],10)[0]: raise RuntimeError('Waiting coordinator did not exit')
                (HERE/'first-free-timing-resumed.json').rename(HERE/f'first-free-timing-resumed-{pid}.json')
            finally:
                if paused: signal.pidfd_send_signal(handle,signal.SIGCONT)
                os.close(handle)
            with (HERE/'first-free-timing.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return resume_handoff()
    with (HERE/'first-free-timing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume_handoff:
            assert not (HERE/'first-free-timing-resumed.json').exists(), 'Already resumed'
            return resume_handoff()
        if (HERE/'first-free-timing-started.json').exists():
            raise RuntimeError('Already dispatched; no implicit restart')
        config=read_json(HERE/'campaign.json')
        for run in [q['run'] for q in config['training_queues']]+[config['timing_run']]: verify_plan(run)
        timing_state=read_json(Path(config['timing_run'])/'state.json')
        assert all(j['status']=='planned' for j in timing_state['jobs'].values())
        controller=read_json(HERE/'started.json')
        old=proc(controller['pid'])
        assert old and old['ticks']==controller['start_ticks'] and old['state']=='T'
        oldfd=open_handle(controller['pid'],controller['start_ticks'])
        advance=read_json(HERE/'advance-dense.json')
        dense_pid=advance['supervisor_pid']
        dense_info=proc(dense_pid)
        assert dense_info, 'Dense supervisor must still be running during handoff'
        q1=Path(config['training_queues'][1]['run'])
        state=read_json(q1/'state.json')
        n20=state['jobs']['n020-engineer-maxu-s1243']
        assert n20['status']=='running'
        train=proc(n20['pid'])
        assert train
        worker_pid=train['ppid']
        worker_info=proc(worker_pid)
        assert worker_info and worker_info['ppid']==controller['pid']
        groups=[{'pid':dense_pid,'ticks':dense_info['ticks'],'queue':config['training_queues'][0]['run']},
                {'pid':worker_pid,'ticks':worker_info['ticks'],'queue':str(q1)}]
        handles=[open_handle(g['pid'],g['ticks']) for g in groups]
        assert all(h is not None for h in handles)
        # Only terminate the stopped, idle coordination parent. Training and Dense supervisors remain untouched.
        signal.pidfd_send_signal(oldfd,signal.SIGTERM)
        signal.pidfd_send_signal(oldfd,signal.SIGCONT)
        if not select.select([oldfd],[],[],10)[0]: raise RuntimeError('Old coordinator did not exit')
        os.close(oldfd)
        write_json(HERE/'first-free-timing-started.json',{'pid':os.getpid(),'time':time.time(),
                    'groups':groups,'replaced_controller':controller,'policy':'First completed group; all timing on same four GPUs; Dense timing last; no training restarts.'})
        write_json(STATUS,{'status':'waiting_first_group','groups':groups})
        first=wait_first(handles)
        time.sleep(5)
        state=read_json(Path(groups[first]['queue'])/'state.json')
        assert all(j['status'] in {'completed','failed'} for j in state['jobs'].values()), state
        hardware=read_json(Path(groups[first]['queue'])/'hardware.json')['gpus']
        check_idle(hardware)
        metadata={'selected_group':first,'gpus':[g['index'] for g in hardware],
                  'gpu_uuids':[g['uuid'] for g in hardware],
                  'concurrent_pretraining_possible':True,
                  'comparison_note':'Timing may overlap the other four-GPU pretraining workload; shared host/CPU/IO interference is possible.'}
        with campaign_lease(hardware,{'pid':worker_pid,'ticks':worker_info['ticks']}) as lease:
            metadata.update(lease)
            write_json(HERE/'first-free-timing-allocation.json',metadata)
            run_timing(config,hardware,metadata)
        write_json(STATUS,{'status':'timing_finished_waiting_full_training',**metadata})
        for handle in handles:
            select.select([handle],[],[])
            os.close(handle)
        states={}
        for run in [q['run'] for q in config['training_queues']]+[config['timing_run']]:
            plan=verify_plan(run)
            states[run]=read_json(Path(run)/'state.json')
            write_json(Path(run)/'continuation-report.json',report(run,plan))
        complete=all(j['status']=='completed' for s in states.values() for j in s['jobs'].values())
        status='completed' if complete else 'finished_with_failures'
        write_json(HERE/'continuation-result.json',{'status':status,'runs':states,'timing_allocation':metadata})
        write_json(STATUS,{'status':status,'phase':'finished',**metadata})
        write_json(HERE/'status.json',{'status':status,'phase':'finished','timing_status_file':str(STATUS)})

if __name__=='__main__':
    try: main()
    except BaseException as error:
        write_json(STATUS,{'status':'controller_error','error':str(error),'time':time.time()})
        raise
