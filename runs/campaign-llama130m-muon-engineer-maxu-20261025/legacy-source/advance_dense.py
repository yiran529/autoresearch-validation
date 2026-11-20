"""Run the existing Dense job early on the idle allocation; preserve the original coordinator."""
import argparse
import csv
import fcntl
import io
import os
from pathlib import Path
import select
import signal
import subprocess
import time

from glval.common import ROOT, read_json, write_json
from glval.plans import verify_plan
from glval.runner import query_gpus, run_plan

HERE = Path(__file__).resolve().parent

def start_ticks(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    return text[text.rfind(')')+2:].split()[19]

def require_idle(hardware):
    uuids = {g['uuid'] for g in hardware}
    current = {g['uuid']:g for g in query_gpus()}
    if len(uuids) != 4 or {current[u]['index'] for u in uuids} != {'0','1','2','3'}:
        raise RuntimeError('Expected the original GPUs0-3 allocation')
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'],text=True)
    if any(row and row[0].strip() in uuids for row in csv.reader(io.StringIO(active))):
        raise RuntimeError('GPU allocation has an active compute process')
    if any(int(current[u]['memory_mib']) > 512 or int(current[u]['utilization']) > 5 for u in uuids):
        raise RuntimeError('GPU allocation is not idle')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    if not parser.parse_args().execute:
        parser.error('Explicit --execute required')
    with (HERE/'advance-dense.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller = read_json(HERE/'started.json')
        pid = controller['pid']
        assert start_ticks(pid) == controller['start_ticks'], 'Controller identity changed'
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        assert b'continue.py' in command and b'--execute' in command
        parent_fd = os.pidfd_open(pid)
        self_fd = os.pidfd_open(os.getpid())
        # A separate CPU-only guard resumes the coordinator even if this supervisor is killed.
        guard = os.fork()
        if guard == 0:
            try:
                select.select([self_fd],[],[])
                try: signal.pidfd_send_signal(parent_fd,signal.SIGCONT)
                except ProcessLookupError: pass
            finally: os._exit(0)
        os.close(self_fd)
        paused = False
        try:
            signal.pidfd_send_signal(parent_fd,signal.SIGSTOP)
            paused = True
            for _ in range(100):
                if '\nState:\tT' in Path(f'/proc/{pid}/status').read_text(): break
                time.sleep(.01)
            else: raise RuntimeError('Coordinator did not stop')
            status = read_json(HERE/'status.json')
            assert status.get('phase') == 'pretraining_non_dense', status
            config = read_json(HERE/'campaign.json')
            run = Path(config['training_queues'][0]['run'])
            plan = verify_plan(run)
            state = read_json(run/'state.json')
            assert all(state['jobs'][j['id']]['status'] in {'completed','failed'}
                       for j in plan['jobs'] if j['method_id'] != 'dense')
            dense = [j for j in plan['jobs'] if j['method_id']=='dense']
            assert len(dense)==1 and state['jobs'][dense[0]['id']]['status']=='planned'
            hardware = read_json(run/'hardware.json')['gpus']
            # Borrow only the allocation still held by this campaign coordinator.
            owned = {os.readlink(p) for p in Path(f'/proc/{pid}/fd').iterdir() if p.exists()}
            for gpu in hardware:
                assert str(ROOT/'locks'/(gpu['uuid']+'.lock')) in owned
            require_idle(hardware)
            record = {'status':'running','controller':controller,'supervisor_pid':os.getpid(),
                      'guard_pid':guard,'run':str(run),'job':dense[0]['id'],
                      'gpus':[g['uuid'] for g in hardware],'started_at':time.time(),
                      'policy':'User requested early Dense. Coordinator paused; N020 continues. No recipe/runtime changes.'}
            write_json(HERE/'advance-dense.json',record)
            write_json(HERE/'status.json',{'status':'running','phase':'pretraining_dense_early_and_n020',
                                          'controller_paused':pid,'dense_supervisor':os.getpid()})
            run_plan(run,','.join(g['uuid'] for g in hardware),True,
                     only_methods=['dense'],hardware_allocation=hardware)
            record.update(status=read_json(run/'state.json')['jobs'][dense[0]['id']]['status'],ended_at=time.time())
            write_json(HERE/'advance-dense.json',record)
        except BaseException as error:
            write_json(HERE/'advance-dense-error.json',{'error':str(error),'time':time.time()})
            raise
        finally:
            if paused:
                # Original coordinator skips the now-started Dense job and keeps timing behind both full queues.
                write_json(HERE/'status.json',{'status':'running','phase':'pretraining_non_dense',
                                              'dense_advanced':True,'controller_resumed':pid})
                try: signal.pidfd_send_signal(parent_fd,signal.SIGCONT)
                except ProcessLookupError: pass
            os.close(parent_fd)

if __name__ == '__main__': main()
