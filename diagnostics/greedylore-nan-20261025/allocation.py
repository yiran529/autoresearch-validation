"""Reserve idle GPUs0-3; borrow only this campaign's existing reservation if needed."""
import csv
import fcntl
import io
import os
from pathlib import Path
import subprocess
import threading
from glval.common import ROOT,read_json
from glval.runner import query_gpus

_files=[]

def reserve():
    hardware=[g for g in query_gpus() if g['index'] in {'0','1','2','3'}]
    uuids={g['uuid'] for g in hardware}
    assert len(hardware)==4
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    assert not any(row and row[0].strip() in uuids for row in csv.reader(io.StringIO(active))), 'GPU0-3 busy'
    assert all(int(g['memory_mib'])<=512 and int(g['utilization'])<=5 for g in hardware),'GPU0-3 not idle'
    mutex=(ROOT/'locks/greedylore-nan-diagnostic.lock').open('a')
    fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB);_files.append(mutex)
    pending=[]
    for gpu in sorted(hardware,key=lambda g:g['uuid']):
        path=ROOT/'locks'/(gpu['uuid']+'.lock')
        file=path.open('a');_files.append(file)
        try:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            h=ROOT/'campaigns/llama130m-muon-engineer-maxu-20261025'
            status=read_json(h/'first-free-timing-status.json')
            assert status.get('gpus')==['4','5','6','7'], 'Existing campaign may use diagnostic GPUs'
            owner=read_json(h/'first-free-timing-resumed.json')['pid']
            owned=set()
            for p in Path(f'/proc/{owner}/fd').iterdir():
                try:owned.add(os.readlink(p))
                except FileNotFoundError:pass
            assert str(path) in owned, 'GPU reservation belongs to another job'
            pending.append(file)
    def transfer():
        for file in pending:fcntl.flock(file,fcntl.LOCK_EX)
    threading.Thread(target=transfer,daemon=True).start()
    return hardware,{'borrowed_from_same_campaign':bool(pending),'transfer_queued':bool(pending)}
