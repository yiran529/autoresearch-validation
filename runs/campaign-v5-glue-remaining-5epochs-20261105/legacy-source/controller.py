"""Wait for the existing authorized queue, then run the remaining four GLUE tasks."""
import os,sys,time
from pathlib import Path
ROOT=Path("/home/wyr/greedylore-validation");sys.path.insert(0,str(ROOT))
from glval.common import read_json,write_json
from glval.runner import gpu_lease,run_plan
from glval.plans import preflight
C=Path(__file__).parent
predecessor=ROOT/"campaigns/v5-glue-four-tasks-5epochs-20261105/status.json"
os.environ.update(HF_EVALUATE_OFFLINE="1",HF_DATASETS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1")
def status(stage,**extra):
 write_json(C/"status.json",dict(stage=stage,pid=os.getpid(),updated_at=time.time(),**extra))
try:
 status("waiting_for_mnli",status="waiting",predecessor=str(predecessor))
 while True:
  old=read_json(predecessor)
  if old.get("stage")=="completed" and old.get("status")=="completed":
   if Path("/proc",str(old["pid"])).exists():
    time.sleep(10);continue
   break
  if old.get("status") not in {"running","waiting"}:
   raise RuntimeError("Predecessor stopped without completion; additional jobs remain planned")
  p=Path("/proc",str(old["pid"]))
  if not p.exists():raise RuntimeError("Predecessor exited with stale running state; no GPU tasks launched")
  args=(p/"cmdline").read_bytes()
  if p.stat().st_uid!=os.getuid() or b"v5-glue-four-tasks-5epochs-20261105/controller.py" not in args:
   raise RuntimeError("Predecessor identity mismatch; no GPU tasks launched")
  time.sleep(30)
 runs=[ROOT/'runs'/x for x in ['v5-glue-cola-5epochs-20261105', 'v5-glue-stsb-5epochs-20261105', 'v5-glue-mrpc-5epochs-20261105', 'v5-glue-rte-5epochs-20261105']]
 for run in runs:
  if preflight(run)["missing_inputs"]:raise RuntimeError("Missing inputs")
 with gpu_lease("4,5,6,7",4) as hardware:
  write_json(C/"allocation.json",{"gpus":hardware,"selection":"4,5,6,7","scope":"Remaining GLUE tasks after MNLI"})
  for run in runs:
   status(run.name,status="running")
   print("START",run.name,flush=True)
   run_plan(run,"4,5,6,7",execute=True,hardware_allocation=hardware)
   if read_json(run/"state.json")["status"]!="completed":raise RuntimeError("Stage failed: "+run.name)
   print("COMPLETED",run.name,flush=True)
 status("completed",status="completed")
except BaseException as exc:
 status("stopped",status="failed_or_interrupted",error=str(exc))
 raise

