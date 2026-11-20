"""Authorized serial GLUE validation, fixed GPU4-7, no other process control."""
import os,sys,time,json
from pathlib import Path
ROOT=Path("/home/wyr/greedylore-validation")
sys.path.insert(0,str(ROOT))
from glval.runner import gpu_lease,run_plan
from glval.plans import preflight
from glval.common import write_json,read_json
os.environ.update(HF_EVALUATE_OFFLINE="1",HF_DATASETS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1")
c=Path(__file__).parent
runs=[ROOT/"runs"/x for x in ["v5-glue-integration-smoke-20261104","v5-glue-sst2-20261104","v5-glue-qnli-20261104"]]
for run in runs:
 check=preflight(run)
 if check["missing_inputs"]:raise RuntimeError(check["missing_inputs"])
write_json(c/"status.json",{"stage":"resource_check","pid":os.getpid(),"updated_at":time.time()})
try:
 with gpu_lease("4,5,6,7",4) as hardware:
  write_json(c/"allocation.json",{"gpus":hardware,"selection":"4,5,6,7","scope":"Scaffold lease across smoke, SST-2 and QNLI. CORAL confirmed no active tasks before launch."})
  for run in runs:
   write_json(c/"status.json",{"stage":run.name,"status":"running","pid":os.getpid(),"updated_at":time.time()})
   print("START",run.name,flush=True)
   run_plan(run,"4,5,6,7",execute=True,hardware_allocation=hardware)
   state=read_json(run/"state.json")
   if state["status"]!="completed":raise RuntimeError("Stop queue: "+run.name+" "+state["status"])
   for job in read_json(run/"plan.json")["jobs"]:
    output=run/job["output"]
    forbidden=[p for p in output.rglob("*") if p.is_file() and (p.suffix in {".pt",".pth",".safetensors"} or p.name in {"pytorch_model.bin","optimizer.bin","scheduler.bin"})]
    if forbidden:raise RuntimeError("Unexpected checkpoint output: "+str(forbidden))
   print("COMPLETED",run.name,flush=True)
 write_json(c/"status.json",{"stage":"completed","status":"completed","pid":os.getpid(),"updated_at":time.time()})
except BaseException as exc:
 write_json(c/"status.json",{"stage":"stopped","status":"failed_or_interrupted","error":str(exc),"pid":os.getpid(),"updated_at":time.time()})
 raise
