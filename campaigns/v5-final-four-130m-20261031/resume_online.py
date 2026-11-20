
# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import json,time
from pathlib import Path
from glval.runner import run_plan
from glval.common import write_json,read_json
from glval.plans import verify_plan
b=Path('/home/wyr/greedylore-validation');c=b/'campaigns/v5-final-four-130m-20261031';r=b/'runs/v5-final-four-130m-20261031-r2'
old=verify_plan(b/'runs/v5-final-four-130m-20261031');new=verify_plan(r)
assert [(j['source_commit'],j['method_hash']) for j in old['jobs']]==[(j['source_commit'],j['method_hash']) for j in new['jobs']]
assert read_json(b/'runs/v5-four-130m-smoke-20261031/state.json')['status']=='completed'
assert len(list((c/'checks').glob('*/*/equivalence-rank-*.json')))==32
write_json(c/'recovery.json',{'reason':'W&B initialization timeout on direct network; use existing historical proxy127.0.0.1:28731','old_run':str(b/'runs/v5-final-four-130m-20261031'),'new_run':str(r),'checks_reused':'identical source/method hashes; all smoke jobs completed','training_started_in_failed_attempts':False})
try:
 write_json(c/'status.json',{'stage':'full_training','run':str(r),'at':time.time(),'recovery':'historical W&B proxy restored; original failures retained'})
 run_plan(r,'4,5,6,7',True)
 state=read_json(r/'state.json');write_json(c/'status.json',{'stage':state['status'],'run':str(r),'at':time.time(),'jobs':state['jobs']})
except BaseException as e:
 write_json(c/'status.json',{'stage':'failed','run':str(r),'error':str(e),'at':time.time()});raise
