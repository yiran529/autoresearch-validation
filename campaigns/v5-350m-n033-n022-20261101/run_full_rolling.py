
# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import time
from pathlib import Path
from glval.runner import run_plan
from glval.common import write_json,read_json
b=Path('/home/wyr/greedylore-validation');c=b/'campaigns/v5-350m-n033-n022-20261101';r=b/'runs/v5-350m-n033-n022-rolling20k-20261101'
try:
 write_json(c/'status.json',{'stage':'full_training','run':str(r),'at':time.time(),'order':['N33','N22'],'gpus':'4,5,6,7','short_tests':'skipped by explicit user request','checkpoint':'rolling20k;latest only per method under/dev/shm/wyr_tmp'})
 run_plan(r,'4,5,6,7',True)
 s=read_json(r/'state.json');write_json(c/'status.json',{'stage':s['status'],'run':str(r),'at':time.time(),'jobs':s['jobs']})
except BaseException as e:
 write_json(c/'status.json',{'stage':'failed','run':str(r),'error':str(e),'at':time.time()});raise
