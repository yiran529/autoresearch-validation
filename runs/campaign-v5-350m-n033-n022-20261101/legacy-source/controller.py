import json,time
from pathlib import Path
from glval.runner import gpu_lease,run_plan
from glval.common import read_json,write_json
from glval.plans import verify_plan
b=Path('/home/wyr/greedylore-validation');c=b/'campaigns/v5-350m-n033-n022-20261101';full=b/'runs/v5-350m-n033-n022-20261101';smoke=b/'runs/v5-350m-n033-n022-smoke-20261101'
def state(stage,**kw):write_json(c/'status.json',{'stage':stage,'at':time.time(),**kw})
try:
 assert read_json(b/'runs/v5-final-four-130m-20261031-offline/state.json')['status']=='completed'
 for p in [full,smoke]:verify_plan(p)
 with gpu_lease('4,5,6,7',4) as hardware:
  write_json(c/'allocation.json',{'gpus':hardware,'selection':'4,5,6,7','mode':'serial N33 then N22'})
  state('real_350m_smoke',scope='8steps compression2;memory/path test only;no full-quality claim')
  run_plan(smoke,'4,5,6,7',True,hardware_allocation=hardware)
  s=read_json(smoke/'state.json');assert s['status']=='completed', '350M smoke failed; full queue blocked'
  for j in verify_plan(smoke)['jobs']:
   for rank in range(4):
    z=read_json(smoke/j['output']/f'runtime-rank-{rank}.json');assert z['iter']==8 and len(z.get('timing_refresh_steps',[]))>=6
  state('full_training',order=['N033 Research','N022 Engineer'],seed=1243,updates=60000,tracking='offline')
  run_plan(full,'4,5,6,7',True,hardware_allocation=hardware)
  s=read_json(full/'state.json');state(s['status'],jobs=s['jobs'])
except BaseException as e:
 state('failed',error=str(e));raise
