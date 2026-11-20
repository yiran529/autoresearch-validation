import json,time
from pathlib import Path
from glval.runner import run_plan
from glval.common import write_json,read_json
from glval.plans import verify_plan
b=Path('/home/wyr/greedylore-validation');c=b/'campaigns/v5-final-four-130m-20261031';r=b/'runs/v5-final-four-130m-20261031-offline'
old=verify_plan(b/'runs/v5-final-four-130m-20261031');new=verify_plan(r)
assert [(j['source_commit'],j['method_hash']) for j in old['jobs']]==[(j['source_commit'],j['method_hash']) for j in new['jobs']]
assert read_json(b/'runs/v5-four-130m-smoke-20261031/state.json')['status']=='completed'
assert len(list((c/'checks').glob('*/*/equivalence-rank-*.json')))==32
write_json(c/'offline-transition.json',{'reason':'User explicitly requested W&B offline; online attempts retained','old_run':str(b/'runs/v5-final-four-130m-20261031'),'new_run':str(r),'checks_reused':'identical source/method hashes; all smoke jobs completed','old_online_attempts_retained':True})
try:
 write_json(c/'status.json',{'stage':'full_training','run':str(r),'at':time.time(),'recovery':'W&B offline authorized; checks reused on identical source/method hashes'})
 run_plan(r,'4,5,6,7',True)
 state=read_json(r/'state.json');write_json(c/'status.json',{'stage':state['status'],'run':str(r),'at':time.time(),'jobs':state['jobs']})
except BaseException as e:
 write_json(c/'status.json',{'stage':'failed','run':str(r),'error':str(e),'at':time.time()});raise
