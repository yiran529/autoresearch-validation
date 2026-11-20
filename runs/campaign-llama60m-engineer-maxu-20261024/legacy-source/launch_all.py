"""Run bounded real-model smoke, then the authorized campaign; no agent polling."""
import multiprocessing
import subprocess
from pathlib import Path
from glval.common import ROOT, read_json, write_json
from glval.runner import run_plan
HERE=Path(__file__).resolve().parent

def smoke_worker(run,gpus,index):
    result=run_plan(run,gpus,execute=True)
    write_json(HERE/f"smoke-result-q{index}.json",result)
    if result["status"] != "completed":raise SystemExit(1)

if __name__=="__main__":
    write_json(HERE/"bootstrap-status.json",{"status":"running","stage":"real_model_smoke"})
    paths=read_json(HERE/"smoke-plans.json")
    ctx=multiprocessing.get_context("fork")
    children=[]
    for i,(run,gpus) in enumerate(zip(paths,["0,1,2,3","4,5,6,7"])):
        child=ctx.Process(target=smoke_worker,args=(run,gpus,i));child.start();children.append(child)
    for child in children:child.join()
    if any(child.exitcode!=0 for child in children):
        write_json(HERE/"bootstrap-status.json",{"status":"failed","stage":"real_model_smoke","exitcodes":[c.exitcode for c in children]})
        raise SystemExit(1)
    write_json(HERE/"bootstrap-status.json",{"status":"running","stage":"formal_campaign"})
    code=subprocess.call(["python3","-B","-m","glval.campaign",str(HERE/"campaign.json"),"--execute"],cwd=ROOT)
    write_json(HERE/"bootstrap-status.json",{"status":"completed" if code==0 else "failed","stage":"formal_campaign","returncode":code})
    raise SystemExit(code)
