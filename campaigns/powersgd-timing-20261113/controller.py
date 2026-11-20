"""Explicitly authorized timing queue; any four GPUs idle for 600 seconds."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import argparse,csv,fcntl,io,itertools,os,subprocess,sys,time
from contextlib import ExitStack
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE))
from glval.common import read_json,write_json
from glval.plans import preflight
from glval.runner import gpu_lease,query_gpus,run_plan
from idle_policy import update_idle

def publish(stage,**details):
    write_json(HERE/"status.json",{"stage":stage,"pid":os.getpid(),"updated_at":time.time(),**details})

def wait_and_lease(resources,run):
    since={}
    while True:
        gpus=query_gpus()
        raw=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],text=True)
        busy={r[0].strip() for r in csv.reader(io.StringIO(raw)) if r}
        now=time.monotonic()
        eligible=update_idle(gpus,busy,since,now)
        publish("waiting_for_idle",run=run.name,required_idle_seconds=600,poll_seconds=10,
                gpus=[{**g,"idle_seconds":round(now-since[g["uuid"]],1) if g["uuid"] in since else 0} for g in gpus])
        if eligible:
            # Try every qualified quartet; a lease held elsewhere must not block other idle GPUs.
            qualified=[g for g in gpus if g["uuid"] in since and now-since[g["uuid"]] >= 600]
            for group in itertools.combinations(qualified,4):
                selection=",".join(g["index"] for g in group)
                try:
                    hardware=resources.enter_context(gpu_lease(selection,4))
                except (ValueError,BlockingIOError):
                    continue
                return selection,hardware
        time.sleep(10)

def summarize():
    values={"bf16":[],"fp32":[]};jobs=[]
    for name in read_json(HERE/"queue.json"):
        run=Path(name);state=read_json(run/"state.json");plan=read_json(run/"plan.json")
        if state["status"] != "completed": raise RuntimeError("Incomplete timing "+run.name)
        job=plan["jobs"][0];result=read_json(run/job["output"]/"all_results.json")
        precision="bf16" if plan["recipe"]["args"]["dtype"]=="bfloat16" else "fp32"
        values[precision].append(result["mean_iteration_seconds"])
        jobs.append({"run":str(run),"precision":precision,"mean_iteration_seconds":result["mean_iteration_seconds"],
                     "hardware":read_json(run/"hardware.json")})
    import math
    baseline=read_json(HERE/"provenance.json")["historical_dense_seconds"]
    means={k:sum(v)/len(v) for k,v in values.items()}
    speedups={k:baseline[k]/v for k,v in means.items()}
    write_json(HERE/"summary.json",{"jobs":jobs,"mean_iteration_seconds":means,
        "speedup_vs_historical_dense":speedups,"geomean_speedup":math.sqrt(math.prod(speedups.values())),
        "comparison_scope":"Historical Dense; GPU topology and run conditions may differ. No same-batch Dense measured."})

def main():
    p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");args=p.parse_args()
    if not args.execute: p.error("Explicit --execute required")
    os.environ.update(PYTHONDONTWRITEBYTECODE="1",HF_EVALUATE_OFFLINE="1",HF_DATASETS_OFFLINE="1")
    with (HERE/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for name in read_json(HERE/"queue.json"):
                run=Path(name);state=read_json(run/"state.json")
                if state["status"] == "completed": continue
                if any(s["status"] != "planned" for s in state["jobs"].values()):
                    raise RuntimeError("No implicit retries: "+run.name)
                if preflight(run)["status"] != "static_pass": raise RuntimeError("Preflight failed")
                with ExitStack() as resources:
                    selection,hardware=wait_and_lease(resources,run)
                    topology=subprocess.check_output(["nvidia-smi","topo","-m"],text=True)
                    write_json(run/"allocation-topology.json",{"selection":selection,"gpus":hardware,"topology":topology})
                    publish("timing",run=run.name,selection=selection,gpus=hardware)
                    print("START",run.name,selection,flush=True)
                    run_plan(run,selection,execute=True,hardware_allocation=hardware)
                    if read_json(run/"state.json")["status"] != "completed":
                        raise RuntimeError("Timing failed: "+run.name)
                    print("COMPLETED",run.name,flush=True)
            summarize()
            publish("completed")
        except BaseException as error:
            publish("stopped",error=str(error));raise
if __name__=="__main__":main()
