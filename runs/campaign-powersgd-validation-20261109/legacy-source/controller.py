"""Run each authorized 60M method after ten consecutive minutes of four-GPU idle."""
import csv
import io
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path("/home/wyr/greedylore-validation")
sys.path.insert(0, str(ROOT))
from glval.common import read_json, write_json
from glval.plans import preflight
from glval.runner import gpu_lease, query_gpus, run_plan

CAMPAIGN = Path(__file__).parent
SELECTION = "4,5,6,7"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ[key] = "http://127.0.0.1:28731"
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"

def publish(stage, **details):
    write_json(CAMPAIGN / "status.json", {
        "stage": stage, "pid": os.getpid(), "updated_at": time.time(), **details
    })

def wait_for_idle(run_name):
    idle_since = None
    while True:
        selected = [gpu for gpu in query_gpus() if gpu["index"] in SELECTION.split(",")]
        if len(selected) != 4:
            raise RuntimeError("Cannot resolve all four assigned GPUs")
        selected_ids = {gpu["uuid"] for gpu in selected}
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"
        ], text=True)
        busy_process = any(
            row and row[0].strip() in selected_ids
            for row in csv.reader(io.StringIO(output))
        )
        idle = not busy_process and all(
            int(gpu["memory_mib"]) <= 512 and int(gpu["utilization"]) <= 5
            for gpu in selected
        )
        now = time.monotonic()
        if idle:
            if idle_since is None:
                idle_since = now
        else:
            idle_since = None
        elapsed = now - idle_since if idle_since is not None else 0
        publish("waiting_for_idle", run=run_name, idle_seconds=round(elapsed, 1),
                required_idle_seconds=600, poll_seconds=10, gpus=selected)
        if elapsed >= 600:
            return
        time.sleep(10)


os.environ.update(HF_EVALUATE_OFFLINE="1", HF_DATASETS_OFFLINE="1")
try:
    for path in read_json(CAMPAIGN / "queue.json"):
        run = Path(path)
        state = read_json(run / "state.json")
        if state["status"] == "completed":
            print("REUSE", run.name, flush=True)
            continue
        if any(job["status"] != "planned" for job in state["jobs"].values()):
            raise RuntimeError("Existing incomplete attempt requires explicit recovery: " + run.name)
        if preflight(run)["missing_inputs"]:
            raise RuntimeError("Missing inputs for " + run.name)
        with ExitStack() as resources:
            while True:
                wait_for_idle(run.name)
                try:
                    hardware = resources.enter_context(gpu_lease(SELECTION, 4))
                except (ValueError, BlockingIOError):
                    print("GPU allocation changed; restarting idle timer", flush=True)
                    continue
                break
            publish("training", run=run.name, gpus=hardware)
            print("START", run.name, flush=True)
            run_plan(run, SELECTION, execute=True, hardware_allocation=hardware)
            if read_json(run / "state.json")["status"] != "completed":
                raise RuntimeError("Training failed: " + run.name)
            print("COMPLETED", run.name, flush=True)
    publish("completed")
except BaseException as error:
    publish("stopped", error=str(error))
    raise
