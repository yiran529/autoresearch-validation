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
RUN = ROOT / "runs/v5-60m-three-idle10m-20261107"
SELECTION = "4,5,6,7"
METHODS = ["greedylore-engineer-streamfix", "v5-n033-research", "dense"]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ[key] = "http://127.0.0.1:28731"
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"

def publish(stage, **details):
    write_json(CAMPAIGN / "status.json", {
        "stage": stage, "pid": os.getpid(), "updated_at": time.time(), **details
    })

def wait_for_idle(method):
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
        publish("waiting_for_idle", method=method, idle_seconds=round(elapsed, 1),
                required_idle_seconds=600, poll_seconds=10, gpus=selected)
        if elapsed >= 600:
            return
        time.sleep(10)

try:
    if preflight(RUN)["missing_inputs"]:
        raise RuntimeError("Missing experiment inputs")
    for method in METHODS:
        with ExitStack() as resources:
            while True:
                wait_for_idle(method)
                try:
                    hardware = resources.enter_context(gpu_lease(SELECTION, 4))
                except (ValueError, BlockingIOError):
                    print("GPU allocation changed; restarting idle timer", flush=True)
                    continue
                break
            publish("training", method=method, gpus=hardware)
            print("START", method, flush=True)
            run_plan(RUN, SELECTION, execute=True, only_methods=[method],
                     hardware_allocation=hardware)
            state = read_json(RUN / "state.json")
            if state["jobs"][method + "-s1243"]["status"] != "completed":
                raise RuntimeError("Training failed: " + method)
            print("COMPLETED", method, flush=True)
    publish("completed")
except BaseException as error:
    publish("stopped", error=str(error))
    raise

