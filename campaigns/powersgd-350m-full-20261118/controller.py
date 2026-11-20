"""Run the authorized smoke then full experiment on idle GPUs 4-7."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from glval.common import read_json, write_json
from glval.plans import preflight, verify_plan
from glval.runner import gpu_lease, run_plan
from prepare import validate_contract

SELECTION = "4,5,6,7"


def publish(stage, **details):
    write_json(HERE / "status.json", {
        "stage": stage, "pid": os.getpid(), "updated_at": time.time(), **details,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    if not options.execute:
        parser.error("Explicit --execute required")
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_EVALUATE_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    with (HERE / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            queue = read_json(HERE / "queue.json")
            review = read_json(HERE / "review.json")
            if review["status"] != "approved":
                raise ValueError("Independent settings review has not passed")
            for phase, path in zip(("smoke", "full"), queue, strict=True):
                run = Path(path)
                plan = verify_plan(run)
                validate_contract(plan["recipe"], plan["execution"], phase)
                if review["plan_hashes"].get(path) != plan["plan_hash"]:
                    raise ValueError("Independent review does not match frozen plan")
                if any(j["tracking"]["mode"] != "disabled" for j in plan["jobs"]):
                    raise ValueError("W&B must remain disabled")
                state = read_json(run / "state.json")
                if state["status"] == "completed":
                    if any(j.get("validation", {}).get("status") != "verified" for j in state["jobs"].values()):
                        raise ValueError("Completed phase lacks verified evidence")
                    continue
                if any(j["status"] != "planned" for j in state["jobs"].values()):
                    raise ValueError("No implicit retry or resume: " + run.name)
                if preflight(run)["status"] != "static_pass":
                    raise ValueError("Preflight failed: " + run.name)
                while True:
                    # Acquire/check only these dedicated GPUs; never stop foreign processes.
                    from contextlib import ExitStack
                    with ExitStack() as resources:
                        try:
                            hardware = resources.enter_context(gpu_lease(SELECTION, 4))
                        except (ValueError, BlockingIOError) as error:
                            publish("waiting_for_idle", phase=phase, run=str(run), gpus=SELECTION,
                                    poll_seconds=30, reason=str(error))
                        else:
                            topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
                            write_json(run / "allocation-topology.json", {"selection": SELECTION, "topology": topology})
                            publish("training", phase=phase, run=str(run), gpus=hardware)
                            print("START", phase, run.name, SELECTION, flush=True)
                            run_plan(run, SELECTION, execute=True, hardware_allocation=hardware)
                            state = read_json(run / "state.json")
                            if state["status"] != "completed":
                                raise RuntimeError("Phase failed; remaining queue blocked: " + run.name)
                            if phase == "smoke":
                                job = plan["jobs"][0]
                                for rank in range(4):
                                    evidence = read_json(run / job["output"] / f"runtime-rank-{rank}.json")
                                    if evidence.get("iter", 0) < 8 or evidence.get("last_bucket_calls", 0) < 8:
                                        raise RuntimeError("Smoke did not cover the compressed path on every rank")
                            print("COMPLETED", phase, run.name, flush=True)
                            break
                    time.sleep(30)
            publish("completed")
        except BaseException as error:
            publish("stopped", error=str(error))
            raise


if __name__ == "__main__":
    main()
