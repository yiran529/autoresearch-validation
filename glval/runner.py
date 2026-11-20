"""Explicit foreground runner; use tmux for long jobs. No autonomous GPU probing loop."""
import csv
import fcntl
import io
import os
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

from .common import ROOT, read_json, write_json
from .artifacts import run_directory
from .plans import preflight, verify_plan
from .results import inspect_job, report


def query_gpus():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    return [dict(zip(("index", "uuid", "name", "memory_mib", "utilization"),
                     [x.strip() for x in row])) for row in csv.reader(io.StringIO(result))]


@contextmanager
def gpu_lease(selection, count):
    requested = selection.split(",")
    if len(requested) != count or len(requested) != len(set(requested)):
        raise ValueError(f"Specify exactly {count} distinct dedicated GPUs")
    available = query_gpus()
    selected = []
    for token in requested:
        matches = [g for g in available if token in {g["index"], g["uuid"]}]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous GPU: {token}")
        selected.append(matches[0])
    if len({g["uuid"] for g in selected}) != count:
        raise ValueError("GPU aliases resolved to duplicate UUIDs")
    locks = ROOT / "locks"
    locks.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(selected, key=lambda x: x["uuid"]):
            file = stack.enter_context((locks / (gpu["uuid"] + ".lock")).open("a"))
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
        ], text=True)
        selected_ids = {g["uuid"] for g in selected}
        for row in csv.reader(io.StringIO(active)):
            if row and row[0].strip() in selected_ids:
                raise ValueError("Selected GPU already has a compute process")
        fresh = {g["uuid"]: g for g in query_gpus()}
        if any(int(fresh[g["uuid"]]["memory_mib"]) > 512 or int(fresh[g["uuid"]]["utilization"]) > 5 for g in selected):
            raise ValueError("Selected GPU is not idle")
        yield selected


def stop_group(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def probe_plan(run, job_id, selection, steps, execute=False):
    if not execute:
        raise ValueError("GPU probe is disabled without explicit --execute")
    if not 1 <= steps <= 5000:
        raise ValueError("Probe steps must be between 1 and 5000")
    run = run_directory(run)
    plan = verify_plan(run)
    matches = [j for j in plan["jobs"] if j["id"] == job_id]
    if len(matches) != 1:
        raise ValueError("Select a job ID from plan.json")
    job = matches[0]
    output = run / "checks" / (job_id + "-" + str(time.time_ns()))
    output.mkdir(parents=True)
    runtime = run / job["runtime"]
    with gpu_lease(selection, plan["recipe"]["world_size"]) as hardware:
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.update(CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in hardware), PYTHONPATH=str(runtime),
                   PYTHONDONTWRITEBYTECODE="1", GLVAL_OUTPUT=str(output), WANDB_MODE="disabled",
                   XDG_CACHE_HOME=str(ROOT / "cache"), TORCH_HOME=str(ROOT / "cache/torch"),
                   TRITON_CACHE_DIR=str(ROOT / "cache/triton"), CUDA_CACHE_PATH=str(ROOT / "cache/cuda"))
        command = [plan["execution"]["python"], "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node", str(plan["recipe"]["world_size"]), "--module", "glval.probe", "--steps", str(steps)]
        process = None
        try:
            with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
                process = subprocess.Popen(command, cwd=runtime, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                code = process.wait(timeout=600)
        except BaseException:
            if process:
                stop_group(process)
            raise
        record = {"command": command, "returncode": code, "gpus": hardware,
                  "status": "probe_pass" if code == 0 else "probe_failed",
                  "scope": "synthetic only; actual workload, semantics and refresh coverage require additional checks"}
        write_json(output / "process.json", record)
        return {"output": str(output), **record}


def run_plan(run, selection, execute=False, only_methods=None, hardware_allocation=None):
    if not execute:
        raise ValueError("Training is disabled without explicit --execute")
    run = run_directory(run)
    plan = verify_plan(run)
    check = preflight(run)
    if check["missing_inputs"]:
        raise ValueError(f"Missing inputs: {check['missing_inputs']}")
    with (run / "runner.lock").open("a") as runner_lock:
        fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(run / "state.json")
        selected = [j for j in plan["jobs"] if only_methods is None or j["method_id"] in only_methods]
        if not selected or any(state["jobs"][j["id"]]["status"] != "planned" for j in selected):
            raise ValueError("Selected jobs already started or selection is empty; no implicit retries")
        allocation = nullcontext(hardware_allocation) if hardware_allocation is not None else gpu_lease(selection, plan["recipe"]["world_size"])
        with allocation as hardware:
            write_json(run / "hardware.json", {"gpus": hardware, "selection": selection,
                                                "lock_scope": "this scaffold only; use a dedicated allocation"})
            state["status"] = "running"
            write_json(run / "state.json", state)
            for job in selected:
                # Verify again between jobs; runtime and adapter must remain frozen.
                verify_plan(run)
                output = run / job["output"]
                output.mkdir(parents=True, exist_ok=False)
                runtime = run / job["runtime"]
                env = os.environ.copy()
                env.pop("PYTHONHOME", None)
                env.update(plan["execution"].get("environment", {}))
                cache = ROOT / "cache"
                tracking = job.get("tracking", {})
                env.update({"CUDA_VISIBLE_DEVICES": ",".join(g["uuid"] for g in hardware),
                            "PYTHONPATH": str(runtime), "PYTHONDONTWRITEBYTECODE": "1",
                            "GLVAL_OUTPUT": str(output), "WANDB_MODE": tracking.get("mode", "disabled"), "WANDB_DIR": str(output),
                            "WANDB_RUN_GROUP": tracking.get("group", plan["id"]),
                            "WANDB_RUN_ID": plan["plan_hash"][:10] + "-" + job["id"],
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                            "HF_HOME": str(cache / "huggingface"), "HF_DATASETS_CACHE": str(cache / "datasets"),
                            "XDG_CACHE_HOME": str(cache), "TORCH_HOME": str(cache / "torch"),
                            "TRITON_CACHE_DIR": str(cache / "triton"), "CUDA_CACHE_PATH": str(cache / "cuda"),
                            "CUPY_CACHE_DIR": str(cache / "cupy"),
                            "MPLCONFIGDIR": str(cache / "matplotlib"), "TOKENIZERS_PARALLELISM": "false"})
                if plan["recipe"].get("tokenizer_path"):
                    env["GLVAL_TOKENIZER_PATH"] = plan["recipe"]["tokenizer_path"]
                status = state["jobs"][job["id"]]
                status.update(status="running", started_at=time.time())
                write_json(run / "state.json", state)
                started, process = time.monotonic(), None
                try:
                    with (output / "stdout.log").open("w") as stdout, (output / "stderr.log").open("w") as stderr:
                        process = subprocess.Popen(job["command"], cwd=runtime, env=env,
                                                   stdout=stdout, stderr=stderr, start_new_session=True)
                        status["pid"] = process.pid
                        write_json(run / "state.json", state)
                        code = process.wait(timeout=plan["execution"].get("timeout_seconds") or None)
                    status.update(returncode=code, elapsed_seconds=time.monotonic() - started)
                    inspected = inspect_job(run, job, plan["recipe"])
                    status.update(status="completed" if code == 0 and inspected["status"] == "verified" else "failed",
                                  validation=inspected)
                except BaseException as error:
                    if process:
                        stop_group(process)
                    status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                  error=str(error), elapsed_seconds=time.monotonic() - started)
                    state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                    write_json(run / "state.json", state)
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                write_json(run / "state.json", state)
            if any(j["status"] == "planned" for j in state["jobs"].values()):
                state["status"] = "partial"
            else:
                state["status"] = "completed" if all(j["status"] == "completed" for j in state["jobs"].values()) else "partial_or_failed"
            write_json(run / "state.json", state)
            return report(run, plan)
