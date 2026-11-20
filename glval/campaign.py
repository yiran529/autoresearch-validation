"""Parallel full pretraining, Dense last, followed by one fixed-GPU timing queue."""
import argparse
import multiprocessing
from pathlib import Path

from .common import ROOT, read_json, write_json
from .plans import verify_plan
from .runner import gpu_lease, run_plan


def stage_worker(run, hardware, methods, result_file):
    try:
        result = run_plan(run, ",".join(g["uuid"] for g in hardware), True,
                          only_methods=methods, hardware_allocation=hardware)
        write_json(result_file, result)
        state = read_json(Path(run) / "state.json")
        if any(j["status"] not in {"completed", "planned"} for j in state["jobs"].values()):
            raise RuntimeError("Selected stage has failed jobs; inspect results before continuing")
    except BaseException as error:
        write_json(result_file, {"status": "failed", "error": str(error)})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        p.error("Explicit --execute required")
    manifest = Path(args.manifest).resolve()
    if not manifest.is_relative_to(ROOT):
        p.error("Manifest must be inside scaffold")
    config = read_json(manifest)
    for queue in config["training_queues"]:
        verify_plan(queue["run"])
    verify_plan(config["timing_run"])
    context = multiprocessing.get_context("fork")
    status_file = manifest.parent / "status.json"
    def launch_phase(phase, tasks):
        write_json(status_file, {"status": "running", "phase": phase})
        children = []
        for index, (run, hardware, methods) in enumerate(tasks):
            child = context.Process(target=stage_worker, args=(run, hardware, methods,
                                    manifest.parent / (phase + f"-queue{index}.json")))
            child.start()
            children.append(child)
        for child in children:
            child.join()
        if any(child.exitcode != 0 for child in children):
            write_json(status_file, {"status": "failed", "phase": phase,
                                     "exitcodes": [c.exitcode for c in children]})
            raise SystemExit(1)

    with gpu_lease(config["training_queues"][0]["gpus"], 4) as h0:
        with gpu_lease(config["training_queues"][1]["gpus"], 4) as h1:
            tasks = []
            for queue, hardware in zip(config["training_queues"], [h0, h1]):
                plan = verify_plan(queue["run"])
                methods = list(dict.fromkeys(j["method_id"] for j in plan["jobs"] if j["method_id"] != "dense"))
                tasks.append((queue["run"], hardware, methods))
            launch_phase("pretraining_non_dense", tasks)
        launch_phase("pretraining_dense_last", [(config["training_queues"][0]["run"], h0, ["dense"])])
        timing = verify_plan(config["timing_run"])
        methods = list(dict.fromkeys(j["method_id"] for j in timing["jobs"] if j["method_id"] != "dense"))
        launch_phase("timing_non_dense", [(config["timing_run"], h0, methods)])
        launch_phase("timing_dense_last", [(config["timing_run"], h0, ["dense"])])
    write_json(status_file, {"status": "completed"})


if __name__ == "__main__":
    main()
