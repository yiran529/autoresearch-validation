import math
from pathlib import Path

from .common import inside, read_json, write_json
from .artifacts import run_directory


def finite(value):
    return type(value) in {float, int} and math.isfinite(value)


def saved_results(run):
    """Read saved PPL/time values, including archives, without regenerating reports."""
    run = run_directory(run)
    if not run.is_dir():
        raise ValueError(f"Run does not exist: {run}")
    names = ("report.json", "summary.json", "analysis.json", "combined-results.json",
             "state.json", "status.json", "continuation-result.json")
    paths = {run / name for name in names if (run / name).exists()}
    for pattern in ("jobs/*/output/all_results.json", "cases/*/output/all_results.json",
                    "jobs/*/output/timing-summary.json", "cases/*/output/timing-summary.json",
                    "*-status.json", "smoke-result*.json"):
        paths.update(run.glob(pattern))
    def metrics(value, prefix=""):
        found = {}
        if isinstance(value, dict):
            for key, item in value.items():
                label = prefix + "." + key if prefix else key
                if finite(item) and any(word in key.lower() for word in ("ppl", "perplex", "seconds", "speedup")):
                    found[label] = item
                elif isinstance(item, (dict, list)):
                    found.update(metrics(item, label))
        elif isinstance(value, list):
            # Per-step numeric arrays are intentionally omitted from the summary.
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    found.update(metrics(item, prefix + f"[{index}]"))
        return found
    files = []
    for path in sorted(paths):
        relative = path.relative_to(run).as_posix()
        data = read_json(inside(run, relative))
        files.append({"path": relative, "recorded_status": data.get("status", data.get("stage")),
                      "metrics": metrics(data)})
    return {"status": "saved_results", "run": str(run), "files": files,
            "limitations": ["Read-only saved measurements; not a new execution or revalidation.",
                            "Total wall seconds includes startup and evaluation; iteration seconds uses its recorded window."]}


def inspect_job(run, job, recipe):
    output = Path(run) / job["output"]
    errors, metrics = [], {}
    result_path = output / "all_results.json"
    if not result_path.exists():
        return {"status": "unverified", "errors": ["No final results"], "metrics": {}}
    raw = read_json(result_path)
    for key, expected in recipe["completion"].items():
        if raw.get(key) != expected:
            errors.append(f"Completion mismatch: {key}: {raw.get(key)!r} != {expected!r}")
    if recipe["workload"] == "glue":
        if type(raw.get("expected_steps")) is not int or raw["expected_steps"] <= 0 or raw.get("completed_steps") != raw["expected_steps"]:
            errors.append("GLUE update budget is not verified")
    for key in recipe["metrics"]:
        if not finite(raw.get(key)):
            errors.append(f"Missing/nonfinite metric: {key}")
        else:
            metrics[key] = raw[key]
    timing = recipe.get("kind") == "timing"
    if recipe["workload"] == "llama" and not timing and not (finite(raw.get("final_eval_tokens")) and raw["final_eval_tokens"] > 0):
        errors.append("LLaMA evaluated-token count missing")
    if timing:
        expected = recipe["args"]["measured_iterations"]
        elapsed, steps, losses = raw.get("rank_elapsed_seconds", []), raw.get("rank_iteration_seconds", []), raw.get("rank_losses", [])
        if not all(len(v) == recipe["world_size"] for v in (elapsed, steps, losses)):
            errors.append("Missing timing ranks")
        elif (any(not finite(t) or t <= 0 for t in elapsed)
              or any(len(v) != expected or any(not finite(x) or x <= 0 for x in v) for v in steps)
              or any(len(v) != expected or any(not finite(x) for x in v) for v in losses)):
            errors.append("Incomplete/nonfinite timing window")
        elif abs(raw["mean_iteration_seconds"] - max(elapsed) / expected) > 1e-9:
            errors.append("Timing aggregation does not match slowest rank")
    ranks = []
    for rank in range(recipe["world_size"]):
        path = output / f"runtime-rank-{rank}.json"
        if not path.exists():
            errors.append(f"Missing execution evidence for rank {rank}")
            continue
        evidence = read_json(path)
        if evidence.get("method_id") != job["method_id"] or evidence.get("hook_registrations") != 1 or evidence.get("hook_calls", 0) < 1:
            errors.append(f"Invalid binding for rank {rank}")
        source = evidence.get("hook_source", "")
        if job["runtime_files"].get(source) != evidence.get("hook_sha256") or not source:
            errors.append(f"Hook source identity mismatch for rank {rank}")
        ranks.append(evidence)
    if ranks and len({r.get("last_bucket_calls") for r in ranks}) != 1:
        errors.append("Ranks disagree on completed backward hook cycles")
    return {"status": "verified" if not errors else "invalid", "errors": errors, "metrics": metrics,
            "peak_memory_bytes": max((r.get("peak_memory_bytes", 0) for r in ranks), default=None),
            "refresh_coverage": "not certified by generic binding evidence"}


def report(run, plan):
    run = run_directory(run)
    json_target, markdown_target = inside(run, "report.json"), inside(run, "report.md")
    state = read_json(inside(run, "state.json"))
    jobs = []
    for job in plan["jobs"]:
        item = inspect_job(run, job, plan["recipe"])
        process = state["jobs"].get(job["id"], {})
        if process.get("status") != "completed":
            item["status"] = "unverified" if process.get("status") in {"planned", "running"} else "failed"
        item.update(id=job["id"], method_id=job["method_id"], seed=job["seed"], repetition=job.get("repetition", 1),
                    source_commit=job["source_commit"], process_status=process.get("status"),
                    total_wall_seconds=process.get("elapsed_seconds"))
        jobs.append(item)
    comparisons = []
    for candidate in jobs:
        if candidate["status"] != "verified":
            continue
        for base in jobs:
            references = plan.get("reference_methods", [j["method_id"] for j in jobs if j["method_id"] != plan["candidate"]])
            if (candidate["method_id"] == base["method_id"] or base["method_id"] not in references
                    or base["seed"] != candidate["seed"] or base["repetition"] != candidate["repetition"] or base["status"] != "verified"):
                continue
            comparisons.append({"method": candidate["method_id"], "seed": candidate["seed"], "repetition": candidate["repetition"], "baseline": base["method_id"],
                                "metric_deltas_candidate_minus_baseline": {
                                    k: candidate["metrics"][k] - v for k, v in base["metrics"].items()},
                                "total_wall_speedup": base["total_wall_seconds"] / candidate["total_wall_seconds"],
                                **({"measured_step_speedup": base["metrics"]["mean_iteration_seconds"] / candidate["metrics"]["mean_iteration_seconds"]}
                                   if plan["recipe"].get("kind") == "timing" else {})})
    payload = {"status": state["status"], "plan_hash": plan["plan_hash"], "recipe": plan["recipe"]["id"],
               "evaluation_split": plan["recipe"]["evaluation_split"], "jobs": jobs,
               "comparisons": comparisons,
               "limitations": ["Total wall time includes process startup and evaluation; not pure training throughput.",
                               "Generic hook evidence does not certify algorithmic equivalence or refresh coverage.",
                               "No aggregate scientific pass threshold is imposed."]}
    write_json(json_target, payload)
    lines = [f"# {plan['id']}", "", f"Recipe: `{plan['recipe']['id']}`; split: `{payload['evaluation_split']}`", "",
             "| Method | Seed | Status | Metrics | Total wall seconds |", "|---|---:|---|---|---:|"]
    for j in jobs:
        lines.append(f"| {j['method_id']} | {j['seed']} | {j['status']} | {j['metrics']} | {j['total_wall_seconds']} |")
        for error in j["errors"]:
            lines.extend(["", f"- {j['id']}: {error}"])
    lines.extend(["", "Comparisons: `report.json` (matched seeds, verified jobs only).", "",
                  *["- " + x for x in payload["limitations"]]])
    markdown_target.write_text("\n".join(lines) + "\n")
    return payload
