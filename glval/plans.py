import ast
import os
import shlex
import shutil
import sys
import uuid
from pathlib import Path

from .build import adapt_formal_runtime, build_runtime
from .artifacts import new_run_directory, run_directory
from .common import ROOT, VENDOR, digest, file_hash, identifier, inside, inventory, load_ref, read_json, write_json

DRIVERS = {"llama": "c4/run_llama_pretraining.py", "resnet": "pytorch-cifar/train_ddp.py",
           "glue": "glue/run_glue_no_trainer_HF.py"}


def validate_recipe(recipe):
    if recipe.get("schema_version") != 1 or recipe.get("workload") not in DRIVERS:
        raise ValueError("Unsupported recipe")
    identifier(recipe["id"])
    if not isinstance(recipe["world_size"], int) or recipe["world_size"] < 2:
        raise ValueError("Distributed validation requires world_size >= 2")
    args = recipe["args"]
    if not isinstance(args, dict):
        raise ValueError("Recipe args must be an object")
    # Prevent a full recipe from silently becoming early stopping / a smoke test.
    forbidden = {"stop_after_steps", "max_train_tokens", "continue_from", "resume_from_checkpoint",
                 "smoke-steps", "smoke_steps", "single_gpu", "output_dir", "output-dir", "save_dir",
                 "seed", "compressor", "compress_rank", "update_proj_gap", "start_compress_iter",
                 "start-compress-iter", "dry-run", "trust_remote_code", "push_to_hub"}
    if forbidden.intersection(args):
        raise ValueError(f"Reserved recipe arguments: {sorted(forbidden.intersection(args))}")
    for key, value in args.items():
        if not key or key.startswith("-") or not key.replace("-", "_").isidentifier():
            raise ValueError(f"Invalid argument: {key}")
        if not isinstance(value, (str, int, float, bool)) or value is None:
            raise ValueError(f"Expected a scalar argument: {key}")
    work = recipe["workload"]
    architecture = recipe.get("model_architecture")
    if architecture is not None and (work != "resnet" or architecture not in {"ResNet18", "ResNet50"}):
        raise ValueError("Unsupported explicit model_architecture")
    if recipe.get("kind") == "timing":
        if work != "llama" or args.get("warmup_iterations", -1) < recipe["compression_start_step"]:
            raise ValueError("Timing requires LLaMA and warmup past compression activation")
        if args.get("measured_iterations", 0) < 1 or args.get("strict_blocking_communication"):
            raise ValueError("Timing uses asynchronous DDP without blocking wrappers")
    elif work == "llama":
        if args.get("num_training_steps", 0) < 1 or args.get("gradient_accumulation") != 1:
            raise ValueError("LLaMA v1 requires full steps and accumulation=1")
        if args.get("total_batch_size") != args.get("batch_size", 0) * recipe["world_size"]:
            raise ValueError("LLaMA global batch does not match local batch * world size")
    elif work == "resnet":
        if args.get("epochs", 0) < 1:
            raise ValueError("Missing full ResNet epoch budget")
    elif args.get("num_train_epochs", 0) < 1 or args.get("gradient_accumulation_steps") != 1 or "max_train_steps" in args:
        raise ValueError("GLUE v1 requires full epochs and accumulation=1")
    if not isinstance(recipe.get("compression_start_step"), int) or recipe["compression_start_step"] < 0:
        raise ValueError("Invalid shared compression activation step")
    if not recipe.get("completion") or not recipe.get("required_paths"):
        raise ValueError("Recipe must define completion criteria and required paths")
    policy = recipe.get("checkpoint_policy", "none")
    if isinstance(policy, dict):
        destination = Path(policy.get("root", "checkpoints"))
        if (work != "llama" or recipe.get("kind") == "timing" or policy.get("mode") != "rolling"
                or type(policy.get("every_updates")) is not int or policy["every_updates"] < 1
                or destination.is_absolute() or ".." in destination.parts or str(destination) == "."):
            raise ValueError("Rolling checkpoint root must be relative to the job output in runs/; use root=checkpoints")
    elif policy != "none":
        raise ValueError("Unsupported checkpoint policy")


def driver_args(recipe, output, seed, method=None):
    args = dict(recipe["args"])
    if recipe["workload"] == "resnet":
        args.update({"output-dir": str(output), "seed": seed, "compressor": "none", "wandb-mode": "disabled"})
    else:
        args.update(output_dir=str(output), seed=seed, compressor="none")
        if method:
            common = {"compressor", "compress_rank", "update_proj_gap", "use_error_feedback",
                      "min_compression_rate", "beta_ef", "error_inherit", "scale", "sigma_type"}
            args.update({k: v for k, v in method["parameters"].items() if k in common})
        args["start_compress_iter"] = recipe["compression_start_step"]
        if recipe.get("kind") == "timing":
            args["tokenizer_path"] = recipe["tokenizer_path"]
        elif recipe["workload"] == "llama":
            args["save_dir"] = str(output / "checkpoints")
            tracking = recipe.get("tracking", {})
            args["wandb_project"] = tracking.get("project", "GreedyLore-Validation")
            args["wandb_run_name"] = tracking.get("name", output.parent.name)
    result = []
    for key, value in args.items():
        if value is False:
            continue
        result.append("--" + key)
        if value is not True:
            result.append(str(value))
    return result


def make_plan(experiment_file, output=None):
    experiment = read_json(experiment_file)
    if experiment.get("schema_version") != 1:
        raise ValueError("Expected experiment schema_version=1")
    recipe_path = load_ref(experiment["recipe"], "recipes")
    recipe = read_json(recipe_path)
    validate_recipe(recipe)
    methods = experiment.get("methods") or [experiment["method"], *experiment.get("baselines", [])]
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicate candidate/baseline")
    seeds = experiment.get("seeds", [1243])
    if not seeds or len(seeds) != len(set(seeds)) or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    execution = dict(experiment.get("execution", {}))
    # Keep venv symlinks: resolving the executable loses pyvenv.cfg discovery.
    python = str(Path(execution.get("python", sys.executable)).absolute())
    if not Path(python).is_file():
        raise ValueError("Execution Python does not exist")
    execution["python"] = python
    # Freeze only an allowlist of noncredential environment settings.
    allowed = {"HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM",
               "OMP_NUM_THREADS", "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
               "NCCL_SHM_DISABLE", "NCCL_CUMEM_HOST_ENABLE", "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS"}
    if set(execution.get("environment", {})) - allowed:
        raise ValueError("Unsupported environment key; keep credentials out of experiment files")
    tracking = execution.get("tracking", {"mode": "disabled"})
    if tracking.get("mode") not in {"disabled", "offline", "online"}:
        raise ValueError("Invalid W&B tracking mode")
    if set(tracking) - {"mode", "project", "group", "entity"}:
        raise ValueError("Unexpected tracking key; never store credentials")
    repetitions = experiment.get("repetitions", 1)
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("Invalid repetition count")
    timeout = execution.get("timeout_seconds", 0)
    if type(timeout) is not int or timeout < 0:
        raise ValueError("Invalid execution timeout")
    run_id = identifier(experiment.get("id", "validation-" + uuid.uuid4().hex[:10]))
    target = new_run_directory(run_id, output)
    if inventory(VENDOR) != read_json(ROOT / "vendor" / "provenance.json")["files"]:
        raise ValueError("Pinned baseline changed")
    target.mkdir(parents=True)
    try:
        jobs = []
        for repetition, reference in [(r, m) for r in range(repetitions) for m in methods]:
            method_path = load_ref(reference, "methods")
            for seed in seeds:
                mid = identifier(read_json(method_path)["id"])
                job_id = mid + "-s" + str(seed) + ("-r" + str(repetition + 1) if repetitions > 1 else "")
                if any(j["id"] == job_id for j in jobs):
                    raise ValueError("Method references resolved to a duplicate job ID")
                runtime = target / "jobs" / job_id / "runtime"
                runtime.parent.mkdir(parents=True)
                output_dir = runtime.parent / "output"
                job_tracking = {**tracking, "name": run_id + "-" + job_id}
                job_recipe = {**recipe, "tracking": job_tracking}
                driver, method = build_runtime(runtime, method_path, job_recipe)
                adapt_formal_runtime(runtime, job_recipe, method)
                adaptations = runtime / "formal_adaptations.json"
                if adaptations.exists():
                    shutil.copyfile(adaptations, runtime.parent / "formal_adaptations.json")
                source_meta = method_path.parent / "source_manifest.json"
                if source_meta.exists():
                    shutil.copyfile(source_meta, runtime.parent / "source_manifest.json")
                explanation = method_path.parent / "adaptation.md"
                if explanation.exists():
                    shutil.copyfile(explanation, runtime.parent / "adaptation.md")
                args = driver_args(job_recipe, output_dir, seed, method)
                command = [python, "-m", "torch.distributed.run", "--nnodes", "1",
                           "--rdzv_backend", "c10d", "--rdzv_endpoint", "127.0.0.1:0", "--rdzv_id", job_id,
                           "--nproc_per_node", str(recipe["world_size"]),
                           "--module", "glval.worker", "--driver", driver, "--", *args]
                jobs.append({"id": job_id, "method_id": mid, "source_commit": method["source_commit"],
                             "seed": seed, "repetition": repetition + 1, "tracking": job_tracking,
                             "runtime": str(runtime.relative_to(target)),
                             "output": str(output_dir.relative_to(target)), "command": command,
                             "method_hash": digest(method), "runtime_files": inventory(runtime)})
        payload = {"schema_version": 1, "id": run_id, "recipe": recipe, "recipe_hash": digest(recipe),
                   "execution": execution, "candidate": jobs[0]["method_id"],
                   "reference_methods": experiment.get("reference_methods", methods[1:]),
                   "jobs": jobs, "scaffold_files": inventory(ROOT / "glval"),
                   "baseline_commit": read_json(ROOT / "vendor" / "provenance.json")["commit"]}
        payload["plan_hash"] = digest(payload)
        write_json(target / "plan.json", payload)
        write_json(target / "experiment.json", experiment)
        write_json(target / "state.json", {"status": "planned", "jobs": {j["id"]: {"status": "planned"} for j in jobs}})
        (target / "launch.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "# Explicit execution only; provide dedicated GPU indices or UUIDs.\n"
            "cd " + shlex.quote(str(ROOT)) + "\nexec " + shlex.quote(sys.executable) +
            " -m glval run " + shlex.quote(str(target)) + ' --execute --gpus "${1:?GPU list required}"\n')
        return target
    except BaseException:
        shutil.rmtree(target)
        raise


def verify_plan(run):
    run = run_directory(run)
    plan = read_json(run / "plan.json")
    expected = plan.pop("plan_hash")
    if digest(plan) != expected:
        raise ValueError("Frozen plan was modified")
    plan["plan_hash"] = expected
    if inventory(ROOT / "glval") != plan["scaffold_files"]:
        raise ValueError("Scaffold code changed; generate a new plan")
    validate_recipe(plan["recipe"])
    for job in plan["jobs"]:
        runtime = inside(run, job["runtime"])
        inside(run, job["output"])
        if inventory(runtime) != job["runtime_files"]:
            raise ValueError(f"Runtime changed: {job['id']}; create a new plan")
        for relative in job["runtime_files"]:
            if relative.endswith(".py"):
                ast.parse((runtime / relative).read_text(), filename=relative)
    return plan


def preflight(run):
    plan = verify_plan(run)
    missing = []
    for job in plan["jobs"]:
        root = Path(run) / job["runtime"]
        for name in plan["recipe"]["required_paths"]:
            p = Path(name)
            if not (p if p.is_absolute() else root / p).exists():
                missing.append({"job": job["id"], "path": name})
    return {"status": "static_pass" if not missing else "missing_inputs", "missing_inputs": missing,
            "training_executed": False, "gpu_preflight_executed": False,
            "limitations": ["No candidate module imported", "Hook execution, refresh coverage and convergence unverified"]}
