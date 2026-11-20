"""Runtime bridge. Imported only by an explicitly launched training process."""
import importlib.util
import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace

from .common import file_hash, read_json, write_json

_attached = False
_started = time.monotonic()
_observation = None


def _out():
    return Path(os.environ["GLVAL_OUTPUT"])


def _rank():
    return int(os.environ.get("RANK", "0"))


def register_comm_hook_for_ddp_model(model, process_group, args, optimizer=None):
    global _attached, _observation
    if _attached:
        raise RuntimeError("Method was attached more than once")
    runtime = Path(__file__).resolve().parents[1]
    method = read_json(runtime / "method.json")
    recipe = read_json(runtime / "recipe.json")
    spec = importlib.util.spec_from_file_location("candidate_adapter", runtime / "candidate_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.MethodAdapter()
    context = {"workload": recipe["workload"], "rank": _rank(),
               "world_size": int(os.environ["WORLD_SIZE"]), "recipe": recipe}
    config = dict(method["parameters"])
    # Compression activation is part of the shared workload protocol.
    config["start_compress_iter"] = recipe["compression_start_step"]
    adapter.validate(context, config)
    original_register = model.register_comm_hook
    observation = {"rank": _rank(), "method_id": method["id"], "parameters": config,
                   "hook_registrations": 0, "hook_calls": 0, "last_bucket_calls": 0,
                   "refresh_coverage": "not established; requires method-specific evidence",
                   "description": adapter.describe()}
    import torch
    observation["environment"] = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "nccl": {key: os.environ[key] for key in (
            "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_ALGO", "NCCL_PROTO") if key in os.environ}}

    def capture(state, hook):
        observation["hook_registrations"] += 1
        if observation["hook_registrations"] != 1:
            raise RuntimeError("Exactly one method hook is required")
        hook_file = Path(inspect.getfile(hook)).resolve()
        if not hook_file.is_relative_to(runtime):
            raise RuntimeError(f"Hook loaded outside frozen runtime: {hook_file}")
        observation.update(hook_module=hook.__module__, hook_name=hook.__qualname__,
                           hook_source=str(hook_file.relative_to(runtime)),
                           hook_sha256=file_hash(hook_file), state_type=type(state).__name__)
        observation["effective_state_configuration"] = {
            key: getattr(state, key) for key in ("matrix_approximation_rank", "sparse_block_size", "sparse_k",
                                                  "rejuvenation_gap", "update_proj_gap", "use_error_feedback")
            if hasattr(state, key)}
        if hasattr(state, "timing_refresh_steps") or hasattr(state, "timing_period_steps"):
            state.timing_refresh_steps = []
        if hasattr(state, "timing_enabled"):
            state.timing_enabled = True
        _observation["state"] = state

        def observed(state, bucket):
            observation["hook_calls"] += 1
            if bucket.is_last():
                observation["last_bucket_calls"] += 1
            return hook(state, bucket)

        original_register(state, observed)

    _observation = {"record": observation, "state": None}
    model.register_comm_hook = capture
    try:
        state = adapter.attach(model, optimizer, process_group, args, context, config)
        if observation["hook_registrations"] != 1:
            raise RuntimeError("Adapter did not register a DDP hook")
    finally:
        model.register_comm_hook = original_register
    _attached = True
    write_json(_out() / f"binding-rank-{_rank()}.json", observation)
    return state


def attach_resnet(model, args, optimizer):
    import torch.distributed as dist
    register_comm_hook_for_ddp_model(model, dist.group.WORLD, args, optimizer=optimizer)
    # Original CIFAR estimates are specific to its original algorithm; never reuse them.
    return SimpleNamespace(dense_bits=None, payload_bits=None)


def completion_evidence(local):
    import torch
    args = local["args"]
    recipe = read_json(Path(__file__).resolve().parents[1] / "recipe.json")
    result = {"status": "completed", "evaluation_split": recipe["evaluation_split"],
              "world_size": int(os.environ["WORLD_SIZE"])}
    if recipe["workload"] == "llama":
        result.update(completed_steps=local["update_step"], expected_steps=args.num_training_steps)
        if result["completed_steps"] != result["expected_steps"]:
            raise RuntimeError("Incomplete LLaMA training budget")
    else:
        result.update(completed_steps=local["completed_steps"], expected_steps=args.max_train_steps,
                      completed_epochs=local["epoch"] + 1, expected_epochs=args.num_train_epochs)
        if result["completed_steps"] != result["expected_steps"]:
            raise RuntimeError("Incomplete GLUE training budget")
    result["peak_memory_bytes_rank0"] = torch.cuda.max_memory_allocated()
    return result


def upload_timing():
    """Create/log the W&B run only after measurement and process-group teardown."""
    if _rank() != 0:
        return
    root = Path(__file__).resolve().parents[1]
    recipe, method = read_json(root / "recipe.json"), read_json(root / "method.json")
    tracking = recipe.get("tracking", {})
    result = read_json(_out() / "all_results.json")
    result.update(method_id=method["id"], actual_parameters=method["parameters"])
    if tracking.get("mode", "disabled") != "disabled":
        import wandb
        run = wandb.init(project=tracking.get("project", "GreedyLore-Validation"),
                         group=tracking.get("group"), entity=tracking.get("entity"),
                         name=tracking["name"], config={"recipe": recipe, "method": method},
                         dir=str(_out()), mode=tracking["mode"])
        for key in ("mean_iteration_seconds", "measured_steps", "warmup_iterations", "parameter_count"):
            run.summary[key] = result[key]
        rank_steps = result["rank_iteration_seconds"]
        for index in range(result["measured_steps"]):
            run.log({"timing/iteration_seconds_slowest_rank": rank_steps[result["slowest_rank"]][index]}, step=index)
        result["wandb_link"] = run.url
        write_json(_out() / "tracking.json", {"id": run.id, "url": run.url, "mode": tracking["mode"]})
        run.finish()
    write_json(_out() / "all_results.json", result)


def finish_runtime():
    if not _attached:
        raise RuntimeError("Training returned without attaching the candidate")
    import torch
    record, state = _observation["record"], _observation["state"]
    record.update(elapsed_seconds=time.monotonic() - _started,
                  peak_memory_bytes=torch.cuda.max_memory_allocated())
    for key in ("iter", "timing_period_steps", "timing_refresh_steps"):
        value = getattr(state, key, None)
        if value is None or isinstance(value, (int, float, str, list)):
            record[key] = value
    if record["hook_calls"] < 1:
        raise RuntimeError("No candidate DDP hook calls observed")
    write_json(_out() / f"runtime-rank-{_rank()}.json", record)
