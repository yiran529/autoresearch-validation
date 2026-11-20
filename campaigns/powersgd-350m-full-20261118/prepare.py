"""Freeze the approved 350M PowerSGD experiment; CPU-only, no training."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import copy
from pathlib import Path

from glval import plans
from glval.build import build_runtime
from glval.common import ROOT, file_hash, read_json, write_json

HERE = Path(__file__).resolve().parent
REFERENCE = ROOT / "runs/powersgd-llama130m-r32-20261109/jobs/powersgd-paper-r32-s1243/runtime"
BASE_RECIPE = ROOT / "recipes/llama350m-muon-full-offline-v5.json"
N33 = ROOT / "runs/v5-350m-n033-n022-rolling20k-20261101"


def validate_contract(recipe, execution, phase):
    if execution["tracking"]["mode"] != "disabled":
        raise ValueError("W&B must be disabled for this campaign")
    if recipe.get("checkpoint_policy") != "none":
        raise ValueError("Model and optimizer checkpoint saving must be disabled")
    expected = read_json(BASE_RECIPE)
    if phase == "smoke":
        expected["args"]["num_training_steps"] = 8
        expected["compression_start_step"] = 2
        expected["completion"].update(update_step=8, completed_steps=8, expected_steps=8)
    elif phase != "full":
        raise ValueError("Unknown campaign phase")
    if recipe["args"] != expected["args"]:
        raise ValueError("Training settings differ from the approved 350M protocol")
    if recipe["compression_start_step"] != expected["compression_start_step"]:
        raise ValueError("Unexpected compression start for this phase")
    for key in ("workload", "world_size", "tokenizer_path", "required_paths",
                "evaluation_split", "completion", "metrics"):
        if recipe[key] != expected[key]:
            raise ValueError("Training settings differ: " + key)


def build(target, method_path, recipe):
    driver, method = build_runtime(target, method_path, recipe)
    hook = target / "comm_hooks/powerSGD_hook.py"
    before = file_hash(hook)
    text = hook.read_text()
    old = "for tensor in tensors:\n        matrix = tensor.view(tensor.shape[0], -1)"
    assert text.count(old) == 1
    text = text.replace(old, "for tensor, parameter in zip(tensors, bucket.parameters()):\n        matrix = tensor.view(tensor.shape[0], -1)")
    old = "        if compress_test[0]:"
    assert text.count(old) == 1
    text = text.replace(old, "        if compress_test[0] and id(parameter) not in state.excluded_parameter_ids:")
    compile(text, str(hook), "exec")
    hook.write_text(text)
    for relative in ("comm_hooks/powerSGD_hook.py", "candidate_adapter.py", driver):
        if file_hash(target / relative) != file_hash(REFERENCE / relative):
            raise ValueError("Runtime differs from completed 130M PowerSGD: " + relative)
    write_json(target / "powersgd_scope_bridge.json", {
        "before": before, "after": file_hash(hook), "reference": str(REFERENCE),
        "change": "Identical to verified 130M hook: embedding/head dense; EF14, warm start, rank32.",
    })
    return driver, method


def main():
    original_build = plans.build_runtime
    queue = []
    try:
        plans.build_runtime = build
        for phase in ("smoke", "full"):
            recipe = read_json(BASE_RECIPE)
            recipe["id"] = f"powersgd-llama350m-r32-{phase}-v1-20261118"
            recipe["notes"] = (
                "Approved 350M Muon protocol; W&B disabled; no model/optimizer checkpoints. "
                "PowerSGD scope and algorithm identical to completed 130M validation."
            )
            if phase == "smoke":
                recipe["args"]["num_training_steps"] = 8
                recipe["compression_start_step"] = 2
                recipe["completion"].update(update_step=8, completed_steps=8, expected_steps=8)
                recipe["notes"] += " SMOKE ONLY: formal batch; compressed path and memory, not convergence."
            execution = copy.deepcopy(read_json(N33 / "plan.json")["execution"])
            execution["tracking"] = {"mode": "disabled"}
            execution["timeout_seconds"] = 1800 if phase == "smoke" else 0
            validate_contract(recipe, execution, phase)
            rp = HERE / f"{phase}-recipe.json"
            ep = HERE / f"{phase}.json"
            write_json(rp, recipe)
            write_json(ep, {
                "schema_version": 1, "id": f"powersgd-llama350m-r32-{phase}-20261118",
                "methods": ["powersgd-paper-r32"], "reference_methods": [],
                "recipe": str(rp), "seeds": [1243], "repetitions": 1, "execution": execution,
            })
            run = plans.make_plan(ep)
            if plans.preflight(run)["status"] != "static_pass":
                raise ValueError("CPU preflight failed: " + str(run))
            queue.append(str(run))
            print(run)
    finally:
        plans.build_runtime = original_build
    write_json(HERE / "queue.json", queue)
    write_json(HERE / "provenance.json", {
        "n33_reference": str(N33), "powersgd_runtime_reference": str(REFERENCE),
        "historical_references": [str(p) for prefix in ("CM250-", "CM251-", "CM252-")
                                  for p in sorted(Path("/home/wyr/greedy_lore/outputs").glob(prefix + "*"))
                                  if p.is_dir()],
        "approved_differences": ["PowerSGD communication method", "W&B disabled",
                                 "No model or optimizer checkpoints"],
        "scope_difference": "N33 compresses full buckets including embedding/head; PowerSGD uses its existing dense exclusions.",
        "gpus": "4,5,6,7", "plan_hashes": {
            str(run): read_json(Path(run) / "plan.json")["plan_hash"] for run in queue},
    })
    print("Both plans frozen; CPU preflight passed; no GPU execution.")


if __name__ == "__main__":
    main()
