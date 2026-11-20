"""Freeze four PowerSGD timing runs; never launches GPU code."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

from pathlib import Path
from glval import plans
from glval.build import build_runtime
from glval.common import ROOT, read_json, write_json, file_hash
HERE = Path(__file__).resolve().parent
REFERENCE = Path("/home/wyr/coral-experiments/greedylore-v5-dense-reuse-20261027/run/.coral/public/eval_logs/9e0844ab3f32e55cc85206082baf1a3dcef722ed/2ae551a9f79e4a9abe098e9200310ba1")
def build(target, method_path, recipe):
    driver, method = build_runtime(target, method_path, recipe)
    hook = target / "comm_hooks/powerSGD_hook.py"
    before = file_hash(hook)
    text = hook.read_text()
    old = "for tensor in tensors:\n        matrix = tensor.view(tensor.shape[0], -1)"
    assert text.count(old) == 1
    text = text.replace(old, "for tensor, parameter in zip(tensors, bucket.parameters()):\n        matrix = tensor.view(tensor.shape[0], -1)")
    assert text.count("        if compress_test[0]:") == 1
    text = text.replace("        if compress_test[0]:", "        if compress_test[0] and id(parameter) not in state.excluded_parameter_ids:")
    compile(text, str(hook), "exec")
    hook.write_text(text)
    # The exact hook code already validated by full 60M/130M training.
    ref = ROOT / "runs/powersgd-llama130m-r32-20261109/jobs/powersgd-paper-r32-s1243/runtime/comm_hooks/powerSGD_hook.py"
    assert file_hash(hook) == file_hash(ref)
    write_json(target / "powersgd_scope_bridge.json", {"before":before,"after":file_hash(hook),"reference":str(ref),
        "change":"Same named embedding/head exclusions as completed full-training validation."})
    return driver, method
def main():
    plans.build_runtime = build
    queue = []
    for repeat in (1,2):
        for precision,dtype in [("bf16","bfloat16"),("fp32","float32")]:
            reference = REFERENCE / f"efficiency/r{repeat}-350m_adamw_{precision}/config.json"
            old = read_json(reference)
            recipe = read_json(ROOT / "recipes/llama60m-adamw-timing-bsz1-local-v1.json")
            recipe.update(id=f"powersgd-350m-adamw-{precision}-timing-v1", compression_start_step=100)
            keys = ("model_config","dataset_path","batch_size","max_length","dtype","warmup_iterations",
                    "measured_iterations","scheduler_steps","lr_warmup_steps","activation_checkpointing",
                    "ddp_bucket_cap_mb","workers","lr","beta1","beta2","eps","weight_decay","grad_clipping","optimizer")
            recipe["args"] = {k:old[k] for k in keys}
            recipe["args"].update(strict_blocking_communication=False,record_bucket_layout=False)
            recipe["required_paths"] = [old["model_config"],old["dataset_path"],old["tokenizer_path"]]
            recipe["completion"].update(warmup_iterations=101,measured_steps=400,
                first_measured_update=102,last_measured_update=501)
            recipe["notes"] = "v5 Engineer timing workload; asynchronous hook; bucket layout observer disabled by scaffold. No W&B/evaluation/checkpoints. Historical Dense speedups are cross-run comparisons; topology can differ."
            rp = HERE / f"{precision}-recipe.json"
            write_json(rp,recipe)
            runid=f"powersgd-350m-{precision}-timing-r{repeat}-20261113"
            exp={"schema_version":1,"id":runid,"methods":["powersgd-paper-r32"],"reference_methods":[],
                 "recipe":str(rp),"seeds":[1243],"repetitions":1,
                 "execution":{"python":"/home/wyr/greedy_lore/.venv/bin/python","timeout_seconds":1800,
                    "tracking":{"mode":"disabled"},"environment":{"OMP_NUM_THREADS":"4",
                    "NCCL_P2P_DISABLE":"1","NCCL_SHM_DISABLE":"0","NCCL_DEBUG":"INFO"}}}
            ep=HERE/f"{precision}-r{repeat}.json";write_json(ep,exp)
            run=plans.make_plan(ep)
            assert plans.preflight(run)["status"] == "static_pass"
            queue.append(str(run))
    write_json(HERE/"queue.json",queue)
    write_json(HERE/"provenance.json",{"historical_reference":str(REFERENCE),"method":"powersgd-paper-r32",
        "historical_dense_seconds":{"bf16":.153403,"fp32":.232346},
        "differences":["GPU selection may differ: hardware and NUMA topology captured for each run.",
                       "Bucket recording wrapper disabled; core workload/window and asynchronous hook preserved."]})
    print("Four plans frozen; CPU preflight passed; no GPU execution.")
if __name__ == "__main__": main()
