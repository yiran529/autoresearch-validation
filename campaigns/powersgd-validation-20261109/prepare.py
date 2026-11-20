"""Freeze PowerSGD arms for existing validation workloads."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import ast
import runpy
from pathlib import Path
from glval import plans
from glval.build import build_runtime as native_build
from glval.common import ROOT, file_hash, read_json, write_json
from glval.methods import create_method

HERE = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "greedy_lore"
BASE = read_json(ROOT / "vendor/provenance.json")["commit"]
RESNET = ROOT / "campaigns/resnet-paper-two-workloads-20261108"
runpy.run_path(str(ROOT / "campaigns/v5-glue-qqp-mnli-20261105/prepare.py"))
original_build = plans.build_runtime
original_args = plans.driver_args

ADAPTER = """
class MethodAdapter:
    def validate(self, context, config):
        if context["world_size"] != 4 or config["start_compress_iter"] < 2:
            raise ValueError("Expected four ranks and stable DDP buckets before compression")

    def attach(self, model, optimizer, process_group, args, context, config):
        from comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook

        class ScopedPowerSGDState(PowerSGDState):
            pass

        excluded = {
            "llama": ("embed_tokens", "lm_head"),
            "glue": ("embeddings", "classifier"),
            "resnet": ("linear",),
        }[context["workload"]]
        state = ScopedPowerSGDState(
            process_group, matrix_approximation_rank=config["compress_rank"],
            start_compress_iter=config["start_compress_iter"],
            min_compression_rate=2, use_error_feedback="ef14",
            warm_start=True, random_seed=args.seed)
        state.excluded_parameter_ids = {
            id(p) for name, p in model.named_parameters()
            if any(part in name.split(".") for part in excluded)
        }
        if not state.excluded_parameter_ids:
            raise RuntimeError("Expected dense parameter exclusions were not found")
        model.register_comm_hook(state, powerSGD_hook)

    def describe(self):
        return {"implementation": "Pinned GreedyLoRE PowerSGD plus explicit dense layer exclusions",
                "error_feedback": "ef14", "warm_start": True, "min_compression_rate": 2}
"""

def add_method(rank):
    name = f"powersgd-paper-r{rank}"
    directory = ROOT / 'methods' / name
    if directory.exists():
        method = read_json(directory / 'method.json')
        assert method['status'] == 'adapted' and method['source_commit'] == BASE
        return name
    directory = create_method(name, SOURCE, BASE)
    method = read_json(directory / "method.json")
    manifest = read_json(directory / "source_manifest.json")
    method.update(status="adapted", supported_workloads=["llama","glue","resnet"],
        support_notes={w:"Native PowerSGD; workload-specific dense exclusions added before freezing runtime." for w in ("llama","glue","resnet")},
        parameters={"compressor":"powersgd","compress_rank":rank,"use_error_feedback":"ef14","min_compression_rate":2.0},
        reviewed_changes={p:{"disposition":"overlay" if p.startswith("comm_hooks/") else "not_required",
                            "reason":"Frozen communication implementation" if p.startswith("comm_hooks/") else "Common scaffold driver retained"}
                          for p in manifest["changes_from_base"]})
    write_json(directory / "method.json", method)
    (directory / "adapter.py").write_text(ADAPTER)
    (directory / "adaptation.md").write_text(
        f"PowerSGD r{rank} from pinned GreedyLoRE {BASE}. EF14, warm start, min ratio2. "
        "The campaign runtime bridge keeps embedding/head or CIFAR classifier dense. "
        "The original source snapshot is untouched. All workloads use existing frozen training recipes.\n")
    return name

def build(target, method_path, recipe):
    builder = original_build if recipe['args'].get('task_name') == 'mnli' else native_build
    driver, method = builder(target, method_path, recipe)
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
    write_json(target / "powersgd_scope_bridge.json", {"before":before,"after":file_hash(hook),
        "change":"Exclude named embedding and output modules from matrix compression; dense reduction and original EF preserved."})
    if recipe["workload"] == "resnet":
        dataset = recipe["args"]["dataset"]
        reference = RESNET / "cases" / f"resnet18-{dataset}-dense/runtime" / driver
        text = reference.read_text()
        assert "ResNet(BasicBlock, [2, 2, 2, 2]," in text
        compile(text, str(target / driver), "exec")
        (target / driver).write_text(text)
        write_json(target / "resnet_recipe_bridge.json", {"historical_driver":str(reference),"sha256":file_hash(reference)})
    return driver, method

def driver_args(recipe, output, seed, method=None):
    result = original_args(recipe, output, seed, method)
    if recipe["workload"] == "resnet":
        result[result.index("--wandb-mode") + 1] = "online"
    return result

def main():
    ranks = {rank:add_method(rank) for rank in (8,32,64)}
    plans.build_runtime = build
    plans.driver_args = driver_args
    sources = [
        ("cifar10",64,None), ("cifar100",64,None),
        *[(task,8,f"v5-glue-{task}-5epochs-20261105") for task in
          ("sst2","cola","stsb","mrpc","rte","qnli","qqp","mnli")],
        ("llama60m",32,"v5-60m-three-idle10m-20261107"),
        ("llama130m",32,"v5-final-four-130m-20261031-offline"),
    ]
    runs = []
    for label, rank, reference in sources:
        if reference:
            base = ROOT / "runs" / reference
            experiment = read_json(base / "experiment.json")
            recipe = read_json(base / "plan.json")["recipe"]
        else:
            experiment = read_json(ROOT / "campaigns/v5-60m-three-idle10m-20261107/experiment.json")
            recipe = read_json(RESNET / "cases" / f"resnet18-{label}-dense/case.json")["recipe"]
            experiment["execution"]["tracking"].update(mode="online", project="GreedyLore-CIFAR")
        recipe.update(id=f"powersgd-{label}-full-v1")
        write_json(HERE / (label + "-recipe.json"), recipe)
        experiment.update(id=f"powersgd-{label}-r{rank}-20261109", methods=[ranks[rank]],
                          reference_methods=[], recipe=str(HERE / (label + "-recipe.json")),
                          seeds=[1243], repetitions=1)
        experiment["execution"]["tracking"]["group"] = HERE.name
        write_json(HERE / (label + ".json"), experiment)
        run = plans.make_plan(HERE / (label + ".json"))
        assert not plans.preflight(run)["missing_inputs"]
        runs.append(str(run))
        print(run)
    write_json(HERE / "queue.json", runs)

if __name__ == "__main__":
    main()

