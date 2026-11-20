"""Prepare the two CIFAR workloads using the historical AdamW driver and Appendix F."""
import ast
import runpy
import shutil
from pathlib import Path

from glval.build import build_runtime
from glval.common import ROOT, file_hash, inventory, read_json, write_json

HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "campaigns/resnet50-adamw-v5-n33-20261108"
patch_adamw_driver = runpy.run_path(str(PREVIOUS / "prepare.py"))["patch_driver"]

def build_cases():
    cases = []
    datasets = (
        ("cifar10", "ResNet18", "BasicBlock", [2, 2, 2, 2], 11173962, 0.005, 500, 750),
        ("cifar100", "ResNet18", "BasicBlock", [2, 2, 2, 2], 11220132, 0.0005, 4000, 1200),
    )
    for dataset, model, block, blocks, params, lr, start, gap in datasets:
        greedy = f"greedylore-streamfix-paper-{dataset}-v1"
        path = ROOT / "methods" / greedy
        shutil.copytree(ROOT / "methods/greedylore-engineer-streamfix-resnet-v1", path)
        method = read_json(path / "method.json")
        method["id"] = greedy
        method["parameters"].update(compress_rank=64, update_proj_gap=gap, min_compression_rate=2.0)
        write_json(path / "method.json", method)
        with (path / "adaptation.md").open("a") as note:
            note.write(f"\nCIFAR paper-arm parameters: rank64, refresh{gap}, min_compression_rate2 from historical paper arm. Original streamfix source unchanged.\n")
        recipe = read_json(PREVIOUS / "cases.json")["full_recipe"]
        recipe.update(
            id=f"{model.lower()}-{dataset}-paper-adamw-20261108",
            model_architecture=model,
            compression_start_step=start,
            tracking={"mode":"online","project":"GreedyLore-CIFAR","group":HERE.name},
            notes="Appendix F CIFAR budget, LR, compression prefix and GreedyLoRE rank/gap. Historical scaffold AdamW/WD/FP32 and data transforms."
        )
        recipe["args"].update(dataset=dataset, lr=lr)
        if dataset == "cifar100":
            data = Path(recipe["args"]["data-dir"]) / "cifar-100-python"
            recipe["required_paths"] = [str(data / split) for split in ("train", "test", "meta")]
        assert all(Path(p).is_file() for p in recipe["required_paths"])
        for name in (greedy, "v5-n033-research-resnet-v1", "dense"):
            case_id = f"{model.lower()}-{dataset}-{name}"
            directory = HERE / "cases" / case_id
            runtime = directory / "runtime"
            driver, method = build_runtime(runtime, ROOT / "methods" / name / "method.json", recipe)
            patch = patch_adamw_driver(runtime)
            driver_path = runtime / driver
            source = driver_path.read_text()
            if model == "ResNet18":
                source = source.replace("Bottleneck", "BasicBlock").replace("[3, 4, 6, 3]", "[2, 2, 2, 2]").replace("resnet_blocks=[3,4,6,3]", "resnet_blocks=[2,2,2,2]").replace("ResNet50", "ResNet18")
            if name.startswith("v5-n033"):
                source = source.replace("choices=['none', 'top_subspace']", "choices=['none', 'top_subspace', 'sparse_q4_ef']")
                source = source.replace("uncompressed_names=['linear']", "uncompressed_names=[]")
            ast.parse(source)
            driver_path.write_text(source)
            patch.update(after=file_hash(driver_path), architecture=f"{model}: {block}{blocks}", optimizer=f"AdamW lr={lr}, betas=(0.9,0.999), eps=1e-8, WD=0.0005")
            method_args = method["parameters"]
            args = dict(recipe["args"], seed=1243)
            args.update({
                "output-dir":str(directory / "output"),
                "compressor":method_args["compressor"],
                "compress-rank":method_args.get("compress_rank",32),
                "update-proj-gap":method_args.get("update_proj_gap",200),
                "min-compression-rate":method_args.get("min_compression_rate",1.0),
                "start-compress-iter":start,
                "wandb-mode":"online","wandb-project":"GreedyLore-CIFAR",
            })
            cli = [item for key,value in args.items() for item in ("--"+key,str(value))]
            case = dict(id=case_id, method_id=name, phase="full", source_commit=method["source_commit"],
                runtime=str(runtime), output=str(directory / "output"), driver=driver, args=cli,
                recipe=recipe, seed=1243, expected_model=model, expected_parameter_count=params,
                expected_block=block, expected_blocks=blocks, runtime_files=inventory(runtime), driver_patch=patch)
            write_json(directory / "case.json", case)
            cases.append(case)
    write_json(HERE / "cases.json", {"cases":cases})
    write_json(HERE / "status.json", {"status":"prepared","gpu_execution":False})
    print([(case["id"], case["recipe"]["args"]["lr"]) for case in cases])

if __name__ == "__main__":
    build_cases()

