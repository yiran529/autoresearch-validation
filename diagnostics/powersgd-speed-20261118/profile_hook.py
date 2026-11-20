"""Bounded real-shape PowerSGD hook profile; no model training or downloads."""
import argparse
from collections import Counter
import importlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REFERENCE = ROOT / "runs/powersgd-llama350m-r32-full-20261118/jobs/powersgd-paper-r32-s1243/runtime"
sys.path.insert(0, str(ROOT))
from glval.common import write_json
from glval.runner import gpu_lease


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit --execute required")
    with gpu_lease("4", 1) as hardware:
        os.environ.update(CUDA_VISIBLE_DEVICES=hardware[0]["uuid"], WANDB_MODE="disabled",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        sys.path.insert(0, str(REFERENCE))
        import torch
        import torch.distributed as dist
        from transformers import AutoConfig
        from c4.pept_utils.modeling_llama import LlamaForCausalLM
        module = importlib.import_module("comm_hooks.powerSGD_hook")
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29681", rank=0, world_size=1)
        config = AutoConfig.from_pretrained(REFERENCE / "c4/configs/llama_350m.json", local_files_only=True)
        with torch.device("meta"):
            model = LlamaForCausalLM(config)
        named = list(model.named_parameters())
        torch.manual_seed(1243)
        buffer = torch.randn(sum(p.numel() for _, p in named), device="cuda", dtype=torch.bfloat16)
        gradients = []
        offset = 0
        for _, p in named:
            gradients.append(buffer[offset:offset+p.numel()].view(p.shape))
            offset += p.numel()

        class Bucket:
            def buffer(self): return buffer
            def gradients(self): return gradients
            def parameters(self): return [p for _, p in named]
            def index(self): return 0
            def is_last(self): return True

        class State(module.PowerSGDState):
            pass

        state = State(dist.group.WORLD, matrix_approximation_rank=32, start_compress_iter=2,
                      min_compression_rate=2, use_error_feedback="ef14", warm_start=True, random_seed=1243)
        state.excluded_parameter_ids = {id(p) for name, p in named if any(x in name.split('.') for x in ('embed_tokens', 'lm_head'))}
        state.iter = 2
        bucket = Bucket()
        original = module._orthogonalize
        calls = []

        def measured(tensor, epsilon=0):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            begin = time.perf_counter()
            result = original(tensor, epsilon)
            cpu = time.perf_counter() - begin
            b.record()
            calls.append((a, b, cpu, tuple(tensor.shape)))
            return result

        with torch.no_grad():
            for _ in range(2):
                module.powerSGD_hook(state, bucket).wait()
            torch.cuda.synchronize()
            module._orthogonalize = measured
            begin = time.perf_counter()
            module.powerSGD_hook(state, bucket).wait()
            torch.cuda.synchronize()
            total = time.perf_counter() - begin
        result = {
            "scope": "Single-rank real350M shapes, full single BF16 bucket; hook only, not training or four-rank timing",
            "hardware": hardware, "torch": torch.__version__, "cuda": torch.version.cuda,
            "hook_wall_seconds": total,
            "orthogonalize_calls": len(calls),
            "orthogonalize_gpu_seconds": sum(a.elapsed_time(b) for a, b, _, _ in calls) / 1000,
            "orthogonalize_host_seconds": sum(cpu for _, _, cpu, _ in calls),
            "orthogonalize_shapes": {str(k): v for k, v in Counter(shape for _, _, _, shape in calls).items()},
            "model_parameters": offset,
            "parameter_shapes": [{"name": name, "shape": list(p.shape)} for name, p in named],
        }
        write_json(HERE / "profile.json", result)
        print({k: v for k, v in result.items() if k != "parameter_shapes"}, flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
