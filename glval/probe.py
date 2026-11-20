"""Optional synthetic DDP integration probe. Never run during scaffold setup/tests."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    options = parser.parse_args()
    if not 1 <= options.steps <= 5000:
        raise ValueError("Probe steps must be between 1 and 5000")
    import torch
    import torch.distributed as dist
    from torch import nn
    from .bridge import register_comm_hook_for_ddp_model, finish_runtime
    from .common import read_json, write_json
    root = Path(__file__).resolve().parents[1]
    recipe = read_json(root / "recipe.json")
    args = argparse.Namespace(**recipe["args"])
    args.seed = 1243
    args.optimizer = getattr(args, "optimizer", "muon")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    # Includes 4D convolution gradients for ResNet; does not certify the actual model.
    model = (nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                           nn.Flatten(), nn.Linear(512, 128), nn.Linear(128, 10))
             if recipe["workload"] == "resnet" else
             nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 10)))
    model = model.cuda()
    if args.optimizer == "muon":
        from optimizer import build_muon_optimizer
        optimizer = build_muon_optimizer(model, lr=getattr(args, "lr", .02),
                                         scalar_lr=getattr(args, "muon_scalar_lr", .001))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(args, "lr", getattr(args, "learning_rate", .001)))
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    register_comm_hook_for_ddp_model(model, dist.group.WORLD, args, optimizer)
    torch.manual_seed(args.seed + dist.get_rank())
    shape = (4, 3, 16, 16) if recipe["workload"] == "resnet" else (4, 128)
    losses = []
    for step in range(options.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(*shape, device="cuda")).float().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite synthetic loss")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    max_delta = torch.zeros((), device="cuda")
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, 0)
        max_delta = torch.maximum(max_delta, (reference - parameter).abs().max())
    dist.all_reduce(max_delta, op=dist.ReduceOp.MAX)
    finish_runtime()
    write_json(Path(os.environ["GLVAL_OUTPUT"]) / f"probe-rank-{dist.get_rank()}.json", {
        "scope": "synthetic integration only; not workload equivalence, full training or performance evidence",
        "steps": options.steps, "losses": losses, "max_parameter_delta": max_delta.item(),
        "compression_start_step": recipe["compression_start_step"],
        "compression_phase_reached": options.steps > recipe["compression_start_step"]})
    if max_delta.item() > 1e-5:
        raise RuntimeError(f"Rank parameter divergence: {max_delta.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
