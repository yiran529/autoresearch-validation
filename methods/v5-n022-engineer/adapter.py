"""Thin bridge to the candidate's public registration function. No algorithm rewrite."""
import argparse


class MethodAdapter:
    def validate(self, context, config):
        if context["world_size"] < 2:
            raise ValueError("This adapter requires distributed training")
        if config["compressor"] != "none":
            if config.get("compress_rank", 0) < 1 or config.get("update_proj_gap", 0) < 1:
                raise ValueError("Invalid low-rank compression parameters")

    def attach(self, model, optimizer, process_group, args, context, config):
        from comm_hooks.utils import add_comm_hook_args, register_comm_hook_for_ddp_model
        parser = argparse.ArgumentParser(add_help=False)
        add_comm_hook_args(parser)
        merged = vars(parser.parse_args([]))
        merged.update(vars(args))
        merged.setdefault("optimizer", "muon")
        merged.update(config)
        return register_comm_hook_for_ddp_model(
            model, process_group, argparse.Namespace(**merged), optimizer=optimizer)

    def describe(self):
        return {"registration": "comm_hooks.utils.register_comm_hook_for_ddp_model",
                "implementation": "frozen runtime comm_hooks; base optimizer and training loop",
                "communication_volume": "unavailable; no generic byte estimator"}
