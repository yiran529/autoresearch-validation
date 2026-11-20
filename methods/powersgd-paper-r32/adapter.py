
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
