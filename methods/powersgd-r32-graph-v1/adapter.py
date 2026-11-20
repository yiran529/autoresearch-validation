"""Replay exactly the original BF16 orthogonalization kernels on each CUDA stream."""
import threading
import torch


# CUDA permits only one capture per process. The lock also makes each
# copy-in/replay/copy-out enqueue sequence atomic for threads sharing a stream.
_CAPTURE_LOCK = threading.RLock()


class GraphOrthogonalize:
    def __init__(self, original):
        self.original = original
        self.cache = {}

    def __call__(self, matrices, epsilon=0):
        if (not matrices.is_cuda or matrices.dtype != torch.bfloat16
                or not matrices.is_contiguous()):
            return self.original(matrices, epsilon)
        with _CAPTURE_LOCK, torch.cuda.device(matrices.device):
            if torch.cuda.is_current_stream_capturing():
                return self.original(matrices, epsilon)
            return self._replay(matrices, epsilon)

    def _replay(self, matrices, epsilon):
        current = torch.cuda.current_stream(matrices.device)
        key = (matrices.device, matrices.dtype, tuple(matrices.shape), tuple(matrices.stride()),
               epsilon, current.cuda_stream)
        entry = self.cache.get(key)
        if entry is None:
            static = torch.empty_like(matrices)
            static.copy_(matrices)
            capture_stream = torch.cuda.Stream(device=matrices.device)
            capture_stream.wait_stream(current)
            # Warm up the exact ATen kernels outside capture. The input tensor
            # remains untouched; staging is restored before the first replay.
            with torch.cuda.stream(capture_stream):
                self.original(static, epsilon)
            current.wait_stream(capture_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
                self.original(static, epsilon)
            current.wait_stream(capture_stream)
            entry = (static, graph, capture_stream)
            self.cache[key] = entry
        static, graph, _ = entry
        static.copy_(matrices)
        graph.replay()
        matrices.copy_(static)



class MethodAdapter:
    def validate(self, context, config):
        if context["world_size"] != 4 or config["start_compress_iter"] < 2:
            raise ValueError("Expected four ranks and stable DDP buckets before compression")

    def attach(self, model, optimizer, process_group, args, context, config):
        from comm_hooks import powerSGD_hook as implementation
        implementation._orthogonalize = GraphOrthogonalize(implementation._orthogonalize)
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
