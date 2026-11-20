"""GPU differential test and synchronized timing for the proposed execution wrapper."""
import argparse
import importlib
import importlib.util
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
    options = parser.parse_args()
    if not options.execute:
        parser.error("Explicit --execute required")
    candidate = HERE / "graph_orthogonalize.py"
    assert candidate.exists(), "CUDA graph execution wrapper is not implemented"
    with gpu_lease("4", 1) as hardware:
        os.environ["CUDA_VISIBLE_DEVICES"] = hardware[0]["uuid"]
        sys.path.insert(0, str(REFERENCE))
        import torch
        original = importlib.import_module("comm_hooks.powerSGD_hook")._orthogonalize
        spec = importlib.util.spec_from_file_location("candidate_graph", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        optimized = module.GraphOrthogonalize(original)
        checks = []
        with torch.no_grad():
            for rows, rank, epsilon, zero in [(1024, 32, 0, False), (2736, 32, 0, False),
                                             (1024, 32, 1e-8, False), (1024, 32, 0, True),
                                             (1024, 2, 0, False)]:
                g = torch.Generator(device='cuda').manual_seed(1243)
                x = torch.randn((1, rows, rank), device='cuda', dtype=torch.bfloat16, generator=g)
                if zero: x.zero_()
                expected, actual = x.clone(), x.clone()
                for iteration in range(3):
                    original(expected, epsilon)
                    optimized(actual, epsilon)
                    torch.cuda.synchronize()
                    same = torch.equal(expected.view(torch.int16), actual.view(torch.int16))
                    assert same, (rows, rank, epsilon, zero, iteration)
                checks.append({'rows': rows, 'rank': rank, 'epsilon': epsilon, 'zero': zero,
                               'iterations': 3, 'all_bf16_bits_equal': True})
            # Different host threads can reach the wrapper via Future callbacks.
            # Cover first capture on two streams and shared-stream staging reuse.
            from concurrent.futures import ThreadPoolExecutor
            for shared_stream in (False, True):
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                if shared_stream: streams[1] = streams[0]
                fixtures = []
                for _ in range(2):
                    x = torch.randn((1, 1024, 32), device='cuda', dtype=torch.bfloat16)
                    expected, actual = x.clone(), x.clone()
                    for _ in range(3): original(expected)
                    fixtures.append((expected, actual))
                torch.cuda.synchronize()
                def on_thread(index):
                    with torch.no_grad(), torch.cuda.device(0), torch.cuda.stream(streams[index]):
                        for _ in range(3): optimized(fixtures[index][1])
                    streams[index].synchronize()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(on_thread, range(2)))
                for expected, actual in fixtures:
                    assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), ('threads', shared_stream)
                checks.append({'host_threads': 2, 'shared_stream': shared_stream, 'all_bf16_bits_equal': True})
            def benchmark(fn):
                # The same mathematical orthogonalization workload as one 350M hook.
                inputs = [torch.randn((1, rows, 32), device='cuda', dtype=torch.bfloat16)
                          for rows, count in ((1024, 264), (2736, 72)) for _ in range(count)]
                torch.cuda.synchronize()
                begin = time.perf_counter()
                for x in inputs: fn(x)
                torch.cuda.synchronize()
                return time.perf_counter() - begin
            timings = []
            for _ in range(3):
                timings.append({'reference_seconds': benchmark(original), 'graph_seconds': benchmark(optimized)})
        result = {'checks': checks, 'timings': timings,
                  'cache_entries': len(optimized.cache), 'scope': 'orthogonalization only; four-rank whole-hook check still required'}
        write_json(HERE / 'orthogonalize-check.json', result)
        print(result, flush=True)


if __name__ == '__main__': main()
