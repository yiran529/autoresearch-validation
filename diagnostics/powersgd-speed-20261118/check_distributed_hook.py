"""Four-rank, real350M-shape whole-hook exact-state comparison and timing."""
import argparse
import datetime
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REFERENCE = ROOT / "runs/powersgd-llama350m-r32-full-20261118/jobs/powersgd-paper-r32-s1243/runtime"
sys.path.insert(0, str(ROOT))
from glval.common import read_json, write_json
from glval.runner import gpu_lease, stop_group


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker():
    sys.path.insert(0, str(REFERENCE))
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig
    from c4.pept_utils.modeling_llama import LlamaForCausalLM
    from graph_orthogonalize import GraphOrthogonalize
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    reference = load('comm_hooks.powersgd_reference', REFERENCE / 'comm_hooks/powerSGD_hook.py')
    candidate = load('comm_hooks.powersgd_candidate', REFERENCE / 'comm_hooks/powerSGD_hook.py')
    candidate._orthogonalize = GraphOrthogonalize(candidate._orthogonalize)
    config = AutoConfig.from_pretrained(REFERENCE / 'c4/configs/llama_350m.json', local_files_only=True)
    with torch.device('meta'):
        model = LlamaForCausalLM(config)
    named = list(model.named_parameters())
    size = sum(p.numel() for _, p in named)
    buffer = torch.empty(size, device=rank, dtype=torch.bfloat16)
    source = torch.empty_like(buffer)
    gradient = []
    offset = 0
    for _, p in named:
        gradient.append(buffer[offset:offset+p.numel()].view(p.shape))
        offset += p.numel()

    class Bucket:
        def buffer(self): return buffer
        def gradients(self): return gradient
        def parameters(self): return [p for _, p in named]
        def index(self): return 0
        def is_last(self): return True

    def make_state(module):
        class State(module.PowerSGDState): pass
        state = State(dist.group.WORLD, matrix_approximation_rank=32, start_compress_iter=2,
                      use_error_feedback='ef14', warm_start=True, min_compression_rate=2, random_seed=1243)
        state.excluded_parameter_ids = {id(p) for name, p in named if any(x in name.split('.') for x in ('embed_tokens','lm_head'))}
        return state

    left, right = make_state(reference), make_state(candidate)
    bucket = Bucket()
    generator = torch.Generator(device=rank).manual_seed(12000+rank)
    timings, checks = [], []
    with torch.no_grad():
        # Exercise dense prefix, first random-Q compression, and warm-start+EF.
        for step in range(8):
            source.normal_(mean=0, std=.01, generator=generator)
            buffer.copy_(source)
            dist.barrier()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            reference.powerSGD_hook(left, bucket).wait()
            torch.cuda.synchronize()
            old_seconds = time.perf_counter()-begin
            expected = buffer.clone()
            buffer.copy_(source)
            dist.barrier()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            candidate.powerSGD_hook(right, bucket).wait()
            torch.cuda.synchronize()
            new_seconds = time.perf_counter()-begin
            assert torch.equal(buffer.view(torch.int16), expected.view(torch.int16)), ('gradient', rank, step)
            for name in ('p_memory_dict', 'q_memory_dict', 'error_dict'):
                a, b = getattr(left, name), getattr(right, name)
                assert a.keys()==b.keys(), (name, 'keys')
                for key in a:
                    assert torch.equal(a[key].view(torch.int16), b[key].view(torch.int16)), (name, rank, step, key)
            assert left.iter == right.iter
            checks.append({'step':step+1,'gradient_and_p_q_ef_bitwise_equal':True})
            if step >= 3:
                timings.append({'reference_seconds':old_seconds,'graph_seconds':new_seconds})
            del expected
    write_json(HERE / f'hook-rank-{rank}.json', {
        'rank':rank,'checks':checks,'timings':timings,'graph_cache_entries':len(candidate._orthogonalize.cache),
        'parameters':size,'dtype':'bfloat16','torch':torch.__version__,
    })
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    options=parser.parse_args()
    if options.worker:
        worker()
        return
    if not options.execute: parser.error('Explicit --execute required')
    with gpu_lease('4,5,6,7',4) as hardware:
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in hardware),WANDB_MODE='disabled',
                   PYTHONDONTWRITEBYTECODE='1',NCCL_P2P_DISABLE='1',NCCL_SHM_DISABLE='0',NCCL_DEBUG='WARN',OMP_NUM_THREADS='4',
                   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','4',str(Path(__file__).resolve()),'--worker']
        process=None
        try:
            with (HERE/'distributed-hook.log').open('w') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=process.wait(timeout=300)
            if code: raise RuntimeError('Differential run failed; inspect distributed-hook.log')
        except BaseException:
            if process: stop_group(process)
            raise
    ranks=[read_json(HERE/f'hook-rank-{r}.json') for r in range(4)]
    old=[max(r['timings'][i]['reference_seconds'] for r in ranks) for i in range(5)]
    new=[max(r['timings'][i]['graph_seconds'] for r in ranks) for i in range(5)]
    result={'status':'passed','scope':'Four-rank hook with actual350M parameter shapes, synthetic gradients; not training',
            'hardware':hardware,'checks_per_rank':8,'all_gradient_p_q_ef_bitwise_equal':True,
            'reference_seconds':sum(old)/5,'graph_seconds':sum(new)/5,'speedup':sum(old)/sum(new),'ranks':ranks}
    write_json(HERE/'distributed-hook-check.json',result)
    print({k:v for k,v in result.items() if k!='ranks'},flush=True)


if __name__=='__main__':main()
