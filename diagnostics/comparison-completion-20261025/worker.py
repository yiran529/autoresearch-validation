"""Real model performance ablations using cached synthetic tokens on four or eight GPUs."""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

def main():
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--execute',action='store_true')
    options=parser.parse_args()
    if not options.execute:parser.error('Explicit --execute required')
    case=json.loads(Path(options.case).read_text());runtime=Path(case['runtime']);out=Path(case['output'])
    sys.path.insert(0,str(runtime));os.chdir(runtime)
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig
    from c4.pept_utils.modeling_llama import LlamaForCausalLM
    from c4.pept_utils.training_utils import get_scheculer
    from glval.bridge import register_comm_hook_for_ddp_model,finish_runtime
    rank=int(os.environ['RANK']);device=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device);dist.init_process_group('nccl')
    torch.manual_seed(1243);random.seed(1243)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    recipe=json.loads((runtime/'recipe.json').read_text());args=SimpleNamespace(**recipe['args'])
    args.seed=1243;args.start_compress_iter=2;args.muon_compile=False;args.muon_local_orthogonalization=case['local_muon']
    config=AutoConfig.from_pretrained(args.model_config);config.use_cache=False
    model=LlamaForCausalLM(config)
    if args.activation_checkpointing:model.gradient_checkpointing_enable()
    model.to(device=device,dtype=getattr(torch,case['dtype']));model.train()
    parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()})
    gradient_dtypes=[]
    spec=importlib.util.spec_from_file_location('timing_driver',runtime/'c4/table_v_timing.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    optimizer=driver.build_timing_optimizer(model,args)
    scheduler=get_scheculer(optimizer=optimizer,scheduler_type='cosine',num_training_steps=args.scheduler_steps,
                           warmup_steps=args.lr_warmup_steps,min_lr_ratio=.1)
    ddp_kwargs={'device_ids':[device],'output_device':device,'broadcast_buffers':False}
    # Passing25 explicitly changes PyTorch's first-bucket policy. Preserve None for the historical default.
    if case['bucket_mb'] is not None:ddp_kwargs['bucket_cap_mb']=case['bucket_mb']
    model=torch.nn.parallel.DistributedDataParallel(model,**ddp_kwargs)
    parameter_names={p:n for n,p in model.module.named_parameters()}
    active=False;stage='setup';counts=Counter();payload=Counter();bucket_meta={};sync_count=0
    original_reduce=dist.all_reduce;original_gather=dist.all_gather
    original_sync=torch.cuda.synchronize
    def counted_reduce(tensor,*a,**kw):
        if active:
            counts[(stage,'all_reduce')]+=1;payload[(stage,'all_reduce',str(tensor.dtype))]+=tensor.numel()*tensor.element_size()
        with torch.profiler.record_function('collective/all_reduce/'+stage):return original_reduce(tensor,*a,**kw)
    def counted_gather(outputs,tensor,*a,**kw):
        if active:
            counts[(stage,'all_gather')]+=1;payload[(stage,'all_gather',str(tensor.dtype))]+=tensor.numel()*tensor.element_size()
        with torch.profiler.record_function('collective/all_gather/'+stage):return original_gather(outputs,tensor,*a,**kw)
    def counted_sync(*a,**kw):
        nonlocal sync_count
        if active:sync_count+=1
        return original_sync(*a,**kw)
    dist.all_reduce=counted_reduce;dist.all_gather=counted_gather;torch.cuda.synchronize=counted_sync
    raw_register=model.register_comm_hook
    def capture(state,hook):
        def observed(state,bucket):
            if active and bucket.index() not in bucket_meta:
                bucket_meta[bucket.index()]={'bytes':bucket.buffer().numel()*bucket.buffer().element_size(),
                  'tensors':[{'name':parameter_names[p],'shape':list(g.shape),'numel':g.numel()}
                             for p,g in zip(bucket.parameters(),bucket.gradients())]}
            return hook(state,bucket)
        raw_register(state,observed)
    model.register_comm_hook=capture
    register_comm_hook_for_ddp_model(model,dist.group.WORLD,args,optimizer)
    torch.manual_seed(1243+rank)
    tokens=torch.randint(3,config.vocab_size,(case['batch_size'],256),device=device)[:,:case['sequence_length']].contiguous()
    mask=torch.ones_like(tokens)
    parameters=[p for p in model.parameters() if p.requires_grad]
    times=[];phase_times=Counter();losses=[];profiler=None;elapsed=None
    profiling=case['profile']
    for step in range(case['warmup']+case['measured']+(case['profile_steps'] if profiling else 0)):
        if step==case['warmup']:
            dist.barrier();original_sync();active=True;window=time.perf_counter()
        if step==case['warmup']+case['measured']:
            elapsed=time.perf_counter()-window;active=False
            dist.barrier();original_sync()
            if profiling and rank==0:
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__()
        beginning=time.perf_counter();last=beginning
        stage='forward'
        with torch.profiler.record_function('stage/forward'):
            loss=model(input_ids=tokens,attention_mask=mask,labels=tokens).loss
        now=time.perf_counter()
        if active:phase_times['forward_cpu_wall']+=now-last
        last=now;stage='backward'
        with torch.profiler.record_function('stage/backward'):loss.backward()
        if step==case['warmup']:
            gradient_dtypes=sorted({str(p.grad.dtype) if p.grad is not None else 'None' for p in parameters})
        now=time.perf_counter()
        if active:phase_times['backward_cpu_wall']+=now-last
        last=now;stage='optimizer'
        with torch.profiler.record_function('stage/optimizer'):
            torch.nn.utils.clip_grad_norm_(parameters,args.grad_clipping)
            optimizer.step();scheduler.step();optimizer.zero_grad()
        now=time.perf_counter()
        if active:phase_times['optimizer_cpu_wall']+=now-last
        last=now;stage='iteration_end_sync';torch.cuda.synchronize()
        now=time.perf_counter()
        if active:
            phase_times['end_sync_cpu_wall']+=now-last;times.append(now-beginning);losses.append(loss.detach())
    if elapsed is None:elapsed=time.perf_counter()-window
    active=False
    if profiler is not None:
        profiler.__exit__(None,None,None)
        events=[]
        for event in profiler.key_averages():
            events.append({'name':event.key,'count':event.count,'self_cpu_us':event.self_cpu_time_total,
                           'total_cpu_us':event.cpu_time_total,'device_us':getattr(event,'device_time_total',getattr(event,'cuda_time_total',0))})
        (out/'profile-summary.json').write_text(json.dumps(events,indent=2)+'\n')
        profiler.export_chrome_trace(str(out/'trace.json'))
    local={'rank':rank,'elapsed':elapsed,
           'parameter_dtypes':parameter_dtypes,'gradient_dtypes':gradient_dtypes,
           'optimizer_state_dtypes':sorted({str(value.dtype) for state in optimizer.state.values() for value in state.values() if torch.is_tensor(value)}),
           'input_shape':list(tokens.shape),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
           'float32_matmul_precision':torch.get_float32_matmul_precision(),'steps':times,'losses':torch.stack(losses).float().cpu().tolist(),
           'calls':[{'stage':k[0],'op':k[1],'count':v/case['measured']} for k,v in counts.items()],
           'payload':[{'stage':k[0],'op':k[1],'dtype':k[2],'local_input_bytes':v/case['measured']} for k,v in payload.items()],
           'syncs_per_step':sync_count/case['measured'],
           'phase_cpu_wall_ms':{k:v/case['measured']*1000 for k,v in phase_times.items()},'buckets':bucket_meta}
    gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,local)
    if rank==0:
        import math
        assert all(len(r['steps'])==case['measured'] and all(math.isfinite(x) for x in r['losses']) for r in gathered)
        result={'case':case['id'],'mean_step_ms':max(r['elapsed'] for r in gathered)/case['measured']*1000,
                'ranks':gathered,'scope':case['scope'],'world_size':dist.get_world_size(),'global_batch':case['batch_size']*dist.get_world_size(),'peak_memory_bytes_rank0':torch.cuda.max_memory_allocated(),'p2p_access':[[bool(torch.cuda.can_device_access_peer(i,j)) if i!=j else True for j in range(case['world_size'])] for i in range(case['world_size'])]}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'case':case['id'],'mean_step_ms':result['mean_step_ms']}),flush=True)
    finish_runtime();dist.destroy_process_group()

if __name__=='__main__':main()
