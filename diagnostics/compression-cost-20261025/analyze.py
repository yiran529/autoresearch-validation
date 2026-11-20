"""Aggregate archived timing, controlled ablations and profiler evidence. CPU only."""
from pathlib import Path
import json
import statistics
from glval.common import ROOT,read_json,write_json

HERE=Path(__file__).resolve().parent

def archived(model):
    run=ROOT/'runs'/('llama'+model+'-timing-bsz1-local-20261025')
    plan=read_json(run/'plan.json');result={}
    for method in dict.fromkeys(j['method_id'] for j in plan['jobs']):
        regular=[];events=[];means=[]
        for job in plan['jobs']:
            if job['method_id']!=method:continue
            d=read_json(run/job['output']/'all_results.json');means.append(d['mean_iteration_seconds'])
            for k,t in enumerate(d['rank_iteration_seconds'][d['slowest_rank']]):
                step=d['first_measured_update']+k
                event=(method=='greedylore-engineer' and (step-1001)%200==0) or (method=='n020-engineer-maxu' and (step-1001)%10==0)
                (events if event else regular).append(t)
        result[method]={'mean_ms':statistics.mean(means)*1000,
            'regular_ms':statistics.mean(regular)*1000,
            'event_ms':statistics.mean(events)*1000 if events else None,'event_count':len(events),
            'event_excess_amortized_ms':(sum(events)-len(events)*statistics.mean(regular))/(len(events)+len(regular))*1000 if events else 0}
    return result

def profile(path,steps):
    if not path.exists():return None
    events=read_json(path)
    def summed(name,key):return sum(e[key] for e in events if e['name']==name)
    return {'cuda_kernel_launches_per_step':summed('cudaLaunchKernel','count')/steps,
            'cuda_launch_cpu_ms_per_step':summed('cudaLaunchKernel','self_cpu_us')/steps/1000,
            'device_synchronize_cpu_ms_per_step':summed('cudaDeviceSynchronize','self_cpu_us')/steps/1000,
            'nccl_allreduce_device_ms_per_step':summed('nccl:all_reduce','device_us')/steps/1000,
            'nccl_allgather_device_ms_per_step':summed('nccl:all_gather','device_us')/steps/1000,
            'topk_device_ms_per_step':summed('aten::topk','device_us')/steps/1000,
            'qr_device_ms_per_step':summed('aten::linalg_qr','device_us')/steps/1000,
            'qr_calls_per_step':summed('aten::linalg_qr','count')/steps,
            'warning':'Profiled3steps only; profiler overhead affects time, do not sum these into unprofiled wall time.'}

def main():
    cases=read_json(HERE/'cases.json')['cases'];controlled=[];missing=[]
    for case in cases:
        out=Path(case['output']);path=out/'result.json'
        process=out/'process.json'
        if not path.exists() or not process.exists() or read_json(process).get('returncode')!=0:
            missing.append(case['id']);continue
        d=read_json(path);r=d['ranks'][0]
        gradients=[x for x in r['payload'] if x['stage']=='backward']
        optimizer=[x for x in r['payload'] if x['stage']=='optimizer']
        controlled.append({'id':case['id'],'mean_ms':d['mean_step_ms'],'buckets':len(r['buckets']),
             'syncs_per_step':r['syncs_per_step'],'calls':r['calls'],'gradient_payload':gradients,
             'gradient_local_input_MiB':sum(x['local_input_bytes'] for x in gradients)/2**20,
             'optimizer_payload':optimizer,
             'optimizer_local_input_MiB':sum(x['local_input_bytes'] for x in optimizer)/2**20,
             'phase_cpu_wall_ms':r['phase_cpu_wall_ms'],'p2p_access':d['p2p_access'],
             'profile':profile(out/'profile-summary.json',case['profile_steps']),
             'scope':case['scope'],'source_runtime_hashes_recorded':True})
    payload={'status':'complete' if not missing else 'partial','missing':missing,
             'archived':{m:archived(m) for m in ['60m','130m']},'controlled':controlled,
             'setting_difference':{'autoresearch':{'model':'130M','optimizer':'AdamW','activation_checkpointing':False,
                 'bucket_cap_mb':1024,'observed_buckets':1,'compression_start':100},
                 'current':{'60M_optimizer':'AdamW','130M_optimizer':'Muon','activation_checkpointing':True,
                 'bucket_cap_mb':None,'observed_60M_buckets':5,'observed_130M_buckets':8,'compression_start':1000}},
             'limitations':['Cached synthetic tokens, early compression, 30 measured steady-state steps per controlled case; not final800-step timing.',
                 'Profile intervals are separate from wall-time measurement and perturb execution.',
                 'No-DDP is a diagnostic bound, not a valid training algorithm.',
                 'Measurements may share CPU/host-memory resources with130M full training on GPUs0-3.',
                 'Per-call local payload is not physical NIC/PCIe wire traffic.']}
    write_json(HERE/'analysis.json',payload)
    print(json.dumps({'status':payload['status'],'completed':len(controlled),'missing':missing,
                      'wall_ms':{c['id']:round(c['mean_ms'],3) for c in controlled}},ensure_ascii=False))

if __name__=='__main__':main()
