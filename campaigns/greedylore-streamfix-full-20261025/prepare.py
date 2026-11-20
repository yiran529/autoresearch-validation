"""Freeze full-budget GreedyLoRE streamfix experiments. Preparation does not use GPUs."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

from pathlib import Path
from glval.common import ROOT,read_json,write_json
from glval.methods import check_method
from glval.plans import make_plan,preflight

HERE=Path(__file__).resolve().parent
METHOD='greedylore-engineer-streamfix'

def main():
    diagnostics=ROOT/'diagnostics/greedylore-nan-20261025'
    result=read_json(diagnostics/'model-formal-results.json')
    assert len(result['checks'])==2
    assert all(c['status']=='diagnostic_pass' and c['returncode']==0 for c in result['checks'])
    for case in result['checks']:
        assert len(case['ranks'])==4
        assert all(r['completed_steps']==1402 and r['last_loss_finite'] for r in case['ranks'])
    method_path=ROOT/'methods'/METHOD/'method.json'
    method=read_json(method_path)
    assert method['source_commit']=='8316c7eacc991173dee03f430ebec20efd99760b'
    method['support_notes']['llama']='Four-rank race reproduction and early 60M contrast passed. Real C4 60M AdamW/130M Muon each passed1402 steps with formal batch128, original LR schedules and compression start1000; full convergence unverified until these new full experiments finish.'
    write_json(method_path,method)
    with (method_path.parent/'adaptation.md').open('a') as f:
        f.write('\n2026-10-25: Real 60M AdamW and130M Muon formal-transition diagnostics both completed1402 steps on4ranks, batch128, BF16, activation checkpointing, compression start1000. All loss/gradient checks finite. These skipped validation and are not full pretraining results. New full-budget runs preserve the standard driver and evaluation.\n')
    check_method(method_path,'llama')
    execution={'python':'/home/wyr/greedy_lore/.venv/bin/python','timeout_seconds':0,
        'tracking':{'mode':'online','project':'GreedyLore-Validation','group':'greedylore-streamfix-full-20261025'},
        'environment':{'OMP_NUM_THREADS':'4','NCCL_P2P_DISABLE':'1','NCCL_SHM_DISABLE':'0','NCCL_DEBUG':'WARN'}}
    runs={}
    for model,recipe in [('130m','llama130m-muon-full-online-v1'),('60m','llama60m-adamw-full-online-v2')]:
        rid='greedylore-streamfix-'+model+'-full-20261025'
        experiment={'schema_version':1,'id':rid,'methods':[METHOD],'reference_methods':[],
                    'recipe':recipe,'seeds':[1243],'repetitions':1,'execution':execution}
        path=HERE/(rid+'.json');write_json(path,experiment)
        run=make_plan(path)
        assert preflight(run)['status']=='static_pass'
        plan=read_json(run/'plan.json')
        assert plan['recipe']['args']['batch_size']==128
        assert plan['recipe']['checkpoint_policy']=='none'
        for job in plan['jobs']:
            assert '--stop_after_steps' not in job['command']
            assert not any('diagnostic_guard' in p for p in job['runtime_files'])
            assert job['tracking']['mode']=='online'
        runs[model]=str(run)
    old=ROOT/'campaigns/llama130m-muon-engineer-maxu-20261025'
    previous=read_json(old/'first-free-timing-resumed.json')
    pid=previous['pid']
    stat=Path(f'/proc/{pid}/stat').read_text()
    ticks=stat[stat.rfind(')')+2:].split()[19]
    config=read_json(old/'campaign.json')
    write_json(HERE/'campaign.json',{'runs':runs,'allocations':{'130m':'0,1,2,3','60m':'4,5,6,7'},
        'timing_dependency':{'pid':pid,'start_ticks':ticks,
            'runs':[config['timing_run'],*config.get('additional_timing_runs',[])]},
        'method':METHOD,'source_commit':method['source_commit'],
        'policy':'130M immediately on idle0-3;60M on4-7 after existing timing controller and all planned timing finish. No retries/resume. Full budgets; original pretraining batch128 and W&B online.'})
    write_json(HERE/'preflight.json',{'status':'static_pass','runs':runs,'diagnostics':result,'training_executed':False})
    print({'prepared':runs,'source_commit':method['source_commit'],'training_executed':False})

if __name__=='__main__':main()
