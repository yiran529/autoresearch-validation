"""Freeze the confirmed 130M recipe and existing candidate identities; no GPU work."""
import json
from pathlib import Path
from glval.common import ROOT, read_json, write_json
from glval.plans import make_plan, preflight

HERE = Path(__file__).resolve().parent
OLD = ROOT / 'campaigns/llama60m-engineer-maxu-20261024'

def main():
    full = read_json(ROOT / 'recipes/llama130m-muon-full-v1.json')
    full.update(id='llama130m-muon-full-online-v1', checkpoint_policy='none')
    full['notes'] += ' Online campaign: activation checkpointing retained; model/optimizer saving disabled.'
    write_json(ROOT / 'recipes' / (full['id']+'.json'), full)
    timing = {**full, 'id':'llama130m-muon-timing-online-v1', 'kind':'timing',
              'evaluation_split':'none', 'metrics':['mean_iteration_seconds'],
              'completion':{'status':'completed','measured_steps':800,'warmup_iterations':2201,
                            'first_measured_update':2202,'last_measured_update':3001,
                            'strict_blocking_communication':False}}
    excluded = {'num_training_steps','warmup_steps','scheduler','min_lr_ratio','eval_every',
                'save_every','total_batch_size','gradient_accumulation','repeat_training_data'}
    timing['args'] = {k:v for k,v in full['args'].items() if k not in excluded}
    timing['args'].update(warmup_iterations=2201, measured_iterations=800,
                          scheduler_steps=20000, lr_warmup_steps=2000,
                          strict_blocking_communication=False, record_bucket_layout=False)
    timing['notes'] = ('Same 130M model, Muon, BF16, local batch128 and activation checkpointing as full training; '
                       'default DDP buckets; 800 measured updates after LR warmup. Three repetitions. '
                       'W&B uploaded only after timing. No held-out quality claim from timing.')
    write_json(ROOT / 'recipes' / (timing['id']+'.json'), timing)
    selections = read_json(OLD / 'selections.json')
    for mid, selection in selections.items():
        method = read_json(ROOT / 'methods' / mid / 'method.json')
        assert method['source_commit'] == selection['implementation_sha']
        selection['campaign_note'] = 'Existing fixed implementation and effective parameters retained; no source/adapter changes.'
    # Engineer-only U for the unchanged N020 Engineer. Historical cross-stage parameter provenance remains explicit.
    selections['n020-engineer-maxu']['selected_engineer_U'] = 1.179245816626705
    write_json(HERE / 'selections.json', selections)
    execution = {'python':'/home/wyr/greedy_lore/.venv/bin/python','timeout_seconds':0,
                 'tracking':{'mode':'online','project':'GreedyLore-Validation',
                             'group':'llama130m-muon-engineer-maxu-20261025'},
                 'environment':{'OMP_NUM_THREADS':'4','NCCL_P2P_DISABLE':'1',
                                'NCCL_SHM_DISABLE':'0','NCCL_DEBUG':'WARN'}}
    groups = [
        ('llama130m-muon-full-q0-20261025',full['id'],['greedylore-engineer','n031-engineer-maxu','dense'],1),
        ('llama130m-muon-full-q1-20261025',full['id'],['n023-engineer-maxu','n020-engineer-maxu'],1),
        ('llama130m-muon-timing-20261025',timing['id'],['greedylore-engineer','n023-engineer-maxu','n031-engineer-maxu','n020-engineer-maxu','dense'],3)]
    runs = []
    for rid, recipe, methods, repetitions in groups:
        experiment = {'schema_version':1,'id':rid,'methods':methods,'recipe':recipe,
                      'reference_methods':['greedylore-engineer','dense'], 'seeds':[1243],
                      'repetitions':repetitions,'execution':execution}
        path = HERE / (rid+'.json')
        write_json(path, experiment)
        run = make_plan(path)
        result = preflight(run)
        assert result['status'] == 'static_pass', result
        runs.append(str(run))
    config = {'training_queues':[{'run':runs[0],'gpus':'0,1,2,3'},
                               {'run':runs[1],'gpus':'4,5,6,7'}],
              'timing_run':runs[2], 'selection_manifest':str(HERE/'selections.json'),
              'policy':'Parallel non-Dense full training; Dense full last; timing on GPUs0-3; Dense timing last. Failed jobs retained, no retries; continue other planned jobs.'}
    write_json(HERE/'campaign.json', config)
    write_json(HERE/'status.json', {'status':'prepared','training_executed':False})
    print(json.dumps({'campaign':str(HERE/'campaign.json'),'runs':runs,'static_preflight':'pass'}))

if __name__ == '__main__':
    main()
