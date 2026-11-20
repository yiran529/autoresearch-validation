"""Reproducible preparation only. Does not launch training or contact W&B."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import ast
import json
from pathlib import Path

from glval.common import ROOT, read_json, write_json
from glval.methods import create_method, check_method
from glval.plans import make_plan, preflight

HERE = Path(__file__).resolve().parent
RUN = Path('/home/wyr/coral-experiments/root-reuse-20261022/run')
SOURCES = {
    'greedylore-engineer': ('N000', '382bfae897a4db84af45e021ed1952bab7584445'),
    'n023-engineer-maxu': ('N023', '63205995a2c0380e7bab2c254a001df3306741bd'),
    'n031-engineer-maxu': ('N031', '8684c7acad3850ed029cf6039345b890cf674472'),
    'n020-engineer-maxu': ('N020', '72a57d8aa3e1c5b85fcc0e5240c353fd97c77363'),
}


def main():
    selections = {}
    for mid, (node, sha) in SOURCES.items():
        directory = ROOT / 'methods' / mid
        if not directory.exists():
            create_method(mid, RUN / 'repo', sha)
        method = read_json(directory / 'method.json')
        assert method['source_commit'] == sha
        choices = []
        for path in (RUN / '.coral/public/attempts').glob('*.json'):
            attempt = read_json(path)
            metadata = attempt.get('metadata', {})
            utility = metadata.get('utility') or {}
            if metadata.get('node_id') == node and isinstance(utility.get('value'), (int, float)):
                choices.append((utility['value'], attempt['commit_hash'], metadata['mode'], utility.get('reference')))
        if choices:
            value, parameter_sha, mode, reference = max(choices, key=lambda x: x[0])
        else:
            value, parameter_sha, mode, reference = None, 'd6009710690cd4f1985ec6f00f03a94e5acf504a', 'research', None
        # The final Engineer source retains the selected method parameters; verify exact assignments.
        import subprocess
        parameter_source = subprocess.check_output(['git', '-C', str(RUN/'repo'), 'show', parameter_sha+':comm_hooks/subspace_hook.py'], text=True)
        implementation_source = (directory / 'source/comm_hooks/subspace_hook.py').read_text()
        keys = {'matrix_approximation_rank', 'sparse_block_size', 'sparse_k', 'rejuvenation_gap', 'update_proj_gap', 'beta_ef', 'error_inherit'}
        def assignments(text):
            result = {}
            for item in ast.walk(ast.parse(text)):
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self' and target.attr in keys:
                            result[target.attr] = ast.unparse(item.value)
            return result
        before, after = assignments(parameter_source), assignments(implementation_source)
        assert before == after, (mid, before, after)
        changes = read_json(directory / 'source_manifest.json')['changes_from_base']
        review = {}
        for path in changes:
            if path.startswith('comm_hooks/'):
                review[path] = {'disposition': 'overlay', 'reason': 'Fixed Engineer communication implementation and required helper/registration code; source unchanged.'}
            elif path.startswith('c4/'):
                review[path] = {'disposition': 'not_required', 'reason': 'Reviewed differences are local tokenizer/contract metadata, optional diagnostic checkpoint and timing evidence plumbing. No optimizer/training update change. Scaffold supplies the equivalent local tokenizer, binding/result evidence and disables checkpoints.'}
            elif path.startswith('pytorch-cifar/'):
                review[path] = {'disposition': 'not_required', 'reason': 'Legacy CIFAR entry files are outside this LLaMA-only campaign; no imports from these paths.'}
            else:
                raise ValueError('Unreviewed change '+path)
        selection = {'node': node, 'implementation_sha': sha, 'parameter_sha': parameter_sha,
                     'historical_U': value, 'U_stage': mode, 'U_reference': reference,
                     'verified_state_assignments': after,
                     'note': 'Historical U read as stored, not recomputed across Research/Engineering references. Engineer implementation has identical key parameter assignments.'}
        selections[mid] = selection
        method.update(status='adapted', supported_workloads=['llama'],
                      support_notes={'llama': 'Pinned 60M config matches source; public DDP registration preserved; AdamW training loop unchanged. Real-model smoke required before formal launch.'},
                      reviewed_changes=review, selection_provenance=selection)
        method['parameters'] = {'compressor': 'top_subspace', 'compress_rank': 32,
                                'update_proj_gap': 200, 'use_error_feedback': 'ef14',
                                'min_compression_rate': 1.0, 'beta_ef': 0.0, 'error_inherit': 0}
        write_json(directory/'method.json', method)
        (directory/'adaptation.md').write_text(
            '# '+mid+'\n\nImplementation: `'+sha+'`\nParameter source: `'+parameter_sha+'`\n\n'
            'DDP gradient compression before AdamW. Public registration and method sources are reused unchanged. '
            'Communication module overlay includes all required sparse CUDA helpers; no candidate optimizer or training loop is transplanted. '
            'Optional diagnostic checkpoint branch remains disabled.\n\n'
            'Key source assignments are compared by AST against the maximum stored-U version. N020 requested rank32 is internally clamped to effective rank8 under EF14; K10 and period80 are retained. '
            'N023 keeps 8/256 and N031 keeps 128/4096. GreedyLoRE uses N000 Engineer rank32/gap200.\n\n'
            'Only LLaMA supported in this campaign. Static review is complete; runtime evidence is saved with the real-model smoke/experiment.\n')
        check_method(directory/'method.json', 'llama')
    write_json(HERE/'selections.json', selections)
    recipe = read_json(ROOT/'recipes/llama60m-adamw-full-v1.json')
    recipe.update(id='llama60m-adamw-full-online-v2', checkpoint_policy='none')
    recipe['args'].update(warmup_steps=1000, activation_checkpointing=True)
    recipe['notes'] = '8393 full updates, warmup1000, BF16, activation checkpointing ON, no model/optimizer saving. Five fixed methods, seed1243. CPU/data/algorithm settings otherwise shared.'
    write_json(ROOT/'recipes'/ (recipe['id']+'.json'), recipe)
    timing = {**recipe, 'id': 'llama60m-adamw-timing-online-v1', 'kind': 'timing',
              'evaluation_split': 'none', 'metrics': ['mean_iteration_seconds'],
              'completion': {'status':'completed','measured_steps':800,'warmup_iterations':1201,
                             'first_measured_update':1202,'last_measured_update':2001,
                             'strict_blocking_communication':False}}
    timing['args'] = {k:v for k,v in recipe['args'].items() if k not in {
        'num_training_steps','warmup_steps','scheduler','min_lr_ratio','eval_every','save_every',
        'total_batch_size','gradient_accumulation','repeat_training_data'}}
    timing['args'].update(warmup_iterations=1201, measured_iterations=800, scheduler_steps=8393,
                          lr_warmup_steps=1000, strict_blocking_communication=False, record_bucket_layout=False)
    timing['notes'] = 'Real C4, identical model/batch/precision/AdamW/activation checkpointing. Native timing window; W&B uploads after measurement. 800 updates cover two common 400-step cycles (Greedy200/N02080). Three repetitions, same four GPUs.'
    write_json(ROOT/'recipes'/(timing['id']+'.json'), timing)
    execution = {'python':'/home/wyr/greedy_lore/.venv/bin/python','timeout_seconds':0,
                 'tracking':{'mode':'online','project':'GreedyLore-Validation','group':'llama60m-engineer-maxu-20261024'},
                 'environment':{'OMP_NUM_THREADS':'4','NCCL_P2P_DISABLE':'1','NCCL_SHM_DISABLE':'0','NCCL_DEBUG':'WARN'}}
    groups = [
        ('llama60m-full-q0', recipe['id'], ['greedylore-engineer','n031-engineer-maxu','dense'], 1),
        ('llama60m-full-q1', recipe['id'], ['n023-engineer-maxu','n020-engineer-maxu'], 1),
        ('llama60m-timing', timing['id'], ['greedylore-engineer','n023-engineer-maxu','n031-engineer-maxu','n020-engineer-maxu','dense'], 3),
    ]
    plans = []
    for suffix, recipe_id, methods, reps in groups:
        experiment = {'schema_version':1,'id':suffix+'-20261024','methods':methods,
                      'reference_methods':['greedylore-engineer','dense'],'recipe':recipe_id,
                      'seeds':[1243],'repetitions':reps,'execution':execution}
        path = HERE/(suffix+'.json');write_json(path, experiment)
        run = make_plan(path)
        result = preflight(run)
        assert result['status'] == 'static_pass', result
        plans.append(str(run))
    write_json(HERE/'campaign.json', {'training_queues':[{'run':plans[0],'gpus':'0,1,2,3'},
                                                        {'run':plans[1],'gpus':'4,5,6,7'}],
                                    'timing_run':plans[2], 'selection_manifest':str(HERE/'selections.json'),
                                    'policy':'Complete all parallel non-Dense pretraining, then Dense pretraining, then timing; Dense timing last.'})
    write_json(HERE/'status.json', {'status':'prepared','training_executed':False})
    print(json.dumps({'plans':plans,'campaign':str(HERE/'campaign.json'),'selections':selections},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
