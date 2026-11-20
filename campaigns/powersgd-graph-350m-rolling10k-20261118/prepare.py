"""Freeze a fresh optimized 350M run with rolling10k saves; CPU-only preparation."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import copy
from pathlib import Path
import shutil

from glval import plans
from glval.build import build_runtime
from glval.common import ROOT, file_hash, read_json, write_json

HERE = Path(__file__).resolve().parent
BASE_RECIPE = ROOT / 'recipes/llama350m-muon-full-offline-v5.json'
N33 = ROOT / 'runs/v5-350m-n033-n022-rolling20k-20261101'
METHOD = 'powersgd-r32-graph-v1'
CHECKPOINT_ROOT = Path('/dev/shm/wyr_tmp/greedylore-validation') / HERE.name
ADAPTER_HASH = '387e057f3ffa41717a9ad54a263848024eb372f65524bcb9cf483376dc107b62'
HOOK_HASH = 'ba49b4f33dcc921d634302f8817d570c8605dd44c8bd4b856a6a33536f83f857'


def definition(phase):
    if phase not in {'smoke', 'full'}:
        raise ValueError('Unknown phase')
    recipe = read_json(BASE_RECIPE)
    recipe['id'] = f'powersgd-graph-350m-rolling10k-{phase}-20261118'
    interval = 4 if phase == 'smoke' else 10000
    recipe['args']['save_every'] = interval
    recipe['checkpoint_policy'] = {'mode': 'rolling', 'every_updates': interval,
                                   'root': str(CHECKPOINT_ROOT / phase)}
    recipe['notes'] = ('Fresh optimized PowerSGD; original350M Muon protocol; W&B disabled. '
                       'All-rank optimizer/scheduler/PowerSGD slots/RNG plus rank0 model; latest only. '
                       'Data iterator and reducer layout are not serialized; exact stream resume is unsupported.')
    execution = copy.deepcopy(read_json(N33 / 'plan.json')['execution'])
    execution['tracking'] = {'mode': 'disabled'}
    execution['timeout_seconds'] = 1800 if phase == 'smoke' else 0
    if phase == 'smoke':
        recipe['args']['num_training_steps'] = 8
        recipe['compression_start_step'] = 2
        recipe['completion'].update(update_step=8, completed_steps=8, expected_steps=8)
        recipe['notes'] += ' SMOKE: eight updates, saves4/8, not convergence evidence.'
    return recipe, execution


def validate_contract(recipe, execution, phase):
    expected, expected_execution = definition(phase)
    if recipe != expected:
        raise ValueError('Training settings or rolling checkpoint policy changed')
    if execution != expected_execution:
        raise ValueError('Execution settings or W&B disabled contract changed')


def build(target, method_path, recipe):
    driver, method = build_runtime(target, method_path, recipe)
    if method['id'] != METHOD or file_hash(target / 'candidate_adapter.py') != ADAPTER_HASH:
        raise ValueError('Only the verified optimized PowerSGD method is authorized')
    hook = target / 'comm_hooks/powerSGD_hook.py'
    text = hook.read_text()
    old = 'for tensor in tensors:\n        matrix = tensor.view(tensor.shape[0], -1)'
    assert text.count(old) == 1
    text = text.replace(old, 'for tensor, parameter in zip(tensors, bucket.parameters()):\n        matrix = tensor.view(tensor.shape[0], -1)')
    old = '        if compress_test[0]:'
    assert text.count(old) == 1
    text = text.replace(old, '        if compress_test[0] and id(parameter) not in state.excluded_parameter_ids:')
    hook.write_text(text)
    if file_hash(hook) != HOOK_HASH:
        raise ValueError('PowerSGD scope bridge differs from verified optimized runtime')
    # Local adaptation only: preserve the global scaffold and all historical runtimes.
    checkpoint = target / 'glval/checkpoints.py'
    text = checkpoint.read_text()
    old = "hook={k:cpu(v) for k,v in vars(state).items() if k not in {'process_group','param_to_name','param_state'}}"
    assert text.count(old) == 1
    text = text.replace(old, 'from .checkpoint_state import snapshot_hook\n        hook=snapshot_hook(state,cpu)')
    checkpoint.write_text(text)
    shutil.copyfile(HERE / 'checkpoint_state.py', target / 'glval/checkpoint_state.py')
    return driver, method


def main():
    original = plans.build_runtime
    queue = []
    try:
        plans.build_runtime = build
        for phase in ('smoke', 'full'):
            recipe, execution = definition(phase)
            validate_contract(recipe, execution, phase)
            rp = HERE / (phase + '-recipe.json')
            ep = HERE / (phase + '.json')
            write_json(rp, recipe)
            write_json(ep, {'schema_version': 1, 'id': recipe['id'], 'methods': [METHOD],
                            'reference_methods': [], 'recipe': str(rp), 'seeds': [1243],
                            'repetitions': 1, 'execution': execution})
            run = plans.make_plan(ep)
            if plans.preflight(run)['status'] != 'static_pass':
                raise ValueError('CPU preflight failed')
            queue.append(str(run))
            print(run)
    finally:
        plans.build_runtime = original
    write_json(HERE / 'queue.json', queue)
    write_json(HERE / 'provenance.json', {
        'plan_hashes': {run: read_json(Path(run) / 'plan.json')['plan_hash'] for run in queue},
        'campaign_files': {name: file_hash(HERE / name) for name in
                           ('prepare.py', 'controller.py', 'checkpoint_state.py', 'verify_checkpoint.py')},
        'timing_manifest_hash': read_json(HERE.parent / 'powersgd-graph-timing-20261118/manifest.json')['manifest_hash'],
        'gpus': '4,5,6,7', 'method_id': METHOD,
        'n33_reference': str(N33), 'checkpoint_root': str(CHECKPOINT_ROOT),
        'prior_run': str(ROOT / 'runs/powersgd-llama350m-r32-full-20261118'),
        'prior_run_status': 'interrupted; no checkpoint; new attempt starts at0',
    })
    print('Two new plans frozen; no GPU execution.')


if __name__ == '__main__':
    main()
