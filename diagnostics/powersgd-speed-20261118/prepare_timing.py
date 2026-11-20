"""Freeze a bounded paired real-model timing run at formal per-device batch128."""
from pathlib import Path
from glval import plans
from glval.build import build_runtime
from glval.common import ROOT, read_json, write_json, file_hash

HERE = Path(__file__).resolve().parent


def build(target, method_path, recipe):
    driver, method = build_runtime(target, method_path, recipe)
    hook = target / 'comm_hooks/powerSGD_hook.py'
    text = hook.read_text()
    old = 'for tensor in tensors:\n        matrix = tensor.view(tensor.shape[0], -1)'
    assert text.count(old) == 1
    text = text.replace(old, 'for tensor, parameter in zip(tensors, bucket.parameters()):\n        matrix = tensor.view(tensor.shape[0], -1)')
    assert text.count('        if compress_test[0]:') == 1
    text = text.replace('        if compress_test[0]:', '        if compress_test[0] and id(parameter) not in state.excluded_parameter_ids:')
    compile(text, str(hook), 'exec')
    hook.write_text(text)
    reference = ROOT / 'runs/powersgd-llama130m-r32-20261109/jobs/powersgd-paper-r32-s1243/runtime/comm_hooks/powerSGD_hook.py'
    assert file_hash(hook) == file_hash(reference)
    return driver, method


def main():
    recipe = read_json(ROOT / 'recipes/llama60m-adamw-timing-bsz1-local-v1.json')
    full = read_json(ROOT / 'recipes/llama350m-muon-full-offline-v5.json')
    recipe.update(id='powersgd-350m-muon-graph-paired-20261118', compression_start_step=2,
                  required_paths=full['required_paths'], tokenizer_path=full['tokenizer_path'])
    source = full['args']
    keys = ['model_config','dataset_path','max_length','dtype','activation_checkpointing',
            'batch_size','workers','optimizer','lr','weight_decay','grad_clipping']
    recipe['args'] = {k: source[k] for k in keys + [k for k in source if k.startswith('muon_')]}
    recipe['args'].update(warmup_iterations=8, measured_iterations=20,
                          scheduler_steps=60000, lr_warmup_steps=6000,
                          strict_blocking_communication=False, record_bucket_layout=False)
    recipe['completion'].update(measured_steps=20,warmup_iterations=8,
                                first_measured_update=9,last_measured_update=28)
    recipe['notes'] = ('Diagnostic only: same350M Muon BF16 batch128/GA1/seq256/activationcheckpointing, '
                       'schedule60k/warmup6k; original vs graph adapter. Compression2 to cover compressed path. '
                       '8 warmup +20 measured updates, no evaluation/checkpoints/W&B. Not a convergence result.')
    rp=HERE/'paired-timing-recipe.json'; ep=HERE/'paired-timing.json'
    write_json(rp, recipe)
    execution=read_json(ROOT/'campaigns/powersgd-350m-full-20261118/full.json')['execution']
    execution['timeout_seconds']=600
    write_json(ep, {'schema_version':1,'id':'powersgd-350m-muon-graph-paired-20261118',
                   'methods':['powersgd-paper-r32','powersgd-r32-graph-v1'],
                   'reference_methods':['powersgd-paper-r32'], 'recipe':str(rp),
                   'seeds':[1243],'repetitions':1,'execution':execution})
    original=plans.build_runtime
    try:
        plans.build_runtime=build
        run=plans.make_plan(ep)
    finally:
        plans.build_runtime=original
    assert plans.preflight(run)['status']=='static_pass'
    write_json(HERE/'paired-timing-plan.json',{'run':str(run),'plan_hash':read_json(run/'plan.json')['plan_hash']})
    print(run)


if __name__=='__main__':main()
