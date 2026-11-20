"""Freeze four graph-only timing cases with the native passive bucket observer."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import ast
from pathlib import Path

from glval.build import build_runtime
from glval.common import ROOT, VENDOR, digest, file_hash, inventory, read_json, write_json
from glval.plans import driver_args

HERE = Path(__file__).resolve().parent
REFERENCE = Path('/home/wyr/coral-experiments/greedylore-v5-dense-reuse-20261027/run/.coral/public/eval_logs/2410afc4f56b989b4040e0b9930011dfbf428102/final-9f954236bdc51647c23d0a0905102c95/timing/efficiency')
METHOD = 'powersgd-r32-graph-v1'
CODE_FILES = ('prepare.py', 'controller.py')
KEYS = ('model_config','dataset_path','batch_size','max_length','dtype','warmup_iterations',
        'measured_iterations','scheduler_steps','lr_warmup_steps','activation_checkpointing',
        'ddp_bucket_cap_mb','workers','lr','beta1','beta2','eps','weight_decay','grad_clipping','optimizer')


def definitions():
    cases = []
    for precision, repetition in [('bf16',1),('fp32',1),('fp32',2),('bf16',2)]:
        source = REFERENCE / f'r{repetition}-350m_adamw_{precision}'
        config = read_json(source / 'config.json')
        historical = read_json(source / 'all_results.json')
        args = {key: config[key] for key in KEYS}
        args.update(record_bucket_layout=True, strict_blocking_communication=False)
        case_id = f'powersgd-graph-350m-{precision}-r{repetition}-20261118'
        recipe = {
            'schema_version':1, 'id':case_id, 'kind':'timing', 'workload':'llama',
            'world_size':4, 'compression_start_step':100, 'evaluation_split':'none',
            'tokenizer_path':config['tokenizer_path'],
            'required_paths':[config['model_config'], config['dataset_path'], config['tokenizer_path']],
            'args':args, 'tracking':{'mode':'disabled'}, 'checkpoint_policy':'none',
            'completion':{'status':'completed','measured_steps':400,'warmup_iterations':101,
                          'first_measured_update':102,'last_measured_update':501,
                          'strict_blocking_communication':False,'bucket_layout_recorded':True},
            'metrics':['mean_iteration_seconds'],
            'notes':'Graph-only PowerSGD, native AutoResearch passive bucket recording; no evaluation/checkpoints/W&B.',
        }
        cases.append({'id':case_id,'method_id':METHOD,'seed':1243,'precision':precision,
                      'repetition':repetition,'recipe':recipe,'reference_config':str(source/'config.json'),
                      'expected_bucket_counts':historical['observed_bucket_counts'],
                      'expected_bucket_bytes':historical['observed_bucket_bytes']})
    return cases


def patch_recording(path):
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='register_timing_comm_hook']
    if len(functions) != 1: raise ValueError('Timing registration anchor changed')
    fn = functions[0]
    if not any(a.arg=='optimizer' for a in fn.args.args): fn.args.args.append(ast.arg(arg='optimizer'))
    fn.body = ast.parse('''
if args.strict_blocking_communication or not args.record_bucket_layout:
    raise ValueError('This campaign requires passive bucket recording and asynchronous communication')
from glval.bridge import register_comm_hook_for_ddp_model
observer = BucketRecordingHookState(None, None)
native_register = model.register_comm_hook
def register_observed(inner_state, inner_hook):
    observer.inner_state = inner_state
    observer.inner_hook = inner_hook
    native_register(observer, bucket_recording_hook)
model.register_comm_hook = register_observed
try:
    register_comm_hook_for_ddp_model(model, process_group, args, optimizer=optimizer)
finally:
    model.register_comm_hook = native_register
return observer
''').body
    path.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
    # The observer class, callback and measurement loop are the original vendor code.
    original = ast.parse((VENDOR/'c4/table_v_timing.py').read_text())
    for name in ('BucketRecordingHookState','bucket_recording_hook','summarize_bucket_layout'):
        a = next(n for n in original.body if getattr(n,'name',None)==name)
        b = next(n for n in tree.body if getattr(n,'name',None)==name)
        assert ast.dump(a)==ast.dump(b), name


def main():
    if (HERE/'manifest.json').exists(): raise ValueError('Campaign already frozen; no overwrite')
    cases = definitions()
    for case in cases:
        directory = HERE/'cases'/case['id']
        runtime, output = directory/'runtime', directory/'output'
        if directory.exists(): raise ValueError('Existing attempt: '+case['id'])
        driver, method = build_runtime(runtime, ROOT/'methods'/METHOD/'method.json', case['recipe'])
        hook = runtime/'comm_hooks/powerSGD_hook.py'
        text = hook.read_text()
        old = 'for tensor in tensors:\n        matrix = tensor.view(tensor.shape[0], -1)'
        assert text.count(old)==1
        text = text.replace(old,'for tensor, parameter in zip(tensors, bucket.parameters()):\n        matrix = tensor.view(tensor.shape[0], -1)')
        assert text.count('        if compress_test[0]:')==1
        text = text.replace('        if compress_test[0]:','        if compress_test[0] and id(parameter) not in state.excluded_parameter_ids:')
        compile(text,str(hook),'exec'); hook.write_text(text)
        assert file_hash(hook)=='ba49b4f33dcc921d634302f8817d570c8605dd44c8bd4b856a6a33536f83f857'
        assert file_hash(runtime/'candidate_adapter.py')=='387e057f3ffa41717a9ad54a263848024eb372f65524bcb9cf483376dc107b62'
        patch_recording(runtime/driver)
        for name in case['recipe']['required_paths']:
            if not (Path(name) if Path(name).is_absolute() else runtime/name).exists():
                raise ValueError('Missing input: '+name)
        case.update(runtime=str(runtime), output=str(output), driver=driver,
                    args=driver_args(case['recipe'],output,case['seed'],method),
                    source_commit=method['source_commit'], runtime_files=inventory(runtime))
        write_json(directory/'case.json',case)
    manifest = {'version':1,'method_id':METHOD,'gpus':'4,5,6,7','cases':cases,
                'python':'/home/wyr/greedy_lore/.venv/bin/python',
                'campaign_files':{name:file_hash(HERE/name) for name in CODE_FILES},
                'scope':'Only optimized PowerSGD. Native passive bucket recording; original math/window. No Dense or original PowerSGD runs.'}
    manifest['manifest_hash']=digest(manifest)
    write_json(HERE/'manifest.json',manifest)
    write_json(HERE/'status.json',{'stage':'prepared','gpu_execution':False,'case_count':4})
    print([(c['precision'],c['repetition'],c['expected_bucket_counts']) for c in cases])


if __name__=='__main__':main()
