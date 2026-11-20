"""After all four verified timing cases, smoke rolling saves then start fresh60k."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import argparse
import fcntl
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from glval.common import file_hash, read_json, write_json
from glval.plans import preflight, verify_plan
from glval.runner import gpu_lease, run_plan
from glval.results import inspect_job
from prepare import CHECKPOINT_ROOT, METHOD, validate_contract

TIMING = HERE.parent / 'powersgd-graph-timing-20261118'
SELECTION = '4,5,6,7'


def publish(stage, **details):
    write_json(HERE / 'status.json', {'stage': stage, 'pid': os.getpid(),
                                    'updated_at': time.time(), **details})


def verify_timing(expected_hash):
    """Verify actual outputs and frozen identity, including all four rank records."""
    scope = runpy.run_path(str(TIMING / 'controller.py'))
    manifest = read_json(TIMING / 'manifest.json')
    if manifest['manifest_hash'] != expected_hash:
        raise ValueError('Timing dependency changed')
    if read_json(TIMING / 'status.json')['stage'] != 'completed':
        raise ValueError('Timing has not completed')
    for case in manifest['cases']:
        scope['verify'](manifest, case)
        process = read_json(Path(case['output']) / 'process.json')
        checked, _ = scope['validate_result'](case)
        if not process['valid'] or process['returncode'] or checked['status'] != 'verified':
            raise ValueError('Timing case lacks verified completion: ' + case['id'])
    write_json(HERE / 'timing-dependency.json', {'status': 'verified',
               'manifest_hash': expected_hash, 'cases': [c['id'] for c in manifest['cases']]})


def verify_review(provenance):
    review = read_json(HERE / 'review.json')
    if review['status'] != 'approved' or review['plan_hashes'] != provenance['plan_hashes']:
        raise ValueError('Independent review must match both new frozen plans')
    if review['campaign_files'] != provenance['campaign_files']:
        raise ValueError('Independent review must match campaign code')
    for name, sha in provenance['campaign_files'].items():
        if file_hash(HERE / name) != sha:
            raise ValueError('Campaign code changed: ' + name)


def check_capacity(required_bytes):
    available = shutil.disk_usage('/dev/shm').free
    if available < required_bytes:
        raise ValueError(f'Insufficient checkpoint space: {available} available, {required_bytes} required')
    return available


def verify_phase_checkpoint(phase, run, plan):
    output = run / plan['jobs'][0]['output']
    with (output / 'checkpoint-verification.log').open('w') as log:
        subprocess.run([plan['execution']['python'], str(HERE / 'verify_checkpoint.py'),
                        '--run', str(run)], check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=600)
    if phase == 'smoke':
        # Preserve smoke evidence, retire only our own temporary smoke payloads.
        expected = CHECKPOINT_ROOT / 'smoke' / METHOD / 'seed-1243'
        evidence = read_json(output / 'checkpoint-verification.json')
        if Path(evidence['path']).parent != expected or evidence['status'] != 'verified':
            raise ValueError('Unexpected smoke checkpoint path')
        shutil.rmtree(expected)
        write_json(output / 'smoke-checkpoint-retired.json',
                   {'path': str(expected), 'reason': 'Verified temporary save/prune smoke; full run starts fresh'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--continue-after-verified-smoke', action='store_true')
    options = parser.parse_args()
    if not options.execute:
        parser.error('Explicit --execute required')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_EVALUATE_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    with (HERE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            provenance = read_json(HERE / 'provenance.json')
            verify_review(provenance)
            while True:
                stage = read_json(TIMING / 'status.json')['stage']
                if stage == 'completed':
                    break
                if stage == 'stopped':
                    raise ValueError('Timing stopped; full training blocked')
                publish('waiting_for_timing', dependency=str(TIMING), timing_stage=stage)
                time.sleep(30)
            verify_timing(provenance['timing_manifest_hash'])
            for phase, path in zip(('smoke', 'full'), read_json(HERE / 'queue.json'), strict=True):
                verify_review(provenance)
                run = Path(path)
                plan = verify_plan(run)
                validate_contract(plan['recipe'], plan['execution'], phase)
                if provenance['plan_hashes'].get(path) != plan['plan_hash']:
                    raise ValueError('Frozen plan identity changed')
                if len(plan['jobs']) != 1 or plan['jobs'][0]['method_id'] != METHOD or plan['jobs'][0]['seed'] != 1243:
                    raise ValueError('Only the optimized PowerSGD seed1243 run is authorized')
                state = read_json(run / 'state.json')
                if phase == 'smoke' and options.continue_after_verified_smoke and state['status'] == 'completed':
                    if any(j['status'] != 'completed' for j in state['jobs'].values()):
                        raise ValueError('Smoke has incomplete jobs')
                    inspected = inspect_job(run, plan['jobs'][0], plan['recipe'])
                    if inspected['status'] != 'verified':
                        raise ValueError('Prior smoke outputs failed re-verification')
                    publish('verifying_completed_smoke', run=path, recovery=True)
                    verify_phase_checkpoint(phase, run, plan)
                    print('REUSED_VERIFIED_SMOKE', run.name, flush=True)
                    continue
                if state['status'] != 'planned' or any(j['status'] != 'planned' for j in state['jobs'].values()):
                    raise ValueError('No implicit retry/resume of a started attempt: ' + run.name)
                if preflight(run)['status'] != 'static_pass':
                    raise ValueError('CPU preflight failed')
                required = 32 * 1024 ** 3
                if phase == 'full':
                    smoke_run = Path(read_json(HERE / 'queue.json')[0])
                    smoke_plan = read_json(smoke_run / 'plan.json')
                    evidence = read_json(smoke_run / smoke_plan['jobs'][0]['output'] / 'checkpoint-verification.json')
                    if evidence['status'] != 'verified':
                        raise ValueError('Rolling checkpoint smoke was not verified')
                    required = max(required, int(evidence['checkpoint_bytes'] * 2.4))
                check_capacity(required)
                while True:
                    from contextlib import ExitStack
                    with ExitStack() as resources:
                        try:
                            hardware = resources.enter_context(gpu_lease(SELECTION, 4))
                        except (ValueError, BlockingIOError) as error:
                            publish('waiting_for_idle', phase=phase, run=path, gpus=SELECTION, reason=str(error))
                        else:
                            check_capacity(required)
                            topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
                            write_json(run / 'allocation-topology.json', {'selection': SELECTION, 'topology': topology})
                            publish('training', phase=phase, run=path, gpus=hardware,
                                    checkpoint_policy=plan['recipe']['checkpoint_policy'])
                            print('START', phase, run.name, flush=True)
                            run_plan(run, SELECTION, execute=True, hardware_allocation=hardware)
                            state = read_json(run / 'state.json')
                            if state['status'] != 'completed':
                                raise RuntimeError('Training failed; no next phase: ' + run.name)
                            verify_phase_checkpoint(phase, run, plan)
                            print('COMPLETED', phase, run.name, flush=True)
                            break
                    time.sleep(30)
            publish('completed')
        except BaseException as error:
            publish('stopped', error=str(error))
            raise


if __name__ == '__main__':
    main()
