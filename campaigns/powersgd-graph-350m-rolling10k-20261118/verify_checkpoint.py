"""Explicitly executed after the GPU smoke; deserialize actual checkpoint payloads."""

# Historical campaign CLI archived; definitions remain importable for CPU tests.
if __name__ == "__main__":
    import sys as _legacy_sys
    print("Historical campaign CLI archived. Use python3 -m glval plan EXPERIMENT.json; read saved results with glval artifacts RUN.", file=_legacy_sys.stderr)
    raise SystemExit(2)

import argparse
import json
from pathlib import Path


def validate_hook_memory(hook, is_tensor):
    """Check the frozen350M layout, including its two uncompressed end buckets."""
    names = ('error_dict', 'p_memory_dict', 'q_memory_dict')
    expected_keys = set(range(23))
    # Verified350M reducer layout: embedding, and head plus final RMSNorm.
    dense_sizes = {0: 32000 * 1024, 22: 32000 * 1024 + 1024}
    for name in names:
        assert set(hook[name]) == expected_keys, f'Incomplete {name} bucket set'
        assert all(is_tensor(v) and v.device.type == 'cpu' for v in hook[name].values()), name
    for bucket in sorted(expected_keys):
        error, p, q = (hook[name][bucket].numel() for name in names)
        assert error > 0, f'Missing EF buffer for bucket{bucket}'
        if bucket in dense_sizes:
            assert error == dense_sizes[bucket] and p == q == 0, f'Dense-only bucket{bucket} changed'
        else:
            assert p > 0 and q > 0, f'Missing compressed P/Q for bucket{bucket}'
    return {name: sum(v.numel() for v in hook[name].values()) for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    options = parser.parse_args()
    run = options.run
    plan = json.loads((run / 'plan.json').read_text())
    job = plan['jobs'][0]
    out = run / job['output']
    recipe = plan['recipe']
    expected = recipe['args']['num_training_steps']
    root = Path(recipe['checkpoint_policy']['root']) / job['method_id'] / f"seed-{job['seed']}"
    latest = root / 'latest'
    assert latest.is_symlink() and latest.resolve() == root / f'step-{expected:08d}'
    assert sorted(p.name for p in root.iterdir()) == ['latest', f'step-{expected:08d}']
    manifest = json.loads((latest / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['update_step'] == expected
    assert manifest['world_size'] == 4
    for item in manifest['files']:
        assert (latest / item['name']).stat().st_size == item['bytes'] > 0
    events = [json.loads(line)['update_step'] for line in (out / 'checkpoint-events.jsonl').read_text().splitlines()]
    assert events == list(range(recipe['checkpoint_policy']['every_updates'], expected + 1,
                                recipe['checkpoint_policy']['every_updates']))
    import torch
    ranks = []
    for rank in range(4):
        data = torch.load(latest / f'rank-{rank}.pt', map_location='cpu', weights_only=False)
        assert data['rank'] == rank and data['world_size'] == 4 and data['update_step'] == expected
        assert data['optimizer']['state'] and data['optimizer']['param_groups']
        assert data['scheduler']['last_epoch'] == expected
        hook = data['hook_state']
        assert hook['iter'] == expected
        assert hook['matrix_approximation_rank'] == 32 and hook['use_error_feedback'] == 'ef14'
        assert hook['warm_start'] is True
        assert {'torch', 'cuda', 'numpy', 'python'} == set(data['rng'])
        assert hasattr(hook['rng'], 'get_state')
        counts = validate_hook_memory(hook, torch.is_tensor)
        assert counts == {'error_dict': 367969280, 'p_memory_dict': 8134656,
                          'q_memory_dict': 6819840}, 'Saved350M EF/P/Q size mismatch'
        ranks.append({'rank': rank, 'hook_tensor_elements': counts, 'hook_iter': hook['iter']})
        del data, hook
    model = torch.load(latest / 'model.pt', map_location='cpu', weights_only=False)
    config = json.loads((latest / 'model_config.json').read_text())
    shape = (config['vocab_size'], config['hidden_size'])
    assert shape == (32000, 1024)
    assert model['model.embed_tokens.weight'].shape == shape
    assert model['lm_head.weight'].shape == shape
    assert len(model) > 100
    result = {'status': 'verified', 'update_step': expected, 'ranks': ranks,
              'checkpoint_bytes': sum(p.stat().st_size for p in latest.iterdir()),
              'path': str(latest), 'retired_steps': events[:-1], 'model_tensor_count': len(model),
              'limits': 'No exact data iterator/DDP layout resume implementation.'}
    (out / 'checkpoint-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
