"""Export small, derived records while preserving original execution evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

from .common import ROOT, file_hash, read_json, write_json
from .artifacts import run_directory

STEP_FIELDS = {'rank_iteration_seconds', 'rank_losses', 'rank_bucket_counts',
               'rank_bucket_bytes', 'rank_hook_step_seconds'}


def sequence_summary(values):
    encoded = json.dumps(values, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    result = {'count': len(values), 'sha256': hashlib.sha256(encoded).hexdigest()}
    if not values:
        return result
    for key, item in (('first', values[0]), ('last', values[-1])):
        if not isinstance(item, (dict, list)) or (isinstance(item, list) and len(item) <= 8
                                                and all(not isinstance(x, (dict, list)) for x in item)):
            result[key] = item
    if all(type(x) in (int, float) and math.isfinite(x) for x in values):
        ordered = sorted(values)
        result.update(min=ordered[0], max=ordered[-1], mean=statistics.fmean(values),
                      median=statistics.median(values), p95=ordered[math.ceil(.95 * len(values))-1],
                      p99=ordered[math.ceil(.99 * len(values))-1])
        if len(values) >= 2 and all(type(x) is int for x in values):
            step = values[1] - values[0]
            if all(x == values[0] + i * step for i, x in enumerate(values)):
                result['step'] = step
    return result


def has_long_sequence(value):
    if isinstance(value, list):
        return len(value) > 32 or any(has_long_sequence(x) for x in value)
    if isinstance(value, dict):
        return any(has_long_sequence(x) for x in value.values())
    return False


def collect_sequences(value, path, output):
    if isinstance(value, list):
        if len(value) > 32 or not any(isinstance(x, (dict, list)) for x in value):
            output[path] = sequence_summary(value)
        else:
            for i, item in enumerate(value):
                collect_sequences(item, f'{path}[{i}]', output)
    elif isinstance(value, dict):
        for key, item in value.items():
            if has_long_sequence(item):
                collect_sequences(item, f'{path}.{key}', output)


def export_file(source):
    source = Path(source)
    data = read_json(source)
    if re.fullmatch(r'runtime-rank-\d+\.json', source.name):
        kind = 'runtime'
        target = source.with_name(source.name.replace('runtime-rank-', 'runtime-summary-rank-'))
    elif source.name == 'all_results.json' and 'mean_iteration_seconds' in data:
        kind, target = 'timing', source.with_name('timing-summary.json')
    else:
        return None
    measurements, omitted = {}, {}
    for key, value in data.items():
        if key == 'timing_refresh_steps' and isinstance(value, list):
            omitted[key] = sequence_summary(value)
        elif kind == 'timing' and key.startswith('rank_') and (
                (key in STEP_FIELDS and isinstance(value, list)) or has_long_sequence(value)):
            collect_sequences(value, key, omitted)
        else:
            measurements[key] = value
    write_json(target, {'schema_version': 1, 'record_type': f'compact_saved_{kind}',
                        'status': data.get('status'), 'source': source.name,
                        'source_sha256': file_hash(source), 'measurements': measurements,
                        'omitted_sequences': omitted,
                        'statistics_convention': 'fmean, median, nearest-rank p95/p99 for finite numeric sequences',
                        'limitations': ['Derived saved evidence; not a new verification or execution.',
                                        'Raw per-step evidence remains local and is excluded from publication.',
                                        'Official scalar measurements are copied without recomputation.']})
    return target


def export_tracked(root=ROOT, runs=None):
    root = Path(root)
    names = [] if runs else subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', 'runs']).decode().split('\0')
    if runs:
        names = set()
        for selected in runs:
            run = run_directory(root / selected)
            if not run.is_dir():
                raise ValueError(f'Run does not exist: {run}')
            for pattern in ('jobs/*/output/runtime-rank-*.json', 'cases/*/output/runtime-rank-*.json',
                            'jobs/*/output/all_results.json', 'cases/*/output/all_results.json'):
                names.update(path.relative_to(root).as_posix() for path in run.glob(pattern))
    records, ignores = [], {}
    for name in sorted(n for n in names if n):
        source = root / name
        if source.name != 'all_results.json' and not re.fullmatch(r'runtime-rank-\d+\.json', source.name):
            continue
        if not source.resolve().is_relative_to(root.resolve()) or source.is_symlink():
            raise ValueError(f'Raw evidence must be a regular file: {name}')
        target = export_file(source)
        if target is None:
            continue
        records.append({'source': name, 'source_sha256': file_hash(source), 'source_bytes': source.stat().st_size,
                        'summary': target.relative_to(root).as_posix(), 'summary_sha256': file_hash(target),
                        'summary_bytes': target.stat().st_size})
        if source.name == 'all_results.json':
            run = root / 'runs' / Path(name).parts[1]
            ignores.setdefault(run / '.gitignore', set()).add('/' + source.relative_to(run).as_posix())
    for path, additions in ignores.items():
        existing = path.read_text() if path.exists() else '# Raw timing observations remain local; publish timing-summary.json.\n'
        lines = set(existing.splitlines())
        new = sorted(additions - lines)
        if new:
            path.write_text(existing.rstrip() + '\n' + '\n'.join(new) + '\n')
    return {'records': records, 'raw_bytes': sum(r['source_bytes'] for r in records),
            'summary_bytes': sum(r['summary_bytes'] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Write derived summaries; original files are unchanged')
    parser.add_argument('--run', action='append', help='Select a run including its new/ignored evidence; repeatable')
    options = parser.parse_args()
    if not options.execute:
        parser.error('Use --execute to write derived records')
    print(json.dumps(export_tracked(runs=options.run), indent=2))


if __name__ == '__main__':
    main()
