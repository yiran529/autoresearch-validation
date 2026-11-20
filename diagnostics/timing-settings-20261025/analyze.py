"""Validate completed short timing cases and report matched-setting comparisons."""
import json
import math
from pathlib import Path
from glval.common import read_json, write_json, inventory, file_hash

HERE=Path(__file__).resolve().parent
METHODS=['dense','greedylore-engineer-streamfix','n023-engineer-maxu','n031-engineer-maxu','n020-engineer-maxu']
LABELS=['Dense','GreedyLoRE','N23','N31','N20']

def main():
    manifest=read_json(HERE/'cases.json');rows=[];all_records=[]
    for setting in manifest['settings']:
        entry={'setting':setting,'methods':{}}
        for case in [c for c in manifest['cases'] if c['setting']['id']==setting['id']]:
            output=Path(case['output']);process=output/'process.json'
            if not process.exists():continue
            record={'id':case['id'],'method':case['method'],'process':read_json(process),'valid':False}
            try:
                assert record['process']['returncode']==0,'Nonzero process exit'
                assert inventory(Path(case['runtime']))==case['runtime_files'],'Runtime modified'
                result=read_json(output/'result.json')
                assert len(result['ranks'])==case['world_size'],'Rank count mismatch'
                assert sorted(r['rank'] for r in result['ranks'])==list(range(case['world_size']))
                assert result['global_batch']==case['world_size']*case['batch_size']
                telemetry=[]
                for r in result['ranks']:
                    assert len(r['steps'])==len(r['losses'])==case['measured']
                    assert all(math.isfinite(v) and v>0 for v in r['steps'])
                    assert all(math.isfinite(v) for v in r['losses'])
                    state=read_json(output/('runtime-rank-'+str(r['rank'])+'.json'))
                    assert state['last_bucket_calls']==case['warmup']+case['measured']
                    assert state['hook_registrations']==1
                    if case['method']=='greedylore-engineer-streamfix':
                        if state.get('timing_refresh_steps') is not None:
                            measured_refresh=[i for i in state['timing_refresh_steps'] if case['warmup']<i<=case['warmup']+case['measured']]
                            assert len(measured_refresh)==1,('Expected one measured SVD refresh',measured_refresh)
                            record['refresh_evidence']='Direct method telemetry'
                        else:
                            # Streamfix source has the optional telemetry append but does not initialize its list.
                            # Do not fabricate that telemetry: validate source schedule + final hook iteration
                            # and show the actual synchronized timing spike at the scheduled update.
                            source=(Path(case['runtime'])/state['hook_source']).read_text()
                            assert '((state.iter - state.start_compress_iter) % state.update_proj_gap == 0)' in source
                            assert state['iter']==case['warmup']+case['measured']
                            refresh=[i for i in range(case['warmup']+1,case['warmup']+case['measured']+1) if (i-1-2)%200==0]
                            assert refresh==[203]
                            spike=max(range(case['measured']),key=lambda i:r['steps'][i])+case['warmup']+1
                            assert spike==203,('Expected refresh cost spike at203',spike)
                            record['refresh_evidence']='Source schedule + final iteration221 + every rank largest measured step at203; optional direct list absent'
                    telemetry.append({k:state.get(k) for k in ['rank','iter','timing_refresh_steps','timing_period_steps','effective_state_configuration','peak_memory_bytes']})
                # All ranks have a per-step device synchronization; use maximum rank wall window as primary metric.
                record.update(valid=True,mean_step_ms=result['mean_step_ms'],
                              tokens_per_second=case['world_size']*case['batch_size']*case['sequence_length']*1000/result['mean_step_ms'],
                              telemetry=telemetry,rank_elapsed=[r['elapsed'] for r in result['ranks']],
                              bucket_counts=[len(r['buckets']) for r in result['ranks']],
                              slowest_step_ms=max(max(r['steps']) for r in result['ranks'])*1000,
                              peak_memory_mib=max(t['peak_memory_bytes'] for t in telemetry)/2**20)
                rank0=result['ranks'][0]
                tensors=[t for bucket in rank0['buckets'].values() for t in bucket['tensors']]
                record.update(parameter_count=sum(t['numel'] for t in tensors),
                              embedding_head_norm_fraction=sum(t['numel'] for t in tensors if len(t['shape'])<2 or 'embed_tokens' in t['name'] or 'lm_head' in t['name'])/sum(t['numel'] for t in tensors),
                              logical_calls=rank0['calls'],local_input_payload=rank0['payload'],
                              syncs_per_step=rank0['syncs_per_step'],phase_cpu_wall_ms=rank0['phase_cpu_wall_ms'])
            except Exception as error:record['error']=str(error)
            entry['methods'][case['method']]=record;all_records.append(record)
        dense=entry['methods'].get('dense',{})
        if dense.get('valid'):
            for record in entry['methods'].values():
                if record['valid']:record['speedup_vs_same_setting_dense']=dense['mean_step_ms']/record['mean_step_ms']
        rows.append(entry)
    summary={'scope':'Single seed1243 per setting/method; synthetic cached tokens, BF16 seq256; warm21 measure200; compression starts2; W&B disabled; no checkpoint files; no quality/convergence claim.',
             'rows':rows,'finished_count':len(all_records),'valid_count':sum(r['valid'] for r in all_records),
             'total':len(manifest['cases'])}
    write_json(HERE/'analysis.json',summary)
    write_json(HERE/'provenance.json',{'source_diagnostic':'compression-cost-20261025','case_manifest_hash':file_hash(HERE/'cases.json'),
               'scripts':{n:file_hash(HERE/n) for n in ['prepare.py','prepare_extra.py','worker.py','run.py','analyze.py','write_zh.py']},
               'methods':{m:read_json(Path(next(c['runtime'] for c in manifest['cases'] if c['method']==m))/'method.json') for m in METHODS},
               'seed':1243,'repeats':1,'runtime_hashes':'cases.json case.runtime_files'})
    lines=['# Timing setting sweep','',summary['scope'],'',
           f"Completed {summary['finished_count']}/{summary['total']}; validated {summary['valid_count']}.",'',
           '| Setting | Dense ms | GreedyLoRE ms | N23 ms | N31 ms | N20 ms |',
           '|---|---:|---:|---:|---:|---:|']
    for entry in rows:
        values=[]
        for method in METHODS:
            record=entry['methods'].get(method,{})
            if record.get('valid'):
                values.append(f"{record['mean_step_ms']:.2f}"+(f" ({record['speedup_vs_same_setting_dense']:.3f}x)" if method!='dense' and 'speedup_vs_same_setting_dense' in record else ''))
            elif record:values.append('FAILED: '+record.get('error',''))
            else:
                planned=any(c['setting']['id']==entry['setting']['id'] and c['method']==method for c in manifest['cases'])
                values.append('pending' if planned else '—')
        lines.append('| '+entry['setting']['id']+' | '+' | '.join(values)+' |')
    lines+=['','Default bucket means PyTorch bucket_cap_mb=None; big means1024MiB. Muon rows have activation checkpointing enabled; AdamW rows disabled. Batch1/device throughout. Four-card runs use GPUs4-7, eight-card runs use0-7; same local batch is weak scaling (global batch4 versus8).',
            '','Every measured200-step window covers one full GreedyLoRE SVD cycle and20 N20 rejuvenation intervals. Greedy streamfix optional refresh list is absent: coverage is checked from frozen source schedule, final iteration221, and every rank timing spike at update203. Compression starts early for cost diagnostics; optimizer LR remains on the original warmup schedule.',
            '','Compare methods within each row. Across rows, distinguish model size, optimizer, activation checkpointing and global batch. No physical network bandwidth throttling is applied. Single-run differences near a few percent require confirmation.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'finished':summary['finished_count'],'valid':summary['valid_count'],'total':summary['total']}))

if __name__=='__main__':main()
