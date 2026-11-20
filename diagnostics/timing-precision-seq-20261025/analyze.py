"""Check precision and shape as well as frozen methods and completed timing windows."""
import math
from pathlib import Path
from glval.common import read_json,write_json,inventory,file_hash

HERE=Path(__file__).resolve().parent
VARIANTS=['dense_default','dense_big','n23_big','n20_big']

def main():
    manifest=read_json(HERE/'cases.json');rows=[];records=[]
    retries=read_json(HERE/'retries.json')['attempts'] if (HERE/'retries.json').exists() else []
    for setting in manifest['settings']:
        row={'setting':setting,'variants':{}}
        for case in [c for c in manifest['cases'] if c['setting']['id']==setting['id']]:
            output=Path(case['output'])
            if not (output/'process.json').exists():continue
            process=read_json(output/'process.json');record={'id':case['id'],'valid':False,'returncode':process['returncode'],
                'prior_attempts':[r for r in retries if r['case']==case['id']]}
            try:
                assert process['returncode']==0,('Process failed',process['returncode'])
                assert inventory(Path(case['runtime']))==case['runtime_files'],'Frozen runtime changed'
                result=read_json(output/'result.json');ranks=result['ranks']
                assert len(ranks)==case['world_size']==8
                assert sorted(r['rank'] for r in ranks)==list(range(8))
                assert result['global_batch']==8
                for r in ranks:
                    assert len(r['steps'])==len(r['losses'])==case['measured']
                    assert all(math.isfinite(v) and v>0 for v in r['steps'])
                    assert all(math.isfinite(v) for v in r['losses'])
                    assert r['parameter_dtypes']==r['gradient_dtypes']==['torch.'+case['dtype']]
                    assert r['input_shape']==[1,case['sequence_length']]
                    assert r['allow_tf32']==case['allow_tf32']==False
                    assert r['float32_matmul_precision']=='highest'
                    assert all('None' not in t for t in r['gradient_dtypes'])
                    state=read_json(output/f"runtime-rank-{r['rank']}.json")
                    assert state['last_bucket_calls']==case['warmup']+case['measured']
                    assert state['hook_registrations']==1
                bucket_counts=[len(r['buckets']) for r in ranks]
                if case['bucket_mb'] is not None:assert bucket_counts==[1]*8,'Expected one big bucket'
                rank0=ranks[0]
                record.update(valid=True,mean_step_ms=result['mean_step_ms'],
                    tokens_per_second=8*case['sequence_length']*1000/result['mean_step_ms'],
                    bucket_counts=bucket_counts,phase_cpu_wall_ms=rank0['phase_cpu_wall_ms'],
                    local_input_payload=rank0['payload'],logical_calls=rank0['calls'],
                    syncs_per_step=rank0['syncs_per_step'],precision={k:rank0[k] for k in ['parameter_dtypes','gradient_dtypes','optimizer_state_dtypes','input_shape','allow_tf32','float32_matmul_precision']},
                    peak_memory_mib=max(read_json(output/f'runtime-rank-{i}.json')['peak_memory_bytes'] for i in range(8))/2**20)
                if case['method']=='greedylore-engineer-streamfix':
                    slowest=max(ranks,key=lambda r:r['elapsed'])
                    steps_ms=[v*1000 for v in slowest['steps']]
                    refresh_index=203-case['warmup']-1
                    assert len(steps_ms)==200 and refresh_index==181
                    regular=(sum(steps_ms)-steps_ms[refresh_index])/199
                    record.update(regular_mean_ms=regular,refresh_update=203,
                                  refresh_step_ms=steps_ms[refresh_index],
                                  amortized_refresh_excess_ms=(steps_ms[refresh_index]-regular)/200)
            except Exception as error:record['error']=str(error)
            row['variants'][case['variant']]=record;records.append(record)
        dense=[r for v,r in row['variants'].items() if v.startswith('dense') and r['valid']]
        if dense:
            best=min(dense,key=lambda r:r['mean_step_ms']);row['best_dense']=best['id']
            for record in row['variants'].values():
                if record['valid']:record['speedup_vs_best_dense']=best['mean_step_ms']/record['mean_step_ms']
        rows.append(row)
    result={'total':len(manifest['cases']),'finished':len(records),'valid':sum(r['valid'] for r in records),
            'scope':'Single seed1243;8GPUs;AdamW;activation checkpointing OFF;batch1/device;warm21 measure100;synthetic cached tokens;fullFP32 TF32 OFF;not quality validation',
            'rows':rows,'archived_startup_failures':len(retries)}
    write_json(HERE/'analysis.json',result)
    write_json(HERE/'provenance.json',{'case_manifest_hash':file_hash(HERE/'cases.json'),
        'scripts':{p.name:file_hash(p) for p in HERE.glob('*.py')},
        'methods':{m:read_json(Path(next(c['runtime'] for c in manifest['cases'] if c['method']==m))/'method.json') for m in sorted({c['method'] for c in manifest['cases']})},
        'seed':1243,'repeats':1,'runtime_file_hashes':'cases.json case.runtime_files'})
    lines=['# FP32 / seq64 timing 扫描','',f"已完成{result['finished']}/{result['total']}项，通过检查{result['valid']}项。",'',
        '八卡RTX4090，AdamW，每卡batch1，关闭激活重计算、W&B和checkpoint保存。每项seed1243一次。预热21步、测量100步；从iter2压缩，N20包含10个rejuvenation周期。固定合成token，无收敛验证。',
        '', 'FP32参数和梯度均为FP32，无autocast，TF32关闭；各rank的实际精度与形状均检查。BF16条件也关闭TF32，作为同轮对照。',
        '', '大bucket为2048MiB，保证350M FP32也只有一个bucket；默认为None。单位ms/步，越小越快。', '',
        '| 模型 / 精度 / seq | Dense默认 | Dense大bucket | N23大bucket | N20大bucket |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        vals=[]
        for variant in VARIANTS:
            r=row['variants'].get(variant,{})
            vals.append(f"{r['mean_step_ms']:.2f}" if r.get('valid') else ('失败:'+r.get('error','') if r else '待完成'))
        lines.append('| '+row['setting']['id']+' | '+' | '.join(vals)+' |')
    lines+=['','相对速度应与同一模型、精度、seq下最快Dense比较。不同seq每步token数不同，ms/步不能直接解释为训练效率；跨精度、seq的总训练收益还需验证质量。',
            '', '每项保留所有rank计时、有限loss、参数/梯度/优化器state dtype、collective payload类型、bucket数、绑定与文件哈希。原始项目、候选实现和依赖未修改。']
    if retries:lines+=['',f'另有{len(retries)}项初次尝试在TCPStore启动阶段遇到EADDRINUSE，未启动模型；原输出完整保留在output-startup-eaddrinuse-attempt1，显式重试记录在retries.json。表中使用重试后的结果。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
