"""Read timing outputs only; never imports torch or launches jobs."""
from pathlib import Path
import math
from glval.common import read_json,write_json,file_hash
HERE=Path(__file__).resolve().parent

def main():
    manifest=read_json(HERE/'cases.json');rows=[]
    for case in manifest['cases']:
        output=Path(case['output']);row={k:case[k] for k in ['id','setting','variant','method','scope']}
        if not (output/'process.json').exists():
            row['status']='pending';rows.append(row);continue
        process=read_json(output/'process.json');row.update(status='completed' if process['valid'] else 'failed',process=process)
        if process['valid']:
            r=read_json(output/'result.json');ranks=r['ranks']
            refresh=[203,403] if case['method']!='dense' else []
            normal=[i for i in range(case['measured']) if i+case['warmup']+1 not in refresh]
            refreshed=[i for i in range(case['measured']) if i+case['warmup']+1 in refresh]
            row.update(mean_step_ms=r['mean_step_ms'],
                ordinary_step_ms=max(sum(rank['steps'][i] for i in normal)/len(normal)*1000 for rank in ranks),
                refresh_step_ms=max(sum(rank['steps'][i] for i in refreshed)/len(refreshed)*1000 for rank in ranks) if refreshed else None,
                local_collective_input_MiB_per_step=sum(v['local_input_bytes'] for v in ranks[0]['payload'])/2**20,
                collective_calls_per_step=sum(v['count'] for v in ranks[0]['calls']),
                peak_memory_GiB_rank0=r['peak_memory_bytes_rank0']/2**30,
                head_embedding=[v for entries in ranks[0]['compression_classification'].values() for v in entries if 'lm_head' in v['name'] or 'embed_tokens' in v['name']],
                tied_embeddings=ranks[0]['tied_embeddings'],refresh_steps=ranks[0]['refresh_steps'],
                mean_loss_rank0=sum(ranks[0]['losses'])/len(ranks[0]['losses']),
                hook_configuration=ranks[0]['hook_configuration'],parameter_dtypes=ranks[0]['parameter_dtypes'])
        rows.append(row)
    comparisons=[]
    for setting in manifest['settings']:
        group={r['variant']:r for r in rows if r['setting']['id']==setting['id'] and r['status']=='completed'}
        if len(group)!=3:continue
        a,b,d=group['greedy_exclude'],group['greedy_include'],group['dense']
        comparisons.append({'setting':setting['id'],'include_speedup_vs_exclude':a['mean_step_ms']/b['mean_step_ms'],
            'exclude_speedup_vs_dense':d['mean_step_ms']/a['mean_step_ms'],
            'include_speedup_vs_dense':d['mean_step_ms']/b['mean_step_ms'],
            'include_time_change_percent':(b['mean_step_ms']/a['mean_step_ms']-1)*100,
            'collective_input_reduction_percent':(1-b['local_collective_input_MiB_per_step']/a['local_collective_input_MiB_per_step'])*100 if a['local_collective_input_MiB_per_step'] else None})
    status=read_json(HERE/'status.json')
    write_json(HERE/'summary.json',{'status':status,'rows':rows,'comparisons':comparisons})
    provenance={'manifest_hash':file_hash(HERE/'cases.json'),'hardware':read_json(HERE/'hardware.json'),
        'scripts':{p.name:file_hash(p) for p in HERE.glob('*.py')},'cpu_checks_hash':file_hash(HERE/'cpu-checks.log')}
    write_json(HERE/'provenance.json',provenance)
    lines=['# GreedyLoRE：压缩 LM head 和 embedding 的 timing','',
        '四卡 GPU4–7；每卡batch1，seed1243；AdamW lr0.0025/betas0.9,0.999/eps1e-8/weight_decay0；无activation checkpoint；TF32关闭；W&B disabled；DDP bucket2048MiB。三臂均重新测量，不与历史八卡数字直接混比。',
        '', '真实LLaMA模型执行forward/backward/update，输入是固定随机token。warmup21步、测量400步（更新22–421），compression start2，rank32/gap200/EF14，测量覆盖更新203与403两次完整SVD刷新。模型配置名130M/350M沿用原仓库，不等于精确参数数目。',
        '', '| Setting | 方法 | 平均 ms/step | 普通步 ms | 刷新步 ms | 本地collective输入 MiB/step | 状态 |', '|---|---|---:|---:|---:|---:|---|']
    labels={'dense':'Dense','greedy_exclude':'GreedyLoRE：排除head/emb','greedy_include':'GreedyLoRE：包含head/emb'}
    for r in rows:
        if r['status']=='completed':
            refresh=f"{r['refresh_step_ms']:.2f}" if r['refresh_step_ms'] is not None else '—'
            lines.append(f"| {r['setting']['id']} | {labels[r['variant']]} | {r['mean_step_ms']:.2f} | {r['ordinary_step_ms']:.2f} | {refresh} | {r['local_collective_input_MiB_per_step']:.2f} | completed |")
        else:lines.append(f"| {r['setting']['id']} | {labels[r['variant']]} | — | — | — | — | {r['status']} |")
    lines+=['','## 配对对照','', '| Setting | 包含/排除耗时变化 | 包含相对排除加速 | 包含相对Dense加速 |','|---|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {c['setting']} | {c['include_time_change_percent']:+.1f}% | {c['include_speedup_vs_exclude']:.3f}x | {c['include_speedup_vs_dense']:.3f}x |")
    lines+=['','原始GreedyLoRE源码和rank/EF/刷新规则不变。独立变体只在注册时传空的名字排除列表；一维张量仍由原生shape判定不压缩。运行时审计原生分组返回值，确认lm_head、embed_tokens两个独立矩阵分别实际纳入/排除压缩，并验证每rank执行421次更新。',
        '', '普通步与刷新步均按最慢rank取各类平均。总体平均使用最慢rank连续窗口耗时，包含刷新及少量仪表开销。本地collective输入为Python层all_reduce/all_gather提交字节数，包含原生投影/统计通信，非网络实际流量；不应解读为精确网络带宽。',
        '', '仅单seed性能筛查：不代表真实C4数据吞吐、压缩后收敛质量或完整预训练收益。未进行其他候选的比较。新脚本/快照/输出均留在独立scaffold，未修改原项目或停止其他GPU任务。',
        '', '远程目录：`/home/wyr/greedylore-validation/diagnostics/head-emb-timing-20261026`。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')
    print({'status':status['status'],'completed':sum(r['status']=='completed' for r in rows),'comparisons':comparisons,'rows':[{k:r[k] for k in ['id','status','mean_step_ms','ordinary_step_ms','refresh_step_ms','local_collective_input_MiB_per_step'] if k in r} for r in rows]})
if __name__=='__main__':main()
