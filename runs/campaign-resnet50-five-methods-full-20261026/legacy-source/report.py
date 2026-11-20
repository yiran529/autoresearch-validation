"""Read-only campaign progress; no torch imports or GPU execution."""
import re
from pathlib import Path
from glval.common import read_json,write_json,file_hash

HERE=Path(__file__).resolve().parent
LABELS={'greedylore-engineer-streamfix-resnet-v1':'GreedyLoRE','n023-engineer-maxu-resnet-v1':'N23',
        'n031-engineer-maxu-resnet-v1':'N31','n020-engineer-maxu-resnet-v1':'N20','dense':'Dense'}

def main():
    manifest=read_json(HERE/'cases.json');status=read_json(HERE/'status.json')
    rows=[];checks=[]
    for case in manifest['cases']:
        output=Path(case['output']);process=output/'process.json'
        if case['phase']=='synthetic_check':
            checks.append({'method':LABELS[case['method_id']],**(read_json(process) if process.exists() else {'valid':False,'status':'pending'})})
            continue
        row={'method':LABELS[case['method_id']],'id':case['id'],'status':'planned','output':str(output)}
        if process.exists():
            record=read_json(process)
            row.update(status='completed' if record['valid'] else 'failed',process=record)
        elif status.get('case')==case['id']:row['status']=status['status']
        metrics=output/'metrics.jsonl'
        if metrics.exists():
            lines=metrics.read_text().splitlines()
            if lines:
                import json
                row['last_epoch']=json.loads(lines[-1])
        stdout=output/'stdout.log'
        if stdout.exists():
            updates=re.findall(r'TRAIN epoch=(\d+) step=(\d+) loss=([\d.eE+-]+)',stdout.read_text())
            if updates:row['last_logged_update']={'epoch':int(updates[-1][0]),'step':int(updates[-1][1]),'loss':float(updates[-1][2])}
        if (output/'tracker.json').exists():row['tracking']=read_json(output/'tracker.json')
        rows.append(row)
    summary={'status':status['status'],'gpu_selection':status.get('selection'),'recipe':manifest['full_recipe'],
             'synthetic_checks_passed':sum(bool(c.get('valid')) for c in checks),'methods':rows,'checks':checks,
             'boundary':'Synthetic adapter checks do not establish full-data convergence. Only40epoch/15640step outputs count as full completion.'}
    write_json(HERE/'summary.json',summary)
    write_json(HERE/'provenance.json',{'manifest_hash':file_hash(HERE/'cases.json'),
        'scripts':{p.name:file_hash(p) for p in HERE.glob('*.py')},'cpu_checks_hash':file_hash(HERE/'cpu-checks.log'),
        'source_identity':'Each case.runtime_files; new ResNet method variants preserve parent source and parameter hashes; only classifier-name registration bridge added.'})
    lines=['# ResNet50五方法完整训练','',
        f"状态：{status['status']}；GPU：{status.get('selection')}；已通过合成检查：{summary['synthetic_checks_passed']}/5。",'',
        'CIFAR-10，ResNet50，40epoch，15640updates，每卡batch32、四卡全局128，seed1243。FP32模型，Muon矩阵LR0.02、标量LR0.005，weight decay0.0005，cosine到0，按epoch调度，无LR warmup。压缩从更新501开始，rank32/gap200；N20内部rank8/rejuvenation10。linear分类层不压缩。',
        '', '五方法串行使用四张GPU。W&B online，project GreedyLore-CIFAR；不保存模型/优化器checkpoint。CIFAR10每epoch评估全部10000个测试样本。',
        '', '| 顺序 / 方法 | 状态 | 最近训练日志 | 最近完整epoch | W&B |','|---|---|---|---|---|']
    for index,row in enumerate(rows,1):
        update=row.get('last_logged_update',{});epoch=row.get('last_epoch',{})
        train=f"step{update['step']}, loss{update['loss']:.4f}" if update else '—'
        metric=f"epoch{epoch['epoch']}, test_acc{epoch['test_accuracy']:.2f}%" if epoch else '—'
        link=row.get('tracking',{}).get('url');tracker=f'[查看]({link})' if link else '—'
        lines.append(f"| {index}. {row['method']} | {row['status']} | {train} | {metric} | {tracker} |")
    lines+=['','四个候选保留当前Engineer源码与关键超参数。新adapter只补充ResNet线性分类层名字映射；卷积展开使用候选原生实现。未修改原始项目或依赖。',
            '', 'GPU接入检查是205步合成数据、compression start2，包含更新203的刷新与逐步参数一致性检查。它与真实CIFAR10完整训练分开保存，不作为完整收敛结论。',
            '', '远程目录：`/home/wyr/greedylore-validation/campaigns/resnet50-five-methods-full-20261026`；tmux：`glval-resnet50-full-20261026`。',
            '', '查看进度：在scaffold根目录执行 `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python3 campaigns/resnet50-five-methods-full-20261026/report.py`，只读日志并更新报告，不启动训练。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')
    print({'status':summary['status'],'checks_passed':summary['synthetic_checks_passed'],
           'methods':[{k:v for k,v in row.items() if k in ['method','status','last_logged_update','last_epoch','tracking']} for row in rows]})

if __name__=='__main__':main()
