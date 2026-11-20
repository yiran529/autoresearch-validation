"""Explicit timing defaults update: batch1, W&B disabled. No training during preparation."""
from copy import deepcopy
from pathlib import Path
from glval.common import ROOT,read_json,write_json
from glval.plans import make_plan,preflight

HERE=Path(__file__).resolve().parent

def main():
    created=[]
    for model,old_recipe,old_experiment in [
        ('130m','llama130m-muon-timing-online-v1',HERE/'llama130m-muon-timing-20261025.json'),
        ('60m','llama60m-adamw-timing-online-v1',ROOT/'campaigns/llama60m-engineer-maxu-20261024/llama60m-timing.json')]:
        recipe=deepcopy(read_json(ROOT/'recipes'/(old_recipe+'.json')))
        recipe['id']=f'llama{model}-'+('muon' if model=='130m' else 'adamw')+'-timing-bsz1-local-v1'
        recipe['args']['batch_size']=1
        recipe['notes']='Timing default: per-device batch1, global batch4, no gradient accumulation; W&B disabled. All other recipe settings and frozen method identities retained. Does not match full-training batch128.'
        path=ROOT/'recipes'/(recipe['id']+'.json')
        assert not path.exists(),path
        write_json(path,recipe)
        experiment=deepcopy(read_json(old_experiment))
        experiment.update(id=f'llama{model}-timing-bsz1-local-20261025',recipe=recipe['id'])
        experiment['execution']['tracking'].update(mode='disabled',group=experiment['id'])
        exp_path=HERE/(experiment['id']+'.json')
        write_json(exp_path,experiment)
        run=make_plan(exp_path)
        assert preflight(run)['status']=='static_pass'
        plan=read_json(run/'plan.json')
        assert plan['recipe']['args']['batch_size']==1
        assert all(j['tracking']['mode']=='disabled' for j in plan['jobs'])
        assert all(j['command'][j['command'].index('--batch_size')+1]=='1' for j in plan['jobs'])
        created.append(str(run))
    config=read_json(HERE/'campaign.json')
    write_json(HERE/'campaign-before-bsz1-local.json',config)
    config.update(timing_run=created[0],additional_timing_runs=[created[1]],
                  timing_policy='First free group; fixed same four GPUs; 130M then 60M; Dense last within each recipe; batch1 and W&B disabled.')
    write_json(HERE/'campaign.json',config)
    write_json(ROOT/'timing_defaults.json',{'per_device_batch_size':1,'world_size':4,
               'gradient_accumulation':1,'tracking_mode':'disabled','scope':'Future timing only; full pretraining unchanged',
               'recipe_templates':['llama130m-muon-timing-bsz1-local-v1','llama60m-adamw-timing-bsz1-local-v1']})
    skill=ROOT/'skills/adapt-greedylore-candidate/SKILL.md'
    text=skill.read_text()
    marker='## 检查与计划'
    instruction=('## Timing 默认配置\n\n'
        '本项目后续 timing 默认每卡 `batch_size=1`、4卡全局 batch4、梯度累积1，W&B `mode=disabled`，仅保存本地结果。用户明确指定时可覆盖。默认值见项目根目录 `timing_defaults.json`；recipe 模板使用其中列出的 bsz1-local 配方。\n'
        '这项默认仅适用于 timing，不改变完整预训练的 batch 或 W&B online 设置。保留其他已选 recipe 超参数，包括 activation checkpointing、优化器、学习率与测量窗口。更改已冻结设置时创建新计划，保留历史结果并调整待执行队列，不修改冻结 runtime。\n\n')
    assert marker in text and '## Timing 默认配置' not in text
    skill.write_text(text.replace(marker,instruction+marker,1))
    with (ROOT/'README.md').open('a') as f:
        f.write('\n## Timing 默认值\n\n后续 timing 默认每卡 batch1（4卡全局4）、W&B disabled，仅保留本地结果；完整预训练设置不变。默认与模板见 `timing_defaults.json`。历史 batch128 / online 的已冻结计划和结果保留。\n')
    with (HERE/'README.md').open('a') as f:
        f.write('\n用户最新设置：130M 与60M timing均每卡batch1、W&B disabled；待任一4卡组完成后，固定该组依次跑130M、60M timing。每个recipe保持三次重复、Dense最后。旧timing计划留存不执行。完整预训练不变。\n')
    write_json(HERE/'timing-bsz1-local-preparation.json',{'runs':created,'static_preflight':'pass','gpu_execution':False})
    print({'new_timing_runs':created,'batch_per_device':1,'wandb_mode':'disabled','static_preflight':'pass'})

if __name__=='__main__':main()
