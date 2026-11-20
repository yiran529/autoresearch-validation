"""Produce a Chinese final report after all frozen cases validate."""
from pathlib import Path
from glval.common import read_json

HERE=Path(__file__).resolve().parent
LABEL={'dense':'Dense','greedylore-engineer-streamfix':'GreedyLoRE（streamfix）',
       'n023-engineer-maxu':'N23','n031-engineer-maxu':'N31','n020-engineer-maxu':'N20'}
ORDER=list(LABEL)

def main():
    data=read_json(HERE/'analysis.json')
    assert data['finished_count']==data['valid_count']==data['total']==41
    rows={r['setting']['id']:r for r in data['rows']}
    lines=['# 四卡 / 八卡 timing 设置扫描','',
      '41 项均成功并通过结果检查：8 个完整 setting × 5 个方法，加一项四卡350M默认 bucket 的 Dense 参考。每项只跑 seed1243 一次。',
      '', '## 测量设置','',
      '- 同一台8×RTX4090；四卡用GPU4–7，八卡用GPU0–7。顺序执行，期间独占资源。GPU互联为SYS，本机peer access检查不可用。',
      '- BF16、seq256、每卡batch1、无梯度累积；固定合成token缓存在GPU，排除数据读取。W&B关闭，不保存checkpoint。',
      '- Muon行开启激活重计算，AdamW行关闭。其他优化器参数沿用对应配置；这是不同setting的性能探索。',
      '- 预热21步、测量200步；压缩从iter2开启。测量更新22–221，计入一次GreedyLoRE的200步周期SVD，以及N20的周期成本。',
      '- 每步设备同步；主指标是各rank测量窗口总时间最大值÷200。全部rank的进程、有限loss、步数、hook调用与runtime哈希均通过检查。',
      '- GreedyLoRE使用已修复CUDA stream竞态的版本。可选刷新列表未初始化，刷新覆盖证据为固定源码周期、最终iter221和每个rank在更新203出现的实测峰值。',
      '', '这些是短程性能诊断，不是原800步正式timing协议，也不验证真实数据收敛。学习率仍处在warmup阶段。',
      '', '## 全部结果','', '单位ms/步，越小越快。默认bucket为PyTorch的None；大bucket为1024MiB，实际只有一个bucket。', '',
      '| Setting | Dense | GreedyLoRE | N23 | N31 | N20 |','|---|---:|---:|---:|---:|---:|']
    for row in data['rows']:
        vals=[f"{row['methods'][m]['mean_step_ms']:.2f}" if m in row['methods'] else '—' for m in ORDER]
        lines.append('| '+row['setting']['id']+' | '+' | '.join(vals)+' |')
    lines+=['','## 各组最快配置比较','',
            '在本次实际测过的bucket配置中，分别选择最快Dense和最快压缩方法。此处没有进行穷尽调优。','',
            '| 模型 / 优化器 / 卡数 | 最快Dense | 最快压缩方法 | Dense时间 / 压缩时间 |',
            '|---|---:|---:|---:|']
    for model,opt,world in [('130m','muon',4),('130m','muon',8),('130m','adamw',4),('130m','adamw',8),('350m','adamw',4),('350m','adamw',8)]:
        matching=[r for r in data['rows'] if r['setting']['model']==model and r['setting']['optimizer']==opt and r['setting']['world']==world]
        dense=min((r['methods']['dense'] for r in matching),key=lambda r:r['mean_step_ms'])
        compressed=min((r for row in matching for m,r in row['methods'].items() if m!='dense'),key=lambda r:r['mean_step_ms'])
        lines.append(f"| {model} / {opt} / {world}卡 | {dense['mean_step_ms']:.2f} | {LABEL[compressed['method']]} {compressed['mean_step_ms']:.2f} | {dense['mean_step_ms']/compressed['mean_step_ms']:.3f}× |")
    lines+=['','## 解读','',
      '增加卡数后，每卡batch保持1，因此全局batch从4变成8。每步变慢不等于总吞吐下降；下面是相同配置下Dense的弱扩展吞吐比：','']
    for prefix in ['130m-muon','130m-adam','350m-adam']:
        four=rows[prefix+'-4-default']['methods']['dense']['mean_step_ms']
        eight=rows[prefix+'-8-default']['methods']['dense']['mean_step_ms']
        lines.append(f'- {prefix}：四卡{four:.2f}ms，八卡{eight:.2f}ms，总吞吐为{2*four/eight:.3f}×。')
    lines+=['',
      '大bucket有利于当前压缩实现减少hook和设备同步。130M的压缩方法从默认8个bucket、约9次显式同步/步，减少到1个bucket、约2次同步/步。Dense自身只有外层每步同步，默认bucket还可以保留更多计算通信重叠。',
      '',
      '更宽的350M配置降低了embedding/head/norm等未压缩部分的比例：130M约36.67%，350M约17.82%。350M配置实际参数量367,969,280。压缩更容易超过同样使用大bucket的Dense，但仍需与默认bucket的Dense比较。',
      '',
      '八卡不是自动放大压缩收益的开关。当前稀疏AllGather增加接收数据和重建工作，N31在350M大bucket下由四卡126.85ms增至八卡149.96ms，额外时间主要出现在backward阶段。此处没有进一步profiler分解通信和重建各自占比。',
      '',
      '单次短测中几个百分点的差异不宜判定稳定优劣。后续选出值得验证的配置后，应采用重复测量，并用真实数据比较达到同等验证loss的总时间。',
      '', '## 复现资料','',
      '远程目录：`/home/wyr/greedylore-validation/diagnostics/timing-settings-20261025`。',
      '`analysis.json`保存完整结果与有效性证据；`provenance.json`保存脚本和方法身份；`cases.json`保存每项runtime文件哈希；每个case保留进程退出状态、全部rank计时、hook绑定及刷新证据。原40项manifest保存在`cases-original40.json`，额外Dense参考的追加记录在`extra-reference.json`。',
      '', '全部新代码和结果均留在独立scaffold，未修改CORAL、原始greedy_lore、候选源码或依赖。']
    (HERE/'report-zh.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
