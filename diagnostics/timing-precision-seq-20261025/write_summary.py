"""Final combined screen and selected-setting report; never claims quality validation."""
from pathlib import Path
from glval.common import read_json,write_json,file_hash

HERE=Path(__file__).resolve().parent

def main():
    screen=read_json(HERE/'analysis.json');confirm=read_json(HERE/'confirmation/analysis.json')
    greedy=read_json(HERE/'greedy-all/analysis.json')
    assert screen['finished']==screen['valid']==32
    assert confirm['finished']==confirm['valid']==6
    assert greedy['finished']==greedy['valid']==7
    selection=read_json(HERE/'confirmation/selection.json')
    row=confirm['rows'][0];best_dense=min(row['variants'][v]['mean_step_ms'] for v in ['dense_default','dense_big'])
    labels={'dense_default':'Dense默认bucket','dense_big':'Dense大bucket','n23_big':'N23大bucket',
            'n20_big':'N20大bucket','n31_big':'N31大bucket','greedy_big':'GreedyLoRE streamfix大bucket'}
    greedy_by_setting={r['setting']['id']:r['variants']['greedy_big'] for r in greedy['rows']}
    greedy_by_setting[row['setting']['id']]=row['variants']['greedy_big']
    assert len(greedy_by_setting)==8
    lines=['# FP32 与短序列 timing 实验','',
        '本轮FP32明显改善N23/N20相对同精度Dense的速度收益。单独缩短到seq64没有明显降低步耗时，且GreedyLoRE在FP32短序列下的完整刷新步大幅变慢。',
        '',
        '完成32项筛选，并补齐8个setting的GreedyLoRE，其中1项复用最有利setting的6项完整对照。全部45项有效测量通过检查；每项统一seed1243，未进行多seed收敛实验。',
        '', '## 设置','',
        '八卡RTX4090，AdamW，每卡batch1、全局batch8，无梯度累积。关闭激活重计算、W&B和模型/优化器checkpoint保存。固定合成token在GPU缓存；seq64使用同一seq256输入的前64个token。',
        '', 'FP32为参数和梯度均FP32，无autocast，TF32关闭。每个rank均检查实际参数/梯度dtype、输入形状和matmul precision。BF16控制组使用相同TF32策略。',
        '', 'Dense同时比较默认bucket和2048MiB大bucket；压缩方法使用2048MiB。大bucket实际为1个，避免350M FP32在旧1024MiB设置下被分成多个bucket。',
        '', '筛选每项预热21步、测量100步；选中setting补测预热21步、测量200步。从iter2启用压缩。补测包含GreedyLoRE更新203的一次SVD刷新，N20包含20个rejuvenation周期。结果为最慢rank的测量窗口总时间÷步数，包含每步设备同步。',
        '', '## 筛选结果','',
        '单位ms/步。Dense列取本轮同模型、同精度、同seq下两种bucket中更快的结果；压缩后的吞吐比=最快Dense时间÷压缩方法时间。Dense/N23/N20为100步筛选，GreedyLoRE为200步以计入完整SVD周期；最有利setting的统一200步对照见下一节。', '',
        '| 模型 / 精度 / seq | 最快Dense | GreedyLoRE | N23 | N20 | 最快压缩吞吐比 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in screen['rows']:
        dense=min(r['variants'][v]['mean_step_ms'] for v in ['dense_default','dense_big'])
        n23=r['variants']['n23_big']['mean_step_ms'];n20=r['variants']['n20_big']['mean_step_ms']
        gm=greedy_by_setting[r['setting']['id']]['mean_step_ms']
        lines.append(f"| {r['setting']['id']} | {dense:.2f} | {gm:.2f} | {n23:.2f} | {n20:.2f} | {dense/min(gm,n23,n20):.3f}× |")
    lines+=['','## 最有利setting的补测','',
        f"根据筛选中最大的相对Dense收益，选择 `{selection['setting']['id']}`。以下是独立200步测量窗口，不能与上表100步结果混成重复实验均值。仍使用同一个seed1243。",'',
        '| 方法 | ms/步 | 相对本组最快Dense的吞吐比 |','|---|---:|---:|']
    for variant,label in labels.items():
        ms=row['variants'][variant]['mean_step_ms']
        lines.append(f'| {label} | {ms:.2f} | {best_dense/ms:.3f}× |')
    lines+=['','## GreedyLoRE刷新成本','',
        '采用计时窗口最慢rank的步计时，分别统计199个普通步和更新203的完整刷新步。额外成本=(刷新步−普通步均值)÷200。这是刷新步整体耗时，包含SVD及相关工作，不是单个SVD算子的独立profile。','',
        '| Setting | 普通步ms | 刷新步ms | 每步摊销的额外成本ms |','|---|---:|---:|---:|']
    for r in screen['rows']:
        g=greedy_by_setting[r['setting']['id']]
        lines.append(f"| {r['setting']['id']} | {g['regular_mean_ms']:.2f} | {g['refresh_step_ms']:.2f} | {g['amortized_refresh_excess_ms']:.2f} |")
    lines+=['','130M与350M FP32本轮seq64的刷新步都明显慢于seq256，而普通步较接近。刷新时间受到短序列影响的原因尚未单独定位，不能将缩短序列的FLOPs变化直接解释为刷新成本同比降低。',
        '', '## 缩短序列的效果','',
        '同一模型精度下，seq64每步只有seq256的四分之一token。下面同时报告Dense步耗时与token吞吐比，避免把步数等同于训练工作量。','',
        '| 模型 / 精度 | seq256 Dense ms | seq64 Dense ms | seq64 / seq256 token吞吐 |',
        '|---|---:|---:|---:|']
    byid={r['setting']['id']:r for r in screen['rows']}
    for model in ['130m','350m']:
        for dtype in ['bf16','fp32']:
            a=byid[f'{model}-{dtype}-s256'];b=byid[f'{model}-{dtype}-s64']
            t256=min(a['variants'][v]['mean_step_ms'] for v in ['dense_default','dense_big'])
            t64=min(b['variants'][v]['mean_step_ms'] for v in ['dense_default','dense_big'])
            lines.append(f'| {model} / {dtype} | {t256:.2f} | {t64:.2f} | {.25*t256/t64:.3f}× |')
    lines+=['','## 通信输入量','',
        '下表是每rank每步逻辑collective的输入MiB总和，包含值和索引；不是物理链路流量，AllGather的接收放大和NCCL传输协议不在这个数中。seq不改变参数/梯度规模。','',
        '| 模型 / 精度，seq256 | Dense | N23 | N20 |','|---|---:|---:|---:|']
    for model in ['130m','350m']:
        for dtype in ['bf16','fp32']:
            r=byid[f'{model}-{dtype}-s256']
            amounts=[sum(p['local_input_bytes'] for p in r['variants'][v]['local_input_payload'])/2**20 for v in ['dense_default','n23_big','n20_big']]
            lines.append(f'| {model} / {dtype} | '+' | '.join(f'{v:.2f}' for v in amounts)+' |')
    lines+=['','## 解释与边界','',
        'FP32提高了梯度通信量，而当前压缩方法的一部分编码、误差反馈和调度成本不会同比增加，因此更容易出现相对FP32 Dense的速度收益。这里没有用profiler把通信、重建和计算逐项分离，不能将全部收益归因于单一因素。',
        '', '缩短seq主要减少序列相关计算，梯度字节数、逐参数处理和大量调度成本仍在。在本轮batch1设置下，缩短seq没有带来接近4倍的步速度提升。',
        '', 'FP32的相对收益应与同精度Dense比较；改变精度和序列长度后，还需要比较绝对token吞吐和达到目标验证loss的时间。这轮不包含真实数据收敛验证。',
        '', '所有方法使用冻结Engineer版本和既定参数，GreedyLoRE使用已修复stream竞态的版本。原始CORAL、greedy_lore、候选源码与依赖均未修改。',
        '', '一次Dense首次尝试在TCPStore监听阶段遇到EADDRINUSE，未启动模型；原日志保留，之后用临时端口范围之外的端口显式重试。各rank的完成预算、有限loss、精度、bucket和runtime文件哈希均检查。',
        '', '## 复核资料','',
        '远程目录：`/home/wyr/greedylore-validation/diagnostics/timing-precision-seq-20261025`。筛选配置和结果在根目录，最有利setting补测在`confirmation/`，其余GreedyLoRE在`greedy-all/`，失败归档见`retries.json`。']
    (HERE/'summary-zh.md').write_text('\n'.join(lines)+'\n')
    write_json(HERE/'combined-results.json',{'screen':screen,'confirmation':confirm,'greedy_all':greedy,'selection':selection,
        'report_script_hash':file_hash(HERE/'write_summary.py')})
    print({'screen_valid':32,'confirmation_valid':6,'extra_greedy_valid':7,'greedy_settings':8,'selected_setting':selection['setting']['id']})

if __name__=='__main__':main()
