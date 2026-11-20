"""Produce the final report only after all controlled cases have results."""
from pathlib import Path
from glval.common import read_json

HERE=Path(__file__).resolve().parent

def main():
    data=read_json(HERE/'analysis.json')
    assert data['status']=='complete', data['missing']
    cases={x['id']:x for x in data['controlled']}
    lines=['# 为什么压缩方法比 Dense 慢', '',
        '结论：通信量确实降低了，但当前工作负载中可暴露的DDP通信时间有限，压缩的额外算子、CPU调度、数据搬运和同步开销抵消了节省。搜索协议与当前正式设置不同；embedding/head未压缩、Muon自己的通信、GreedyLoRE周期SVD进一步限制收益。', '',
        '所有新GPU诊断仅使用空闲GPU4–7，原完整训练及源码未修改。新诊断使用真实模型、缓存的合成token、每卡batch1、BF16、相同优化器和activation checkpointing；压缩提前到第2步，预热11步，测量30个稳态步。它们用于解释性能，不替代原800步×3次正式timing，更不用于比较收敛。', '',
        '## 1. 端到端消融', '',
        '| 诊断配置 | 60M ms/步 | 130M ms/步 |', '|---|---:|---:|']
    for suffix,label in [('dense','Dense，默认分桶'),('no_ddp','去掉DDP梯度通信，仅作诊断'),
                         ('n23','N23，默认分桶'),('n31','N31，默认分桶'),('n20','N20，默认分桶'),
                         ('greedy_fixed','修复版GreedyLoRE，普通非刷新步'),
                         ('dense_big','Dense，1024MiB单bucket'),('n23_big','N23，1024MiB单bucket'),
                         ('n31_big','N31，1024MiB单bucket')]:
        lines.append(f'| {label} | {cases["60m-"+suffix]["mean_ms"]:.2f} | {cases["130m-"+suffix]["mean_ms"]:.2f} |')
    lines.extend(['', 'No-DDP仍保留完整前向、反向、裁剪、优化器更新；130M仍保留Muon自身的collective。它不是有效的分布式训练方法，也不是严格数学上界；短测的小幅差异不能解释为显著收益。', '',
        '显式设置25MiB与不传bucket_cap_mb并不完全相同：安装的PyTorch2.4默认还使用较小的首bucket。诊断保留原默认None，实测60M为5个bucket、130M为8个bucket；1024MiB为单bucket。Dense本身会通过分桶将All-Reduce与反向计算重叠，所以必须同时看Dense的大bucket结果，不能只为压缩方法换基线。', '',
        '## 2. 实际通信和同步', '',
        '下面是每rank每步的collective本地输入体积总和，不是物理PCIe/NIC流量。AllGather还会接收其他rank的数据，不能把本地压缩比当成端到端线速收益。', '',
        '| 模型 / 方法 | bucket数 | DDP通信输入 MiB | DDP API calls | 每步显式device sync，含步末1次 |',
        '|---|---:|---:|---|---:|'])
    for model in ['60m','130m']:
        for suffix in ['dense','n23','n31','n20','greedy_fixed']:
            c=cases[model+'-'+suffix]
            ops=', '.join(f'{x["op"]}:{x["count"]:g}' for x in c['calls'] if x['stage']=='backward')
            lines.append(f'| {model}/{suffix} | {c["buckets"]} | {c["gradient_local_input_MiB"]:.2f} | {ops} | {c["syncs_per_step"]:g} |')
    lines.extend(['',
        'N23/N31每个含稀疏张量的bucket需要额外分别AllGather索引和值，随后解码和逐tensor拷贝；FP32 EF增加了内存操作和类型转换。源码中的全设备同步使hook不能保持Dense那样的异步流水，部分AllGather还在当前stream上建立等待。同步的直接CPU等待并非全部损失，CPU不能提前排队、GPU与通信不能重叠也会影响关键路径。', '',
        '## 3. Profiler证据', '',
        'Profiler在独立的3步区间运行，测量段不启用profiler。下面仅用于定位热点，不能将profile中各项相加重建端到端时间。', '',
        '| 模型 / 方法 | CUDA launches/步 | NCCL GPU ms/步 | Top-k GPU ms/步 | QR GPU ms/步 | QR calls/步 |',
        '|---|---:|---:|---:|---:|---:|'])
    for model in ['60m','130m']:
        for suffix in ['dense','n23','n31','n20','greedy_fixed']:
            p=cases[model+'-'+suffix]['profile']
            comm=p['nccl_allreduce_device_ms_per_step']+p['nccl_allgather_device_ms_per_step']
            lines.append(f'| {model}/{suffix} | {p["cuda_kernel_launches_per_step"]:.0f} | {comm:.2f} | {p["topk_device_ms_per_step"]:.2f} | {p["qr_device_ms_per_step"]:.2f} | {p["qr_calls_per_step"]:g} |')
    lines.extend(['',
        '60M是56层、hidden256的深窄模型，130M是12层、hidden768；batch1且开启activation checkpointing时，60M有很多小kernel和重计算/Autograd调度。压缩又增加Python逐tensor处理、cat/copy/FP32转换等工作。N20稳态路径每步都做左右因子的两次归约及QR，并非仅周期刷新才有正交化成本。', '',
        '## 4. 刷新成本：来自原800步正式timing', '',
        '| 模型 | Greedy普通步 ms | 刷新步 ms | 刷新摊销增量 ms/步 |', '|---|---:|---:|---:|'])
    for model in ['60m','130m']:
        g=data['archived'][model]['greedylore-engineer']
        lines.append(f'| {model} | {g["regular_ms"]:.2f} | {g["event_ms"]:.2f} | {g["event_excess_amortized_ms"]:.2f} |')
    lines.extend(['',
        '每种模型3轮共12个刷新步，周期200；每次完整SVD约3秒。原130M正式timing中N23/N31相对Greedy的差值约16–17ms，而Greedy刷新摊销约15.8ms：大部分相对收益可由免除刷新解释，Dense本来就没有这项成本。上面新短诊断中的Greedy固定版只测普通步，不能与包含SVD摊销的原timing直接比均值。即使去掉刷新，旧Greedy普通步也慢于Dense；SVD不是唯一原因。修改刷新周期或换近似分解会影响方法语义/质量，不能当作无代价的工程优化。', '',
        '## 5. embedding/head和Muon的通信下限', '',
        '| 模型 | 未压缩参数占比（含norm） | 全梯度BF16 MiB | N23本地输入MiB | N31本地输入MiB |',
        '|---|---:|---:|---:|---:|',
        '| 60M | 27.05% | 115.74 | 35.26 | 36.58 |',
        '| 130M | 36.67% | 255.79 | 101.38 | 103.91 |', '',
        'lm_head与embed_tokens均为32000×hidden，按方法约定保持Dense。N23/N31的1/32稀疏率仅覆盖其余矩阵；索引也要发送。130M的Muon还独立AllGather梯度presence和分摊计算的正交化结果，这些不会被DDP梯度压缩消除。', '',
        '| 130M优化器设置 | Dense ms/步 | N31 ms/步 |', '|---|---:|---:|',
        f'| 当前分布式正交化 | {cases["130m-dense"]["mean_ms"]:.2f} | {cases["130m-n31"]["mean_ms"]:.2f} |',
        f'| 各rank本地正交化，诊断消融 | {cases["130m-dense_local_muon"]["mean_ms"]:.2f} | {cases["130m-n31_local_muon"]["mean_ms"]:.2f} |', '',
        '本地正交化增加每rank计算，减少optimizer communication。此处只是性能比较，正式更改前仍需数值等价性与训练验证；不要把这个共有优化归功于某个压缩方法。', '',
        '## 6. 设置、硬件和搜索目标', '',
        '| 项目 | autoresearch timing | 当前timing |', '|---|---|---|',
        '| 模型/优化器 | 130M AdamW | 60M AdamW / 130M Muon |',
        '| Activation checkpointing | 关闭 | 开启 |',
        '| 分桶 | 1024MiB，实测1个 | 默认None，实测5/8个 |',
        '| 压缩启动 | 100 | 1000 |',
        '| 测量时LR | 仍在warmup | 已过warmup |', '',
        'N23被选中的U记录为相对Greedy工程基线的speedup（不是相对Dense），quality是inherited/provisional。能赢Greedy并不意味着能赢Dense，尤其Greedy有全SVD摊销；换成多bucket与Muon后，旧U也不能直接外推。', '',
        '本次测试的GPU4–7均为4090，彼此拓扑均为SYS；CUDA peer-access矩阵实测跨卡全False。当前NCCL_P2P_DISABLE=1、SHM可用。NCCL官方说明此变量只禁用P2P transport；本机既然peer access不可用，仅移除此开关并不能推断会变快。未发现额外的NCCL_ALGO/PROTO环境或nccl.conf覆盖。', '',
        '## 7. 建议的优先顺序', '',
        '1. 搜索与验收都加入Dense，用目标recipe的模型、优化器、精度、分桶和activation checkpointing；同时记录通信体积、关键路径、质量，不仅比较相对Greedy的U。',
        '2. 优先改N23/N31的公共工程路径：缓存bucket布局与工作区，融合逐tensor的EF/pack/scatter/copy，合并索引和值的消息；用正确的CUDA stream/event与device-aware Future减少全设备同步并恢复重叠。不能直接删synchronize，否则可能重演NaN竞态。',
        '3. 分桶作为共同设置调优，Dense和压缩都测，比较各自最佳设置；不要只用大bucket拖慢Dense来制造相对加速。',
        '4. N20先处理QR/多阶段collective成本；Greedy先处理SVD与全基底评分成本。改变rank、刷新周期、投影或误差反馈属于方法/质量决策，需单独验证。',
        '5. 对真正通信受限的更大模型、更多GPU或跨节点网络再做扩展验证。当前结果只说明本机这组模型/设置不划算，不代表梯度压缩普遍无效。', '',
        '证据文件：analysis.json、各case的result.json/profile-summary.json/trace.json、原始800步timing。', '',
        '诊断限制：每case仅一个短稳态测量区间，存在共享CPU/内存与阶段间波动；部分GPU profile有较大观测开销；cached synthetic token与真实C4分布不同；通信输入字节不是物理wire计数。一个case在模型初始化前遇到EADDRINUSE，失败记录保留，已用新静态rendezvous启动尝试继续；该失败未计入性能结果。', '',
        '参考：[PyTorch DDP设计与重叠](https://docs.pytorch.org/docs/stable/notes/ddp.html)、[DDP bucket参数](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、[NCCL2.20 transport环境变量](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2205/user-guide/docs/env.html)。'])
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')
    print({'report':str(HERE/'report.md'),'cases':len(cases)})

if __name__=='__main__':main()
