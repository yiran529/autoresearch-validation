# 130M Muon 接续实验

用户已授权执行。等待依赖进程退出；补完原 60M 已计划而未启动的任务，再开始 130M。失败不重跑、不停止其他任务。全部文件位于独立 scaffold。

130M: 原有 Muon recipe，matrix LR .01、scalar LR .001、20k steps、warmup2k、BF16、seq256、batch128×4、activation checkpointing开启。checkpoint_policy=none；W&B online。

130M timing: 预热2201步，测量2202–3001，共800步，3次重复。Dense在各阶段最后。

源码沿用已固定候选。静态预检和17项CPU测试通过；本次130M尚无GPU运行或收敛验证。短测不作为预训练通过证据。

状态：status.json；日志：continuation.log；启动记录：started.json；汇总：continuation-result.json。已知GreedyLoRE在60M训练失败，130M保留原版本独立记录结果。

用户更新调度：Dense已提前到0–3卡；任一4卡组预训练完成即开始timing，全部timing固定在首组释放的4卡上，Dense timing最后。权威调度状态改为first-free-timing-status.json。timing可能与另一组预训练并行，报告需要注明共享主机干扰。仅替换暂停中的旧调度父进程，不停止或重启训练。

调度切换记录见scheduler-handoff-incident.json：N20旧监控父进程退出，实际训练保持原PID。新监控已接管并持有8卡锁。N20最终产物可独立核验，但退出码不可恢复，报告保留未知状态，不伪造正常退出。

用户最新设置：130M 与60M timing均每卡batch1、W&B disabled；待任一4卡组完成后，固定该组依次跑130M、60M timing。每个recipe保持三次重复、Dense最后。旧timing计划留存不执行。完整预训练不变。
