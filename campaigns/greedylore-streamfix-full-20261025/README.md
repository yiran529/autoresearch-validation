# GreedyLoRE streamfix 完整训练

用户已授权完整60M/130M预训练；使用修复commit8316c7eacc991173dee03f430ebec20efd99760b。原Engineer及失败记录保留。

130M优先在空闲GPU0–3启动：Muon、20,000步、warmup2,000、matrix LR .01/scalar LR .001。60M在原timing全部完成后使用GPU4–7：AdamW、8,393步、warmup1,000、LR .0025。

共同设置：seed1243、BF16、seq256、每卡batch128/全局512、梯度累积1、activation checkpointing开启、压缩启动1000、rank32/gap200/EF14、W&B online、不保存模型和优化器checkpoint。使用标准完整驱动，无diagnostic_guard或提前停止。

各模型独立tmux监督，不停止现有timing、不重跑或续训历史任务。状态分别见130m-status.json、60m-status.json；日志见对应controller.log。
