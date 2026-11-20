# LLaMA60M Engineer/max-U campaign

所有工作在独立 scaffold，原项目只读。

- 模型：llama_60m_table_iv，BF16，4卡/任务，local batch128/global512，AdamW lr0.0025，warmup1000，总8393步。
- activation checkpointing 开启；不保存模型/优化器 checkpoint。
- W&B online，project GreedyLore-Validation，group llama60m-engineer-maxu-20261024。
- 方法实现/最大U参数来源见 selections.json。N000 Engineer 用作 GreedyLoRE；N020 参数来自其 Research 最大U版本，关键赋值已与Engineer版核对一致。
- 预训练先行：GPU0-3 GreedyLoRE→N031；GPU4-7 N023→N020。全部四组完成后 GPU0-3 跑 Dense。
- 所有预训练完成后，GPU0-3 做独立 timing；warmup1201，测量800，3轮；所有非Dense轮次结束后再跑Dense三轮。
- timing 的W&B初始化与上传发生在测量窗口结束后。
- smoke v1 因第0步开启压缩与DDP bucket重建冲突而失败，记录完整保留；v2等待2步，再验证实际方法路径。

查看 bootstrap-status.json、status.json 和 controller.log。各run的state.json记录每个job状态。
当前源代码更新后，旧的未执行演示计划可能因完整性检查失效；本campaign计划已按当前代码重新冻结。

汇总（不启动训练）：

```bash
cd /home/wyr/greedylore-validation
PYTHONPATH=. python3 -B campaigns/llama60m-engineer-maxu-20261024/summarize.py
```

完整预训练跨两组GPU，总墙钟时间只作参考；速度结论使用后续同组GPU的独立timing。
