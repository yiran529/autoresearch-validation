# ResNet50五方法完整训练

状态：running；GPU：4,5,6,7；已通过合成检查：5/5。

CIFAR-10，ResNet50，40epoch，15640updates，每卡batch32、四卡全局128，seed1243。FP32模型，AdamW统一LR0.001、betas(0.9,0.999)、eps1e-8，weight decay0.0005，cosine到0，按epoch调度，无LR warmup。压缩从更新501开始，rank32/gap200；N20内部rank8/rejuvenation10。linear分类层不压缩。

五方法串行使用四张GPU。W&B online，project GreedyLore-CIFAR；不保存模型/优化器checkpoint。CIFAR10每epoch评估全部10000个测试样本。

| 顺序 / 方法 | 状态 | 最近训练日志 | 最近完整epoch | W&B |
|---|---|---|---|---|
| 1. GreedyLoRE | running | step700, loss1.2798 | epoch1, test_acc50.62% | [查看](https://wandb.ai/yiran0834-huazhong-university-of-science-and-technology/GreedyLore-CIFAR/runs/20ce14689e75cfe4) |
| 2. N23 | planned | — | — | — |
| 3. N31 | planned | — | — | — |
| 4. N20 | planned | — | — | — |
| 5. Dense | planned | — | — | — |

四个候选保留当前Engineer源码与关键超参数。新adapter只补充ResNet线性分类层名字映射；卷积展开使用候选原生实现。未修改原始项目或依赖。

GPU接入检查是205步合成数据、compression start2，包含更新203的刷新与逐步参数一致性检查。它与真实CIFAR10完整训练分开保存，不作为完整收敛结论。

远程目录：`/home/wyr/greedylore-validation/campaigns/resnet50-adamw-five-methods-full-20261026`；tmux：`glval-resnet50-adamw-full-20261026`。

查看进度：在scaffold根目录执行 `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python3 campaigns/resnet50-adamw-five-methods-full-20261026/report.py`，只读日志并更新报告，不启动训练。
