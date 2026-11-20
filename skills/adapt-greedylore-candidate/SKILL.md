---
name: adapt-greedylore-candidate
description: 将指定 CORAL autoresearch 候选或 Git SHA 接入独立 GreedyLoRE Validation scaffold，编写方法 adapter、审阅依赖与语义并生成可复现的完整实验计划。用于候选适配，不负责继续搜索新算法。
---

# 候选方法适配

在本 scaffold 内完成工作。先阅读项目根目录 README.md 和 AGENTS.md，使用现有 CLI/API，避免为每个候选重写训练流程。
CORAL、greedy_lore、搜索 run 与现有环境均是只读来源。所有新文件与运行副本留在 scaffold。

## 解析用户指定的方法

输入可以是源仓库与 SHA，也可以是 run + Node + Research/Engineering。
用户明确指定 SHA、阶段、实现版本或超参数来源时，优先遵守该指定。

只指定 run + Node/方法、没有指定版本时，默认选择该节点 **U 最大的 Engineer 版本**：

- 从正式 Attempt 记录筛选已完成、评估有效的 `mode=engineering` 候选，读取权威 `metadata.utility.value`；只使用有限数值，不把缺失/null 当零，也不以 score、最新提交或 Research 的 U 替代。
- 在相同 U 定义和可比较评估协议内按 U 降序选择；数值完全相同时按完整 SHA 字典序选择并记录并列情况。多组协议不可比时先说明差异，请用户选择比较组，不能混排后声称“最大 U”。
- 默认实现和关键超参数都来自选中的同一个 Engineer 版本。若用户另外指定参数来自其他最大 U 版本，则分别记录实现 SHA 与参数来源 SHA，并核实 Engineer 实现保留了该参数语义；不得默默切换到 Research 实现。
- 若没有具备有效 U 的 Engineer 候选，列出可用 Engineer 版本及 U 缺失原因，再请求选择；不自动退回 Research 或猜测默认版本。Root/GreedyLoRE 基线没有 U 时也适用。
- 在 method.json/adaptation.md 中记录选中的完整 SHA、Attempt/run ID、U 数值、U 定义、参考版本/协议及关键超参数来源；区分工程质量的 inherited/provisional 标记和新实测结果。

从公开 Node/Attempt/评估记录定位固定完整 SHA，不能用当前 worktree HEAD 或最新 commit 代替。
只有来源 run/Node 无法唯一定位、缺少有效 U 或协议不可比等实际歧义才询问；正常存在唯一最大 U Engineer 时直接选择，不再次确认。
候选选择和代码身份在适配开始时固定，不随搜索进展变化。修改本 skill 的默认规则不追溯改动已冻结的计划或正在运行的实验。

使用 `python3 -m glval import-method --id METHOD --repo REPO --commit SHA` 创建 draft。
从 source_manifest.json 的 changes_from_base 开始审阅；再读取方法 Spec、submission 和实际评估命令等相关证据。
不要导入来源不明的候选代码来做静态检查。

## 明确方法边界

在 adaptation.md 简要记录：方法作用位置、优化器关系、跨 step 状态、压缩/refresh/EF 的触发条件、必要文件与参数、模型形状假设。
查清改动的训练循环/优化器是否承载算法必需行为。只搬 hook 可能遗漏算法。
如果需要改变 optimizer/update-rule 接口，记录 blocked 原因和最小扩展建议；不通过把必需改动标为 not_required 来强行接入。

## 编写薄 adapter

编辑导入方法的 method.json 和 adapter.py，直接调用固定 source 中的候选实现。
按实际需要补齐 helper overlay，遵守公共训练目录保护；不要改其他方法或全局参数来迁就候选。
逐项填写 reviewed_changes，使用 overlay 或 not_required，理由应能解释该差异为什么进入或不进入实际执行路径。
新增 helper 即便不在 changes_from_base 中，也必须记录来源及依赖。
supported_workloads 和 support_notes 按实际适用范围填写。ResNet 的卷积张量、参数排除规则和 hook 注册需要单独检查；不能从 LLaMA 成功推断 ResNet/GLUE 可用。

保持 recipe 决定的模型、数据、基础优化器、LR、训练预算和评估规则。
方法参数可显式按 workload 建立命名变体；涉及压缩位置、更新规则、EF 或未定义张量展开的改变是算法决策，不是机械适配。
合理的接口映射与状态传递可自行完成；语义不确定时提出具体问题，同时推进不依赖该决策的工作。

## Timing 默认配置

本项目后续 timing 默认每卡 `batch_size=1`、4卡全局 batch4、梯度累积1，W&B `mode=disabled`，仅保存本地结果。用户明确指定时可覆盖。默认值见项目根目录 `timing_defaults.json`；recipe 模板使用其中列出的 bsz1-local 配方。
这项默认仅适用于 timing，不改变完整预训练的 batch 或 W&B online 设置。保留其他已选 recipe 超参数，包括 activation checkpointing、优化器、学习率与测量窗口。更改已冻结设置时创建新计划，保留历史结果并调整待执行队列，不修改冻结 runtime。

## 检查与计划

完成审阅后将 status 设置为 adapted（作者声明，不是实测结论）。
运行 `python3 -m glval check METHOD --workload WORKLOAD`。
新增本候选实验 JSON，调用 plan / preflight，确保固定代码、配方、adapter hash 与输入路径清楚。
改 adapter 或配置后生成新计划；不要修改已冻结 runtime 来继续实验。

没有用户明确要求执行时，交付到静态检查、实验计划和运行命令。
probe 也会执行参数更新；“不开始训练”的要求同样禁止 probe。不能用本 skill 推导出执行授权。
如果用户授权运行，先检查资源并使用项目 probe/run 入口；长任务交给 tmux/调度器，不用 Agent 循环等待。
合成 probe 仅验证局部接入，按实际参数判断是否进入压缩、refresh 等分支，没覆盖就明确记录。
需验证真实 workload 时使用独立且明确标记的短配置/脚本，不把缩短后的配置称为完整配方通过；适配一致性应尽可能用固定输入和初态比较梯度/参数更新与状态。

## 交付

输出方法目录、固定来源、支持 workload、适配改动、静态/合成/真实运行各自结果、未覆盖项和下一条可执行命令。
用 ready / partial / blocked 表示适配状态，并同时注明证据级别；只有静态检查时写“static only，GPU/语义尚未验证”。
完整实验另行报告预算完成情况和质量/效率，不把适配完成说成方法已有效。
不自动修改 Node 结论、回写 CORAL、调参、启动新研究或安装全局 skill。
