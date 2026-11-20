# GreedyLoRE Validation

独立候选评测 scaffold。CORAL、原始 greedy_lore、搜索 run 都只作为只读来源。
核心 CLI 只依赖 Python 3.10+ 标准库，无需安装包；训练使用显式指定的现有 Python 环境，不升级任何依赖。

当前状态：候选快照、adapter 桥接、静态检查、实验冻结、显式 GPU probe/训练入口、完成性检查和报告已实现。
已完成多组 GPU 检查及正式训练，具体证据见相应实验归档。 `static_pass` 只表示静态检查通过。

## 安全的起点：不启动训练

```bash
cd /home/wyr/greedylore-validation
python3 -m glval --help
python3 -m glval check original_greedylore --workload llama
python3 -m glval plan examples/llama130m-muon.json
python3 -m glval preflight runs/llama130m-muon-example
python3 -m glval status runs/llama130m-muon-example
python3 -m glval report runs/llama130m-muon-example
```

`plan` 只构建独立 runtime 和命令，不导入候选 Python，不加载数据，不探测 GPU，不启动子训练进程。
`preflight` 检查冻结代码、Python 语法与必要路径是否存在，不验证数据内容或环境依赖。
尚未运行的报告会明确显示 `unverified`，没有实测比较。

## 接入候选

```bash
python3 -m glval import-method --id n012-research \
  --repo /path/to/coral-run/repo --commit FULL_COMMIT_SHA
```

导入从 Git 对象读取已提交源码，忽略工作区未提交修改；不改源仓库，不复制模型、checkpoint、数据或环境。
输出在 `methods/n012-research/`：

- `source/` 与 `source_manifest.json`：固定源码、文件 hash、相对固定基线的改动。
- `method.json`：方法配置、支持范围和改动审阅。
- `adapter.py`：薄接口，调用候选实现。
- `adaptation.md`：语义、依赖、适配理由、验证与限制。

导入的方法默认为 `draft`，不能直接计划实验。用项目内 skill 指导 agent 完成适配。
它需要填写 `supported_workloads`、每个 workload 的 `support_notes`、`reviewed_changes`，最后把 `status` 改为 `adapted`。
`adapted` 是作者声明，不是 GPU 验证结论。

`reviewed_changes` 对每个运行相关差异给出：

```json
{
  "comm_hooks/subspace_hook.py": {
    "disposition": "overlay",
    "reason": "候选方法的投影与误差反馈实现，原样进入 runtime"
  },
  "c4/run_llama_pretraining.py": {
    "disposition": "not_required",
    "reason": "仅添加搜索期诊断输出；逐项确认没有方法必需的更新逻辑"
  }
}
```

不要把方法必需的优化器/训练循环改动标成 `not_required` 来绕过检查；这种方法超出 v1 `ddp_comm_hook` 接口，应标明阻塞原因并设计扩展。
辅助模块通过 `overlay` 显式加入；不能覆盖 `optimizer/`、`c4/`、`glue/`、`pytorch-cifar/` 或 `glval/`。
新辅助模块若不属于这些基线目录，agent 仍须审阅并在 adaptation.md 说明来源和导入关系。
修改固定 source 后必须重新导入新版本；修改 adapter/方法配置后必须生成新计划。

## Adapter API

```python
class MethodAdapter:
    def validate(self, context, config): ...
    def attach(self, model, optimizer, process_group, args, context, config): ...
    def describe(self): ...
```

`validate` 判断真实上下文是否适用；`attach` 必须且仅注册一个 DDP hook；`describe` 返回可 JSON 序列化的信息。
context 提供 workload、rank、world_size、recipe。config 提供方法参数及 recipe 固定的 `start_compress_iter`。
参数可扩展，但不能用方法参数覆盖训练预算、优化器、学习率或数据。
默认 adapter 使用候选的 `comm_hooks.utils.register_comm_hook_for_ddp_model`，不重写算法。
桥接记录实际 hook 文件/hash、状态类型、有效参数、每 rank 调用数及可用状态遥测。
注册与调用证据不等于算法等价，也不证明所有刷新事件已覆盖。

## 配方和实验

`recipes/*.json` 固定 workload/训练预算/优化器/精度/数据/评估规则；`methods/*/method.json` 固定方法参数；实验文件负责组合方法、基线和 seeds。

| Recipe | 预算 | 说明 |
|---|---|---|
| llama130m-muon-full-v1 | 20,000 updates | 对应当前搜索的完整 schedule，matrix LR .01 |
| llama60m-adamw-full-v1 | 8,393 updates | 明确 warmup=839；精确复现历史 run 前需核对其配置 |
| resnet18-cifar10-muon-full-v1 | 40 epochs | 10,000 test samples；方法 rank/gap 独立配置 |
| glue-sst2-bert-full-v1 | 3 epochs | 本地模型/数据路径模板，默认尚未配置 |

配方是可审阅的起点，不声明已复现论文结果。ResNet 原 paper arm 使用 rank64/gap750；当前通用 original_greedylore 方法是 rank32/gap200。
如需原 paper arm，对方法另建命名变体，不静默修改已有基线。GLUE 报告标为 validation，不声称 held-out test。
v1 限制 LLaMA/GLUE accumulation=1，避免继承旧入口的累积语义问题；不支持精确 checkpoint resume。
基线训练驱动来自 `vendor/provenance.json` 指定的原始提交，保留原文件；每次计划只在运行副本中进行显式 AST 桥接。
本地 tokenizer 替换、结果预算字段、CIFAR hook 接口和关闭自动下载是桥接范围。CIFAR 专属通信字节估算被禁用，不用于新方法比较。
语义相关的候选训练入口修改不会自动合并到 scaffold。

可以复制示例实验文件，填写候选方法 ID、固定 recipe、seeds、执行 Python。
修改 JSON 后重新 plan，已有 run 不覆盖。环境敏感值不写配置，跟踪默认关闭，HF 网络默认禁用。

## 未来明确需要执行时

以下命令会使用 GPU/执行更新，**创建项目时不运行**：

```bash
# Synthetic DDP 接入检查。默认 8 步很可能没有进入实际压缩阶段。
python3 -m glval probe runs/RUN --job METHOD-s1243 --gpus 0,1,2,3 --steps 8 --execute

# 完整实验，在 tmux 或调度作业中执行。
python3 -m glval run runs/RUN --gpus 0,1,2,3 --execute
```

probe 使用合成模型（ResNet 分支含 4D 卷积梯度），不是实际 workload 等价性验证；不修改原配方周期。
GPU 锁只协调本项目。使用明确划分的 GPU allocation，不能认为它与 CORAL 的锁互通。
运行器启动前检查 GPU UUID、已有 compute 进程、显存及利用率；不循环抢卡。
同一个计划中的所有方法串行使用同一组 GPU。它不能隔离其他进程后来占卡，需要外部资源分配保证。
长训练不由 agent 轮询等待。日志写在 job output；失败保留证据并继续其他 job，Ctrl-C 只终止本 runner 启动的进程组。
不提供隐式重跑或续训；重新计划为新 attempt。SIGKILL/断电留下 running 状态时，先人工核对记录的 PID，不自动杀进程。

## 报告与验证边界

结果需同时通过：进程成功、预算/样本条件、有限质量指标、每 rank hook 执行和源码身份检查。
报告提供逐 seed 指标差和总墙钟加速比，不强制总分；总墙钟包含初始化与评估，不是纯训练速度。
没有自动缓存复用、科学胜出判定、自动调参或自动回写 CORAL。
当前没有实际 workload 多卡 smoke、性能对照或收敛证据。完成接入后再按用户授权补充，不把静态检查标成完整验证。

## CPU 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

测试仅在本项目临时目录中创建小 Git fixture、解析代码、冻结计划并检查结果；不导入 torch、不加载数据、不运行训练。

## Skill

入口：`skills/adapt-greedylore-candidate/SKILL.md`。它放在本项目内，没有安装到全局 skills。
可直接让 agent 读取这个文件并按它执行。Skill 只提供指导，实际执行仍受用户当前授权范围约束。

## Timing 默认值

后续 timing 默认每卡 batch1（4卡全局4）、W&B disabled，仅保留本地结果；完整预训练设置不变。默认与模板见 `timing_defaults.json`。历史 batch128 / online 的已冻结计划和结果保留。

## 版本管理

代码、固定方法源码、配方和实验定义纳入 Git。正式实验统一使用 `python3 -m glval`，
新计划默认创建 `runs/<experiment-id>/`；`plan --out` 也只允许本项目 `runs/` 下的新目录，
拒绝越界软链接和覆盖已有实验。不要另建 controller 或直接调用底层构建器启动正式实验。
特殊能力应先接入统一接口并验证，不能通过忽略配置或历史专用脚本绕过。
统一计划已接入历史 PowerSGD 的 dense embedding/head 排除桥接、被动 bucket 记录，
以及配方显式声明的 `model_architecture: ResNet50`；适配只修改新运行副本，来源哈希保存在
`jobs/<job>/formal_adaptations.json`。这些 CPU 构建检查不代表新的训练或测速验证。

正式实验的运行副本、冻结计划、指标、报告、状态、日志和 checkpoint 均写入对应 run。
新 rolling checkpoint 使用相对于 job output 的 `root`（默认 `checkpoints`），不再写入
历史 `/dev/shm` 位置。共享缓存、资源锁仍分别位于 `cache/`、`locks/`。
Git 白名单保留核心结果、配置、各 rank 执行摘要和来源记录；日志、trace、逐步 JSONL、
checkpoint 与完整运行副本继续忽略。忽略规则不会删除磁盘文件。

### 精简结果发布

原始 `runtime-rank-*.json` 和 timing 的逐步数组保留本地。提交前可运行
`PYTHONDONTWRITEBYTECODE=1 python3 -m glval.compact --execute`，生成带原文件
SHA256 的执行摘要和 timing 摘要。最终 loss/accuracy/PPL 与官方计时均值原样保留。
详细记录取消 Git 跟踪，忽略规则不会删除原件；历史报告与冻结计划不改写。
新 clone 用 `glval artifacts` 查看已保存摘要，完整重新验证仍要求原始执行证据。
首次公开发布使用无旧历史的精简快照；本地完整历史和原始证据单独保留。
具体范围与操作见 [精简发布说明](docs/compact-publication.md)。

### 历史 campaign 产物迁移

仅 `campaigns/` 内的运行产物迁入新建的 `runs/campaign-<原目录名>/`；
已有 `runs/`、`diagnostics/` 和其他位置的产物保持原位、内容不变。
历史 `cases/` 等内部结构保留。每个迁移目录的 `migration.json` 记录文件哈希和旧路径；
`legacy-source/` 保存迁移时的来源脚本与配置。原始测量和冻结记录不批量改写路径。

`campaigns/` 保留历史来源及本地兼容链接，旧 CLI 已停用。迁移目录是历史归档，
不能把它当作新版本可直接继续执行的计划；新执行必须生成新 ID 和新计划。
`diagnostics/` 的诊断代码与产物保持独立，不迁移，也不作为正式实验入口。

```bash
# 只读查看原始保存的 PPL 和耗时；不重新生成报告或验证历史实验
python3 -m glval artifacts runs/campaign-powersgd-graph-timing-20261118
python3 -m glval artifacts runs/powersgd-llama130m-r32-20261109

# 新 clone：恢复目标仍在 Git 中的历史兼容路径
python3 -m glval restore-campaign-links

# 迁移工具默认只预览；--apply 才移动 campaigns 产物，检查冲突与活动锁
python3 -m glval migrate-campaigns
```

`artifacts` 展示已保存的数值及其原始状态，不赋予新的 verified 结论；总墙钟包含启动和评估，
不可与测量窗口内每步耗时混用。新 clone 只包含版本管理中的核心文件；兼容链接不会恢复被忽略的详细产物。
本次迁移和独立 Luna CPU 工作流验收见 [验收记录](runs/interface-validation-20261120/verification.json)。

本项目作为 CORAL 的 submodule 管理，旧路径 /home/wyr/greedylore-validation 保留兼容软链接。修改和提交 validation 时在本仓库操作，采用新版本时再在 CORAL 提交子模块引用。开发前检出工作分支。
