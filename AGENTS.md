# Project instructions

本项目是独立评测 scaffold。其他仓库与搜索 run 是只读输入，所有新代码、快照、日志、缓存和临时文件留在本项目。
不要修改 CORAL、greedy_lore、候选 run、现有虚拟环境或依赖。

优先阅读 README.md。候选适配读取 skills/adapt-greedylore-candidate/SKILL.md。
使用 JSON 配置，核心 CLI 保持 Python 3.10+ 标准库依赖。

## 正式实验的唯一入口与产物位置

新建正式训练、算法对比和正式 timing，必须使用 `python3 -m glval plan EXPERIMENT.json`，
随后通过 `glval preflight/status/run/probe/report` 操作生成的 run。默认位置是
`runs/<experiment-id>/`；`plan --out` 也只能指定本项目 `runs/` 下尚不存在的目录。
不得另建 controller、调用底层 `build_runtime` 后自行执行训练，或绕过接口向其他位置写产物。
接口缺少实验所需能力时，先扩展统一接口和 CPU 测试，再生成新计划；不得静默丢弃特殊设置。

正式实验的冻结计划、运行源码副本、PPL/耗时、报告、状态、日志、trace 和 checkpoint
均写入对应 run。新 rolling checkpoint 的 `root` 必须是 job output 内的相对路径
（默认 `checkpoints`），不能沿用历史 `/dev/shm` 路径。共享缓存和资源锁仍在 `cache/`、`locks/`。

`campaigns/` 只保留历史配置、来源脚本和指向迁移产物的兼容链接；旧 CLI 已停用，
不得运行其中的历史脚本或 `legacy-source/` 快照来启动新实验。历史指标通过
`python3 -m glval artifacts runs/RUN` 只读查看，不为适配新代码重写旧报告或冻结计划。
源码变化后需要新的计划；迁移目录是归档，不代表可直接续训。

`diagnostics/` 保持独立：专项诊断代码和产物可留在其中，不在本次迁移范围内。
正式训练和正式性能对比不得以诊断名义绕过统一接口。

迁移仅处理 `campaigns/` 内产物。不得移动、改写已有 `runs/`、`diagnostics/` 或其他位置的产物。
Git 跟踪核心结果和来源记录，运行副本、日志、trace、模型和逐步 JSONL 不纳入 Git；忽略不代表删除。

`check` / `plan` / `preflight` / `status` / `report` 不得导入候选 Python 或启动 GPU。
`probe` / `run` 是执行入口，必须有明确的当前用户授权和 `--execute`；用户说“不训练”时不得运行 probe，因为它也执行参数更新。
生成实验配置和脚本不代表获得执行授权。

保留 vendor 原始文件与 provenance；适配只修改每次运行副本。方法不能静默覆盖公共模型、数据、优化器和训练循环。
新方法参数可以扩展，方法语义变化必须显式说明；算法依赖超出 v1 接口就记录边界，不通过忽略源码差异绕过。
静态检查、合成 probe、真实 workload smoke、完整实验分别记录，不混称验证通过。

改动后运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`。
测试不导入 torch 或执行任何训练。先在测试里检查需要保护的行为，再检查示例计划与实际 CLI。
