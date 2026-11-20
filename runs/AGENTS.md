# Experiment artifacts

此目录保存正式实验产物；创建和执行入口见根目录 `AGENTS.md`，统一使用 `python3 -m glval`。
不得编辑已有 run 的冻结计划、源码副本或原始测量来适配新代码；创建新的实验 ID。

`campaign-*` 是从 `campaigns/` 迁入的历史归档：`migration.json` 记录旧路径、文件哈希
及兼容链接；`legacy-source/` 是迁移时的原始来源快照，不是可运行的新实验入口。
查看指标用 `glval artifacts`；不要执行归档脚本，也不要重写历史报告。

核心 JSON/报告由 Git 白名单保留；详细日志、trace、checkpoint、运行副本和逐步 JSONL
留在本地。新 clone 不保证包含这些被忽略的产物。
