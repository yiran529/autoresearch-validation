# Historical campaign sources

此目录是历史入口归档，不用于创建或执行新正式实验。
旧 `.py` / `.sh` CLI 已停用；可读取配置和源码理解历史实验，但不得绕过根目录约定执行其函数。

新实验只用根目录 `AGENTS.md` 中的 `python3 -m glval` 接口，全部产物写入 `runs/`。
历史运行产物已迁入 `runs/campaign-<原目录名>/`。本地旧路径可能是兼容软链接，
不得把它们当作新实验输出入口，也不得修改冻结记录。
查看历史指标用 `python3 -m glval artifacts runs/campaign-<原目录名>`。

在新 clone 中需要旧路径时运行 `python3 -m glval restore-campaign-links`。
该命令仅恢复目标仍存在的兼容链接，不恢复未纳入 Git 的日志、trace 或运行副本。
