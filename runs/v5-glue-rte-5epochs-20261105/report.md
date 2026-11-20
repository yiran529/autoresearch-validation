# v5-glue-rte-5epochs-20261105

Recipe: `glue-rte-roberta-adamw-5epochs-offline-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_accuracy': 0.7328519855595668} | 75.14506686665118 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_accuracy': 0.7509025270758123} | 93.18756085215136 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_accuracy': 0.7472924187725631} | 92.52667045826092 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.7545126353790613} | 58.130695863161236 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_accuracy': 0.7978339350180506} | 58.95365637075156 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.7906137184115524} | 58.33159797405824 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
