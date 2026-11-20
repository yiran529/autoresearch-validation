# v5-glue-sst2-5epochs-20261105

Recipe: `glue-sst2-roberta-adamw-5epochs-offline-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_accuracy': 0.944954128440367} | 1508.312013361603 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_accuracy': 0.9438073394495413} | 2142.0363001069054 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_accuracy': 0.9472477064220184} | 2141.632547797635 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.9461009174311926} | 960.2322639869526 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_accuracy': 0.944954128440367} | 985.0401533110999 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.9472477064220184} | 1020.2946174908429 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
