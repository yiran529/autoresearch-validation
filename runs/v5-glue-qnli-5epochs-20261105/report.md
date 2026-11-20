# v5-glue-qnli-5epochs-20261105

Recipe: `glue-qnli-roberta-adamw-5epochs-offline-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_accuracy': 0.9289767526999817} | 2330.677149381954 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_accuracy': 0.9194581731649277} | 3200.4988449881785 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_accuracy': 0.9264140582097748} | 3195.049381478224 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.9269632070291048} | 1526.187179933302 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_accuracy': 0.9264140582097748} | 1528.754259529058 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.9236683141131247} | 1550.5726752062328 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
