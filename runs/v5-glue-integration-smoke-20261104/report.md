# v5-glue-integration-smoke-20261104

Recipe: `glue-sst2-roberta-integration-smoke-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_accuracy': 0.4375} | 16.13464213628322 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_accuracy': 0.4375} | 20.543024071957916 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_accuracy': 0.4375} | 20.57155199861154 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.4375} | 14.462770141195506 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_accuracy': 0.4375} | 15.01026667887345 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.4375} | 14.624773440416902 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
