# v5-glue-mrpc-5epochs-20261105

Recipe: `glue-mrpc-roberta-adamw-5epochs-offline-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_accuracy': 0.9068627450980392, 'eval_f1': 0.9323843416370107} | 102.5543243419379 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_accuracy': 0.8995098039215687, 'eval_f1': 0.9284467713787086} | 133.22357787983492 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_accuracy': 0.8897058823529411, 'eval_f1': 0.9211908931698775} | 132.71270669763908 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.8921568627450981, 'eval_f1': 0.9225352112676056} | 75.61328238528222 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_accuracy': 0.8946078431372549, 'eval_f1': 0.9241622574955908} | 77.20610547577962 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.8848039215686274, 'eval_f1': 0.9176882661996497} | 75.00911983009428 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
