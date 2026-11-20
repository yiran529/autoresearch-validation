# v5-glue-mnli-5epochs-20261105

Recipe: `glue-mnli-roberta-adamw-5epochs-offline-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_matched_accuracy': 0.8773306164034641, 'eval_mismatched_accuracy': 0.871847030105777} | 8724.6437701541 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_matched_accuracy': 0.8765155374426897, 'eval_mismatched_accuracy': 0.8725589910496339} | 12402.180540627334 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_matched_accuracy': 0.8732552215995925, 'eval_mismatched_accuracy': 0.874593165174939} | 12368.694895042107 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_matched_accuracy': 0.8762098828323994, 'eval_mismatched_accuracy': 0.8740846216436127} | 5436.5674873953685 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_matched_accuracy': 0.8770249617931737, 'eval_mismatched_accuracy': 0.8707282343368592} | 5533.762274033856 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_matched_accuracy': 0.8770249617931737, 'eval_mismatched_accuracy': 0.8703213995117982} | 5812.884938117117 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
