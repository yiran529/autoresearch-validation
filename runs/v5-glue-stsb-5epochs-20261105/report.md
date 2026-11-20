# v5-glue-stsb-5epochs-20261105

Recipe: `glue-stsb-roberta-adamw-5epochs-offline-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_pearson': 0.9057211875915527, 'eval_spearmanr': 0.9032473129291837} | 151.7803909550421 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_pearson': 0.9036684036254883, 'eval_spearmanr': 0.90081161374155} | 204.11212684214115 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_pearson': 0.9052456617355347, 'eval_spearmanr': 0.9017589649282928} | 207.6693374691531 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_pearson': 0.9055466055870056, 'eval_spearmanr': 0.9028427183442783} | 105.66529823513702 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_pearson': 0.9059064984321594, 'eval_spearmanr': 0.9025098373319315} | 107.268493025098 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_pearson': 0.905896008014679, 'eval_spearmanr': 0.9030657950104832} | 106.87192545784637 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
