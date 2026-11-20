# v5-glue-cola-5epochs-20261105

Recipe: `glue-cola-roberta-adamw-5epochs-offline-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_matthews_correlation': 0.6107029287090541} | 207.30178410187364 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_matthews_correlation': 0.582897151722857} | 298.55285101011395 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_matthews_correlation': 0.608358147712731} | 300.1007132669911 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_matthews_correlation': 0.6206942584175241} | 146.14064967539161 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_matthews_correlation': 0.6233293453039225} | 147.44388981722295 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_matthews_correlation': 0.6306644810568987} | 157.88385685300454 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
