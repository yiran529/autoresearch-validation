# v5-60m-three-idle10m-20261107

Recipe: `llama60m-adamw-full-online-v2`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| greedylore-engineer-streamfix | 1243 | verified | {'final_eval_loss': 3.637438061830251, 'final_eval_ppl': 37.99437269665306} | 4505.8798022898845 |
| v5-n033-research | 1243 | verified | {'final_eval_loss': 3.5126557310106663, 'final_eval_ppl': 33.53721544420335} | 4226.517599641811 |
| dense | 1243 | verified | {'final_eval_loss': 3.496805198072765, 'final_eval_ppl': 33.00982346984699} | 4251.494875824079 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
