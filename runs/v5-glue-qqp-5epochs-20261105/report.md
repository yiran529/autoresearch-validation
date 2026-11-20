# v5-glue-qqp-5epochs-20261105

Recipe: `glue-qqp-roberta-adamw-5epochs-offline-v5`; split: `validation`

| Method | Seed | Status | Metrics | Total wall seconds |
|---|---:|---|---|---:|
| dense | 1243 | verified | {'eval_accuracy': 0.9168191936680683, 'eval_f1': 0.8890501798027118} | 8148.942733464297 |
| greedylore-engineer-streamfix-glue-r8-v1 | 1243 | verified | {'eval_accuracy': 0.9140984417511748, 'eval_f1': 0.8855268795939221} | 11983.073215292301 |
| greedylore-engineer-streamfix-glue-r16-v1 | 1243 | verified | {'eval_accuracy': 0.9154588177096216, 'eval_f1': 0.8875065824117957} | 12078.676254052669 |
| v5-n019-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.9167944595597328, 'eval_f1': 0.8889035667107001} | 5206.895129933022 |
| v5-n033-research-glue-v1 | 1243 | verified | {'eval_accuracy': 0.9176354192431363, 'eval_f1': 0.8898518126488489} | 5237.928832509089 |
| v5-n022-engineer-glue-v1 | 1243 | verified | {'eval_accuracy': 0.9163245115013604, 'eval_f1': 0.8884271626925233} | 5303.472233264707 |

Comparisons: `report.json` (matched seeds, verified jobs only).

- Total wall time includes process startup and evaluation; not pure training throughput.
- Generic hook evidence does not certify algorithmic equivalence or refresh coverage.
- No aggregate scientific pass threshold is imposed.
