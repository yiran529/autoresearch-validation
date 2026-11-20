# GLUE rerun, AdamW LR2e-5

User-provided Table VI reference: SST-2 and QNLI both LR2e-5 at ranks8 and16. Only LR changes versus previous full recipes; shared global batch16 retained for fairness, rather than table-specific QNLI rank16 batch32. Other settings match historical GreedyLoRE GLUE: local RoBERTa-base, FP32, 10epochs, sequence256, per-device4, four ranks, GA1, cosine0, warmup10%, WD0, seed1243, compression1000. No model or optimizer checkpoint; W&B offline.

Order SST-2 then QNLI, each Dense / GreedyLoRE r8 / r16 / N19 / N33 / N22. Fixed GPU4-7 serial. Original2e-4 logs preserved under previous campaign. Six real-model integration smokes passed before this LR-only change; no new smoke required. One seed results are descriptive, not statistical significance.

Controller.py and status.json in this directory, tmux v5-glue-lr2em5-20261104.
