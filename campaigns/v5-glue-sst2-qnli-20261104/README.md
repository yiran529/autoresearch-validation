# Frozen v5 GLUE validation

Historical reference: /home/wyr/greedy_lore/glue/scripts/run_table_iii_muon.bash (configuration-audit.json contains hash and differences).

Order: explicit 128-example/8-update real RoBERTa integration smoke, full SST-2, then full QNLI. Within each task: Dense, GreedyLoRE rank8, rank16, N19 Engineer, N33 approved Research, N22 Engineer. One seed1243, GPU4-7 serial with one lease. A failed stage stops advancement to the next dataset. No other process is terminated.

Full tasks use the unchanged local RoBERTa snapshot and historical Arrow datasets; FP32, max_length256 dynamic padding, per-device4/global16, GA1, 10epochs, cosine0, warmup10%, WD0, compression1000. Optimizer AdamW uses unchanged base LR2e-4, betas.9/.999 eps1e-8. This is an optimizer replacement at fixed LR, not a tuned AdamW recipe. Muon matrix/scalar groups no longer apply. GreedyLoRE uses EF14, gap200 and explicit historical ranks8/16 plus previously verified stream synchronization fix. Candidates keep original source SHAs and parameters.

No checkpoint flags are supplied; driver has no final save_pretrained call. Model and optimizer checkpoints are disabled. W&B offline files, train logs, tokenized Arrow caches, all_results, per-rank binding/runtime evidence are retained under this scaffold. Offline official evaluate GLUE metric module is copied into scaffold cache; shared model/data caches are read-only inputs.

Primary metric: final epoch GLUE validation accuracy; best epoch secondary. Historical Muon results are context only; fresh AdamW baselines provide the comparison. Synthetic/integration evidence does not establish quality, statistical significance or periodic refresh coverage.

Controller: controller.py; tmux v5-glue-20261104; status.json; controller.log. Full plans: ../../runs/v5-glue-sst2-20261104 and ../../runs/v5-glue-qnli-20261104.
