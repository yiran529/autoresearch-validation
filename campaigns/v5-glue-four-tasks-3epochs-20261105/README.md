# Four GLUE tasks, fresh three epochs
All main experiments start from the same RoBERTa-base pretrained weights. No continuation from previous training. Order SST-2, QNLI, QQP, then MNLI (one explicit small Dense evaluation integration test immediately before MNLI).
Each full task: six frozen methods Dense / GreedyLoRE r8 / r16 / N19 / N33 / N22, seed1243. GPU4-7 serial. Only requested driver change is num_train_epochs10->3; budget and cosine/warmup10% recomputed accordingly. Shared AdamW2e-5, FP32, per-device4 global16 GA1, sequence256, WD0, compression1000. No model/optimizer checkpoint; W&B offline.
Old attempts/logs retained. MNLI evaluation bridge preserves training-loop AST exactly, reports both splits and verifies mismatched sample count.
Controller.py, status.json and controller.log in this directory. tmux v5-glue-3epochs-20261105.
