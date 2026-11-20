# v5 final validation, one seed

Authorized: only GPU4-7, serial N19 -> N27 -> N33 Research -> N22; seed1243. No Dense/GreedyLoRE training. 130M Muon20k, original online recipe, no model/optimizer saves, activation checkpointing on.

Before full queue: four-rank adapter-vs-direct differential for FP32/BF16 at1010steps including1000dense prefix; gradient/parameters/AdamW/EF/predictor exact in covered steps. Then actual130M formal-batch8step smoke with explicitly shortened compression prefix2. Smoke is not a full recipe or quality result. Full queue starts only if every gate passes.

Sources are immutable v5 snapshots; old scaffold methods with similar names are not reused. Data repetition and order follow existing recipe and common vendor driver; eval fixed across methods. Runtime copied from scaffold baseline, not search checkpoint replay. Single seed cannot establish statistical superiority; historical Dense/GreedyLoRE are context only.

status.json reports stage. Full plan: /home/wyr/greedylore-validation/runs/v5-final-four-130m-20261031. controller.log has supervisor output. No implicit reruns or resume. Do not signal other users' processes.

## Offline transition
User authorized W&B offline after online initialization timeouts. Active run: /home/wyr/greedylore-validation/runs/v5-final-four-130m-20261031-offline; tmux v5-final-four-130m-offline; status.json/offline-controller.log. Original online attempts retained. Training budget, seeds, method order and GPU allocation unchanged. Checks reused for identical candidate/adapter hashes.
