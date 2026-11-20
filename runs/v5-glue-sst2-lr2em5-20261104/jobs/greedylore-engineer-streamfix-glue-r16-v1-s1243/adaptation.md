# GreedyLoRE Engineer stream dependency fix

Parent: `382bfae897a4db84af45e021ed1952bab7584445`
Fix: `8316c7eacc991173dee03f430ebec20efd99760b`

Fix EF14 producer stream -> callback stream ordering using a CUDA event recorded after local R/EF production. The dense All-Reduce still overlaps local projection/reconstruction. Parameters, projection selection, EF update, optimizer and training loop are unchanged. SVD exception logging no longer reads undefined outputs. Original method and failed results retained.

Synthetic four-rank delayed-producer test passed. Real-model/full-schedule validation recorded separately under diagnostics. No claim of full pretraining convergence from a short diagnostic.

2026-10-25: Real 60M AdamW and130M Muon formal-transition diagnostics both completed1402 steps on4ranks, batch128, BF16, activation checkpointing, compression start1000. All loss/gradient checks finite. These skipped validation and are not full pretraining results. New full-budget runs preserve the standard driver and evaluation.

## GLUE 2026-11-04 variant
Same frozen implementation and parameter source as greedylore-engineer-streamfix. Adds GLUE workload declaration only; comm_hooks and adapter are byte-identical. AdamW uses normal gradients (precond_adam momentum mode inactive), no training-loop or optimizer source changes. Classification head and embeddings follow existing hook semantics; no new masking. Compression begins at recipe step100. Static only until campaign smoke records pass; prior Llama results are not GLUE evidence.

Historical GLUE rank16 arm; rank is explicitly overridden to match run_table_iii_muon.bash, update gap200 and EF14 unchanged.
