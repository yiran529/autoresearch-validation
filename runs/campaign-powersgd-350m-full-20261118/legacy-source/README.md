# PowerSGD 350M full training

Authorized 2026-11-18: one fresh PowerSGD rank32 run, seed1243, matching the
350M Muon training protocol used by N33 and historical Dense/GreedyLoRE.
W&B is disabled. No model or optimizer checkpoints are saved.

60k updates, warmup6000, matrixLR0.005, scalarLR0.001, cosine to10%, WD0,
clip1, BF16, activation checkpointing, seq256, batch128/device/global512,
GA1, four RTX4090 GPUs4-7. Existing C4 data/tokenizer only; repeat data
with the baseline rule. Compression starts at1000; evaluate every1000.
PowerSGD rank32/EF14/warm start/minratio2, with embedding/head dense.
The hook, adapter and driver match completed 130M PowerSGD validation.
N33 instead compresses full buckets including embedding/head; this is an
explicit method-scope difference, not a claim of identical communication.

An independent GPT-6.1 Sol high settings review must match both frozen
plan hashes before execution. First run a separate real350M eight-update
smoke at formal batch, compression start2; require verified completion
before launching fresh full training. Smoke is not convergence evidence.
The controller waits for idle GPUs4-7 and scaffold locks, polling every30s.
It does not stop foreign processes or retry/resume a started attempt.

Prepare (CPU only):

    PYTHONDONTWRITEBYTECODE=1 python3 -m campaigns.powersgd-350m-full-20261118.prepare

Execute in tmux after review:

    PYTHONDONTWRITEBYTECODE=1 python3 campaigns/powersgd-350m-full-20261118/controller.py --execute

Status: `status.json` and `controller.log`; complete per-rank evidence,
metrics and logs remain in the two `runs/powersgd-llama350m-r32-*20261118`
directories. Historical baselines are not rerun. Single-seed quality
comparisons are exploratory; wall time includes evaluation and startup,
and historical hardware/runtime conditions may differ.
