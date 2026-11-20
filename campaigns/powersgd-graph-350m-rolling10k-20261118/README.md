# Optimized PowerSGD350M fresh run, rolling10k

Authorized 2026-11-18: after the four optimized-only timing cases are complete
and verified, restart350M from update0, using `powersgd-r32-graph-v1` only.
60k Muon BF16 updates, LR.005/scalar.001, warmup6k, cosine to10%, WD0,
clip1, activation checkpointing, seq256, batch128/device/global512, GA1,
seed1243, existing C4 repeated by the baseline rule. Compression starts1000;
rank32/EF14/warm start/minratio2 and dense embedding/head, evaluation every1000.
W&B disabled. The previous interrupted attempt remains intact.

Save every10000 updates (10k/20k/30k/40k/50k/60k). Publish the complete new
generation before retiring the previous one; keep one committed generation.
Location follows the existing N33 rolling-checkpoint storage convention:
`/dev/shm/wyr_tmp/greedylore-validation/powersgd-graph-350m-rolling10k-20261118/full/powersgd-r32-graph-v1/seed-1243/latest`.
This is RAM-backed storage and does not survive a host reboot. Checkpoints contain
the rank0 model and config plus each rank's optimizer, scheduler, PowerSGD EF/P/Q,
hook RNG, training RNG and counters. Data iterator/DDP reducer layout are not
serialized; exact stream resume is not implemented.

The global scaffold and old snapshots remain unchanged. The new runtime locally
collects inherited `__slots__` as well as `__dict__`, avoiding loss of PowerSGD
state. CUDA graph caches are transient and are not serialized.

Before full training, an independent eight-update smoke at the formal batch
compresses from2 and saves at4/8. The controller deserializes every rank file
and the model, verifies step8/P/Q/EF/optimizer/scheduler/RNG and retirement of
step4, records sizes, then retires its temporary smoke payloads. It checks space
for two generations before fresh full training. CPU tests never import torch.

CPU preparation: `PYTHONDONTWRITEBYTECODE=1 python3 -m campaigns.powersgd-graph-350m-rolling10k-20261118.prepare`.
Launch only after independent review bound to both plans and campaign hashes:
`bash campaigns/powersgd-graph-350m-rolling10k-20261118/launch.sh`.
GPU4-7 only; wait for idle without stopping foreign jobs; no implicit retry/resume.
Status in `status.json`, logs in `controller.log`; metrics/checkpoint pointers in
the new `runs/powersgd-graph-350m-rolling10k-{smoke,full}-20261118/jobs/*/output`.

2026-11-18 recovery: the eight-update smoke completed and published both saves,
but the original post-training validator incorrectly required nonempty P/Q for
the two dense-only embedding/head buckets. The corrected validator checks the
complete23-bucket layout, permits empty P/Q only for these two known buckets,
and checks exact EF/P/Q element totals. Failed validation and original review/
provenance are preserved under `recovery-20261118/`. After updated review,
`bash campaigns/powersgd-graph-350m-rolling10k-20261118/launch.sh --continue-after-verified-smoke`
revalidates the already completed smoke and its actual payloads, then starts
the still-planned full run fresh. It never reruns a started training job.
