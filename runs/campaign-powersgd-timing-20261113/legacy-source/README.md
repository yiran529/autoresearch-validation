# PowerSGD v5 timing
Four serial tasks: 350M AdamW, BF16/FP32 x two repeats, seed1243.
Per-device batch1, seq256, no activation checkpointing, bucket1024MiB, NCCL SHM.
Dense prefix100, warmup101, measure updates102–501. No evaluation/checkpoints/W&B.
PowerSGD rank32, EF14, warm_start, min ratio2; embedding/head dense.
Same patched hook hash as completed PowerSGD 130M full validation.
Only new runtime copies are bridged; existing frozen methods/runs and CORAL are untouched.
CPU plan: PYTHONDONTWRITEBYTECODE=1 python3 -m campaigns.powersgd-timing-20261113.prepare
Launch: PYTHONDONTWRITEBYTECODE=1 python3 campaigns/powersgd-timing-20261113/controller.py --execute
Each task waits for any four physical GPUs individually idle for 600 consecutive seconds,
sampled every10s, no compute processes, memory<=512MiB and utilization<=5%.
Rechecks and takes scaffold locks immediately before launching; never stops foreign processes.
GPU/NUMA topology recorded. Speedups use historical Dense, not same-batch controls.
The scaffold disables the passive bucket layout observer used in v5; asynchronous timing
loop, workload hyperparameters and window are preserved. W&B remains disabled.
Runtime status/queue/provenance/summary are local artifacts.
