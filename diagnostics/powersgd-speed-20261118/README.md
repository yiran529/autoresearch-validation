# PowerSGD speed optimization, 2026-11-18

The user requested pausing the350M run and improving implementation speed while
preserving semantics. The60k run stopped after logged update1582; no model or
optimizer checkpoint exists, so it cannot resume from that point. Its frozen
runtime and logs are retained. This diagnostic does not restart formal training.

The candidate is `methods/powersgd-r32-graph-v1`. Its source snapshot is identical
to `powersgd-paper-r32`; only the adapter adds execution of the original BF16
Gram-Schmidt ATen operations through CUDA graphs. Rank, dtype, EF14, warm start,
RNG use, P/Q updates, collectives and dense embedding/head scope are unchanged.
There is no tensor batching or QR replacement. A process-wide host lock protects
capture and staging enqueues; caches are isolated per CUDA stream. Initial graph
capture adds synchronization, while steady replay retains the existing hook's
synchronization and math. CPU/non-BF16/noncontiguous/nested-capture paths use the
original implementation. No environment or dependency was changed.

Evidence levels:

- `profile.json`: real367,969,280-parameter shapes, one BF16 bucket, one GPU.
 336 orthogonalization calls consume2.870s of a2.930s hook invocation.
- `orthogonalize-check.json`: exact BF16 bitpattern comparison, including zero
 input/NaN bits, epsilon0/1e-8, rank2/32, actual1024/2736 row shapes, repeated
 updates, two host threads with shared/different CUDA streams. Also records
 three synchronized component timing repetitions.
- `distributed-hook-check.json` and `hook-rank-*.json`: four GPUs, actual350M
 parameter shapes with synthetic per-rank gradients. Eight complete hook steps
 cover the dense prefix, random-Q initialization, warm-start and EF. Compare
 every BF16 bit of gradients and packed P/Q/residual states. Timings are hook
 costs, not training speed.
- `paired-timing-recipe.json`: four GPUs, real local C4/350M model, Muon,
 BF16, activation checkpointing, batch128/device/global512, GA1, sequence256,
 same60k/6k LR horizon. A separate bounded diagnostic sets compression2,
 warms8 updates and measures20 updates per method. No evaluation, model or
 optimizer checkpoints, or W&B. Both runtimes use the exact verified130M
 scope hook. This checks real DDP bucket rebuilding/Future execution and
 measures data/H2D/forward/backward/communication/clipping/optimizer/scheduler.

All GPU commands require explicit execution and acquire scaffold GPU leases;
they do not stop or take over foreign processes. From the scaffold root:

    PYTHONDONTWRITEBYTECODE=1 /home/wyr/greedy_lore/.venv/bin/python diagnostics/powersgd-speed-20261118/profile_hook.py --execute
    PYTHONDONTWRITEBYTECODE=1 /home/wyr/greedy_lore/.venv/bin/python diagnostics/powersgd-speed-20261118/check_orthogonalize.py --execute
    PYTHONDONTWRITEBYTECODE=1 /home/wyr/greedy_lore/.venv/bin/python diagnostics/powersgd-speed-20261118/check_distributed_hook.py --execute
    PYTHONDONTWRITEBYTECODE=1 python3 -m diagnostics.powersgd-speed-20261118.prepare_timing
    PYTHONDONTWRITEBYTECODE=1 python3 -m glval run runs/powersgd-350m-muon-graph-paired-20261118 --gpus 4,5,6,7 --execute

Preparation and execution refuse to overwrite/retry existing frozen attempts;
an independently authorized repeat requires a fresh experiment identity.
## Final measured result

All checks below passed on the final adapter, SHA256
`387e057f3ffa41717a9ad54a263848024eb372f65524bcb9cf483376dc107b62`.
The independent Sol6.1 high code review and frozen-runtime review found no
blocking implementation issue; final result review is recorded in `review.json`.

| Measurement | Original | CUDA graph | Speedup |
|---|---:|---:|---:|
| Real model, four GPUs, formal batch128,20 measured updates |3.957877s/step|1.643341s/step|2.4084x|
| Four-rank real-shape synthetic whole hook,5 timed compressed updates |3.104315s|1.246005s|2.4914x|

Both real-model jobs completed and were verified. Each rank executed622 hook
calls across28 backward cycles, demonstrating the multi-bucket/Future path.
The20 measured losses on each of four ranks were exactly equal between methods
(maximum difference0). The separate eight-step hook fixture compared every
BF16 bit of gradients, P/Q and EF residuals on all ranks, including the dense
prefix, random-Q initialization and repeated warm-start/EF updates. Component
fixtures also compare all bit patterns for zero-input/NaN results and exercise
shared-stream and separate-stream calls from two host threads.

The real timing excludes initial graph creation and startup. Total short-job
wall time, including startup/warmup, was166.09s versus144.59s (1.15x). The20-step
steady-state window, not that startup-dominated ratio, measures training-step
speed. Graph setup adds initial device synchronization/cache allocations.
Machine-readable results and scope limitations are in `summary.json`.

These bounded checks cannot establish60k convergence or a multi-seed benefit.
Historical baselines were not rerun as full training. Results are local artifacts;
the original interrupted run remains interrupted.
