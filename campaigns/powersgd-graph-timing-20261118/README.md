# Optimized PowerSGD AutoResearch timing

User authorized2026-11-18: only `powersgd-r32-graph-v1`, no Dense or original
PowerSGD runs. Four serial cases on GPUs4-7: BF16 r1, FP32 r1, FP32 r2, BF16 r2.
All use350M AdamW, per-device batch1/global4, sequence256, LR0.0025,
scheduler20k/warmup2k, BF16/FP32, no activation checkpointing, bucket1024MiB,
compression start100,101 warmup updates,400 measured updates102-501, seed1243.
Settings are read from N033 Research's frozen AutoResearch timing configs.
PowerSGD rank32/EF14/warm-start/minratio2 and dense embedding/head are retained.
FP32 uses the optimized method's original-operation fallback.

W&B disabled; no evaluation or model/optimizer checkpoints. The60k run stays
interrupted. Fresh local runtime copies preserve all historical runs and the
shared scaffold code. This campaign uses the existing independent-case runner
pattern because the generic CLI's timing builder disables layout recording.

The original `BucketRecordingHookState`, `bucket_recording_hook`, measurement
loop and summary logic are preserved. Only registration is bridged to the frozen
method adapter and binding observer. The recording callback forwards the same
inner Future with no wait or device synchronization. Its outputs are checked
against the actual historical bucket counts/byte layouts on all four ranks.

Frozen runtime and campaign-code hashes are verified before each case. Each case
gets an idle four-GPU lease; foreign processes are never stopped. Started/failed
attempts are not implicitly retried. Independent review is bound to the manifest
hash. Results are in `cases/*/output`, with `status.json`, `controller.log`, and
`summary.json` at campaign root.

CPU-only preparation:

    PYTHONDONTWRITEBYTECODE=1 python3 -m campaigns.powersgd-graph-timing-20261118.prepare

Explicit execution in tmux:

    bash campaigns/powersgd-graph-timing-20261118/launch.sh

No same-session Dense baseline is measured. Historical Dense/N33/GreedyLoRE
numbers are cross-run references and may have different hardware topology.
