# PowerSGD rank32, original-kernel CUDA graph execution

Parent: powersgd-paper-r32. The pinned source snapshot and source manifest are
unchanged. Rank32, BF16, EF14, warm-start, RNG use, per-parameter layout, P/Q
updates, reduction ordering, decompression, dense embedding/head exclusions and
the existing completion synchronization are retained. No QR replacement, tensor
batching, dtype conversion, or change to the training/optimizer rule is made.

The adapter wraps only `_orthogonalize`: copy to an identical-shape staging
buffer, replay a CUDA graph of the original ATen calls, then copy back. Cache
keys include device, dtype, shape, strides, epsilon and current CUDA stream.
A process-wide reentrant host lock serializes capture and each staging/replay
enqueue sequence, including same-stream use by multiple callback threads.
CPU, non-BF16, noncontiguous and nested capture paths call the original function.

First cache capture uses PyTorch graph setup, which adds device synchronization
and allocator cleanup; steady-state replay does not change the hook's existing
synchronization or collectives. Graphs retain their static allocations.

Validation and speed evidence: diagnostics/powersgd-speed-20261118. Component
checks, four-rank real-shape synthetic gradient/state checks, and real-model
paired timing are separate evidence levels. This is an implementation variant,
not a new search result or a claim of full training convergence equivalence.
Original methods, archived runs and the interrupted 60k run remain unchanged.
