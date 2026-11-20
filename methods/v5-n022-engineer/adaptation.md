# v5-n022-engineer

Source `8cec48b5e58e89139ba368bfaae98817a475502f`, N022, v5 run only.

Frozen hook is used without algorithm changes. Parameters reproduce source method_policy.py; start_compress_iter comes exclusively from recipe (1000 full training). FP32 residuals, native packet format, rank-order mean and predictor state (N022 only) are retained. Entire comm_hooks directory is overlaid to preserve imports. Source method_policy and training/checkpoint modules are not executed; no candidate optimizer changes exist. Training entry changes reviewed separately in method.json. Adapter only merges defaults, recipe and declared compression parameters and invokes original registration.

Supported: LLaMA DDP only; no claim for other workloads. Ready for execution checks; static evidence only until campaign checks finish. Four-rank adapter-vs-direct update/state comparison, 1010-step synthetic prefix check and actual130M short smoke are mandatory campaign gates. Full quality is inherited/provisional until 20,000-step runs complete.
