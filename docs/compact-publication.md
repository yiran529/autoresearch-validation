# Compact result publication

The public snapshot contains code, frozen method sources, configurations,
original final quality metrics, saved reports and small derived execution/timing
summaries. Original execution records stay on the experiment machine. Publication
does not regenerate measurements or give historical runs a new verified status.

Before committing newly completed experiments, export derived summaries:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m glval.compact --execute
```

By default the exporter reads only Git-tracked `runs/` records. For a new run,
select its path explicitly, including evidence that is ignored by Git:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m glval.compact --execute --run runs/RUN_ID
```

It creates `runtime-summary-rank-N.json` beside original `runtime-rank-N.json`
files, and `timing-summary.json` beside timing `all_results.json`. It never
overwrites original records. Final quality `all_results.json` files remain
unchanged and tracked. New raw evidence does not need to be staged to export a
selected run. Avoid forcing raw evidence into publication commits.

Runtime summaries keep binding identities, configuration, counters, memory and
elapsed time. Explicit refresh-step arrays become count/range/progression/hash
records. Unknown/null refresh information stays unknown; this transformation
never certifies algorithmic refresh coverage.

Timing summaries copy official scalar measurements and configuration exactly.
Per-rank step arrays (including short windows) and other long rank sequences
become count, endpoints, SHA256 and, for finite numeric sequences,
min/max/fmean/median/nearest-rank p95/p99. Derived statistics are not
replacements for the recorded official window mean. Raw arrays are not included
in the public summary. Each summary identifies its original filename and SHA256.

Raw runtime files are ignored repository-wide. For timing, the exporter creates
run-local `.gitignore` entries covering the exact original timing result paths.
Initially tracked raw records must also be removed from the index with
`git rm --cached`; ignoring alone does not untrack them. The generated summaries
and these run-local ignore files are versioned. Do not delete local originals.

Use the read-only saved-result interface in a fresh clone:

```bash
python3 -m glval artifacts runs/campaign-powersgd-graph-timing-20261118
```

`artifacts` reads published timing summaries without raw observations. Full
execution revalidation still requires original evidence and runtime files;
`report` does not silently substitute compact evidence for those requirements.

The first compact GitHub publication is a parentless snapshot, so previous
large records are not reachable through its history. The complete original local
history is preserved on the original branch and in a local Git bundle. Raw-file
archives and hash manifests are retained locally under ignored diagnostics
subdirectories. Future publications should descend from the public snapshot;
pushing the original full-history branch would publish its original large data.
