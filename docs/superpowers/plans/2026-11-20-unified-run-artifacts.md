# Unified formal experiment artifacts implementation plan

> For agentic workers: execute the tasks inline with the executing-plans workflow.
> The final independent simulation uses the user-requested Luna subagent.

**Goal:** Use the existing glval flow for all new formal experiments and move only
campaigns artifacts into fresh runs directories.

**Architecture:** Formal plan/run/report paths share one validated runs boundary.
Historical campaign payloads are moved byte-for-byte with compatibility aliases,
per-file manifests, and original source snapshots. Diagnostics remains independent.

**Tech Stack:** Python 3.10+ standard library, unittest, Git ignore rules.

**Spec:** The user's approved conversation scope: move campaigns artifacts only;
do not move or rewrite existing runs, diagnostics, or other artifact locations;
use the existing default-runs interface and root AGENTS instructions; finish with
a Luna agent simulation. No GPU execution or new training is authorized.

## Global constraints

- Preserve frozen measurements, plans, source identities, and original result bytes.
- Refuse collisions, path traversal, symlink escapes, and active migration locks.
- Leave existing runs and diagnostics artifact trees untouched.
- Preserve the user's initially untracked diagnostic directory and idle-policy test.
- New formal execution requires the normal explicit authorization and --execute.
- Keep caches and shared resource locks at their existing locations.

## Task 1: Validate the existing run boundary

Files: glval/artifacts.py, glval/plans.py, glval/runner.py,
glval/results.py, glval/__main__.py, tests/test_scaffold.py,
tests/test_saved_artifacts.py.

- [x] Add tests that reject campaigns output, runs itself, non-run symlink targets,
  and duplicate destinations; observe failures against the previous implementation.
- [x] Add run_directory(path) and new_run_directory(run_id, output=None).
- [x] Route formal planning, execution, status, and report through the boundary.
- [x] Preserve the documented project-root alias and reject dangling destination links.
- [x] Test historical artifact reads without report rewrites or GPU imports.

Check:
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_scaffold.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_saved_artifacts.py -v
```

## Task 2: Migrate campaign payloads with recovery

Files: glval/migration.py, glval/__main__.py,
tests/test_artifact_migration.py, runs/campaign-*/migration.json.

- [x] Snapshot original runs, diagnostics, and campaigns before migration.
- [x] Implement dry-run selection of generated campaign files and directories.
- [x] Move to fresh runs/campaign-<name>, snapshot the original source, and retain
  relative aliases; write per-file hashes without rewriting frozen records.
- [x] Cover duplicate targets, controller locks, rollback, and alias restoration.
- [x] Execute the migration of 22 campaigns and 5030 files/symlinks.
- [x] Verify every original artifact against its baseline hash after migration.

Check:
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_artifact_migration.py -v
python3 -B -m glval migrate-campaigns
```

## Task 3: Retire the competing active entrypoints

Files: campaigns/*/*.py and tracked launch.sh entrypoints,
glval/build.py, glval/plans.py, glval/checkpoints.py,
tests/test_scaffold.py, tests/test_rolling_checkpoint.py.

- [x] Keep source definitions importable for CPU tests, but stop legacy CLI
  execution before imports or GPU actions. Original source snapshots remain in runs.
- [x] Port passive bucket recording into the standard timing bridge.
- [x] Preserve the PowerSGD dense-exclusion hook hash from the completed 130M run.
- [x] Build explicit ResNet50 recipes through the common formal planner.
- [x] Keep new checkpoints within job output; preserve inherited optimizer-hook slots.

Check:
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_rolling_checkpoint.py -v
```

## Task 4: Retain core results and document the supported workflow

Files: .gitignore, AGENTS.md, README.md, campaigns/AGENTS.md, runs/AGENTS.md,
tests/test_artifact_gitignore.py.

- [x] Permit core result JSON, provenance, reports, and migration source snapshots.
- [x] Keep logs, JSONL, traces, checkpoints, and runtime copies excluded.
- [x] Require root-level glval use; define the independent diagnostics exception.
- [x] Document artifacts (read-only), migration dry-run, and clone alias restoration.
- [x] Stage only source changes and newly migrated core files, not pre-existing runs
  or unrelated untracked diagnostics/test files.

## Task 5: Verify and independently simulate

- [x] Run the full standard-library CPU suite (60 tests) and git diff --check.
- [x] Compare original runs and diagnostics hashes; check all moved campaign payloads.
- [x] Commit the verified implementation and migration records.
- [x] Prepare an isolated source checkout and synthetic CPU-only input files.
- [x] Ask a fresh gpt-6-luna agent to prepare an experiment and consume a saved
  historical result using the normal documentation, with no training.
- [x] Inspect its actual commands, paths, exit outcomes, and final report artifacts.
- [x] Address material findings, rerun affected checks, and retain a concise
  verification record without claiming GPU or scientific validation.

Final check:
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
git diff --check
```

## Observed acceptance

Implementation commit: `00efbcc`. The full CPU suite passed 60 tests. Hash checks
confirmed 27,902 original runs files/symlinks and 14,515 diagnostics files/symlinks
unchanged, plus all 5,030 moved artifacts across 22 campaigns. A fresh Luna agent
executed 24 documented commands/checks in an isolated clone: two plans/reports,
read-only historical metrics, duplicate/output rejection, legacy CLI refusal, and
a JSON/text migration fixture. Both new plans remained planned/unverified with
no training outputs. The parent independently checked the two plans, unchanged
implementation, and all 53 tracked history files in the two consumed archives.

Evidence: `runs/interface-validation-20261120/verification.json`. The simulation
did not exercise output symlink escapes (covered by CPU tests), dataset contents,
actual bucket layouts, GPU performance, or convergence.
