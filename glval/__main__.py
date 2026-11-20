import argparse
import json
import sys
from pathlib import Path

from .common import ROOT, load_ref, read_json
from .artifacts import run_directory
from .methods import check_method, create_method
from .plans import make_plan, preflight, verify_plan
from .results import report, saved_results


def main():
    parser = argparse.ArgumentParser(prog="python3 -m glval")
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import-method", help="Export a fixed Git candidate; never execute it")
    imp.add_argument("--id", required=True)
    imp.add_argument("--repo", required=True)
    imp.add_argument("--commit", required=True)
    check = commands.add_parser("check", help="Static adapter/source checks only; no imports, GPUs or training")
    check.add_argument("method")
    check.add_argument("--workload", choices=["llama", "resnet", "glue"])
    plan = commands.add_parser("plan", help="Freeze runtimes and generate commands; never launch training")
    plan.add_argument("experiment")
    plan.add_argument("--out", help="Fresh directory below this project's runs/ (default: runs/<id>)")
    migration = commands.add_parser("migrate-campaigns", help="Archive only campaign artifacts; dry-run by default")
    migration.add_argument("--apply", action="store_true", help="Move artifacts after collision and active-lock checks")
    commands.add_parser("restore-campaign-links", help="Recreate local historical aliases after cloning")
    commands.add_parser("artifacts", help="Read saved PPL/time from a run or historical archive; no writes").add_argument("run")
    for name in ("preflight", "status", "report"):
        commands.add_parser(name).add_argument("run")
    run = commands.add_parser("run", help="Explicit long-training entrypoint; use tmux")
    run.add_argument("run")
    run.add_argument("--gpus", required=True)
    run.add_argument("--execute", action="store_true")
    probe = commands.add_parser("probe", help="Explicit synthetic multi-GPU probe; does not run automatically")
    probe.add_argument("run")
    probe.add_argument("--job", required=True)
    probe.add_argument("--gpus", required=True)
    probe.add_argument("--steps", type=int, default=8)
    probe.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "import-method":
            result = {"method_dir": str(create_method(args.id, args.repo, args.commit)), "status": "draft"}
        elif args.command == "check":
            method, source, adapter = check_method(load_ref(args.method, "methods"), args.workload)
            result = {"method": method["id"], "status": "static_pass", "training_executed": False,
                      "note": "Adapter syntax and provenance checked; no runtime or semantic equivalence certification"}
        elif args.command == "plan":
            result = {"run": str(make_plan(args.experiment, args.out)), "status": "planned", "training_executed": False}
        elif args.command == "migrate-campaigns":
            from .migration import migrate_campaigns
            result = migrate_campaigns(args.apply)
        elif args.command == "restore-campaign-links":
            from .migration import restore_campaign_links
            result = restore_campaign_links()
        elif args.command == "artifacts":
            result = saved_results(args.run)
        elif args.command == "preflight":
            result = preflight(run_directory(args.run))
        elif args.command == "status":
            result = read_json(run_directory(args.run) / "state.json")
        elif args.command == "report":
            run = run_directory(args.run)
            result = report(run, verify_plan(run))
        elif args.command == "probe":
            from .runner import probe_plan
            result = probe_plan(args.run, args.job, args.gpus, args.steps, args.execute)
        else:
            from .runner import run_plan
            result = run_plan(args.run, args.gpus, args.execute)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 2 if result.get("status") in {"missing_inputs", "probe_failed", "partial_or_failed", "failed"} else 0
    except (ValueError, OSError, KeyError, SyntaxError) as error:
        print(f"glval: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
