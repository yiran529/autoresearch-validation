"""Archive campaign artifacts without rewriting measurements or frozen records."""
from contextlib import ExitStack
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess

from .artifacts import new_run_directory, run_directory
from .common import ROOT, read_json, write_json


GENERATED = {
    "manifest.json", "cases.json", "queue.json", "provenance.json", "hardware.json",
    "state.json", "status.json", "summary.json", "report.json", "report.md",
    "review.json", "allocation.json", "active-process.json", "timing-dependency.json",
}
GENERATED_DIRS = {"cases", "checks", "incidents", "smoke-data", "data", "__pycache__"}


def file_records(path):
    """Hash regular files and record symlinks without following their targets."""
    path = Path(path)
    result = {}
    def visit(current, relative):
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            result[relative] = {"kind": "symlink", "target": os.readlink(current)}
        elif stat.S_ISDIR(mode):
            result[relative] = {"kind": "directory"}
            for child in sorted(current.iterdir()):
                visit(child, relative + "/" + child.name if relative else child.name)
        elif stat.S_ISREG(mode):
            digest = hashlib.sha256()
            with current.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[relative] = {"kind": "file", "sha256": digest.hexdigest(),
                                "size": current.stat().st_size}
        else:
            raise ValueError(f"Unsupported artifact type: {current}")
    visit(path, "")
    return result


def _plans():
    source_root = ROOT / "campaigns"
    if not source_root.exists():
        return []
    if source_root.is_symlink():
        raise ValueError("Campaign migration source must be this project's campaigns directory")
    ignored = set(subprocess.check_output([
        "git", "-C", str(ROOT), "ls-files", "--others", "--ignored",
        "--exclude-standard", "-z", "--", "campaigns",
    ]).decode().split("\0"))
    plans = []
    for campaign in sorted(source_root.iterdir()):
        if campaign.is_symlink() or not campaign.is_dir():
            continue
        entries, sources = {}, {}
        for item in sorted(campaign.iterdir()):
            if item.is_symlink():
                # A prior migration's compatibility alias is never moved again.
                continue
            relative = item.relative_to(ROOT).as_posix()
            artifact = ((item.name in GENERATED_DIRS or item.name.startswith("recovery-"))
                        if item.is_dir() else (item.name in GENERATED
                        or relative in ignored and item.suffix not in {".py", ".md"}))
            if artifact:
                entries[item.name] = file_records(item)
            elif item.is_file():
                sources[item.name] = file_records(item)[""]
        if entries:
            destination = new_run_directory("campaign-" + campaign.name)
            plans.append({"source": str(campaign.relative_to(ROOT)),
                          "destination": str(destination.relative_to(ROOT)),
                          "entries": entries, "legacy_source": sources})
    return plans


def _summary(plans, status):
    return {"status": status, "scope": "campaigns artifacts only; existing runs and diagnostics untouched",
            "campaigns": [{"source": p["source"], "destination": p["destination"],
                           "entries": list(p["entries"]),
                           "files": sum(v["kind"] != "directory" for records in p["entries"].values()
                                        for v in records.values())} for p in plans]}


def migrate_campaigns(apply=False):
    """Dry-run by default; move to fresh runs and retain old paths as aliases."""
    plans = _plans()
    if not apply or not plans:
        return _summary(plans, "dry_run" if not apply else "migrated")
    created, moved = [], []
    with ExitStack() as locks:
        lock_paths = set((ROOT / "locks").glob("*.lock"))
        lock_paths.update(p for p in (ROOT / "campaigns").rglob("*.lock") if not p.is_symlink())
        for path in sorted(lock_paths):
            stream = locks.enter_context(path.open("rb"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Campaign migration blocked by active lock: {path}") from error
        # Validate all inputs and destinations before the first move.
        for plan in plans:
            destination = ROOT / plan["destination"]
            new_run_directory(destination.name, destination)
            for name, records in plan["entries"].items():
                if file_records(ROOT / plan["source"] / name) != records:
                    raise ValueError(f"Campaign artifacts changed while planning: {plan['source']}/{name}")
        try:
            for plan in plans:
                source, destination = ROOT / plan["source"], ROOT / plan["destination"]
                destination.mkdir(parents=True)
                created.append(destination)
                snapshot = destination / "legacy-source"
                snapshot.mkdir()
                for name, record in plan["legacy_source"].items():
                    if file_records(source / name)[""] != record:
                        raise ValueError(f"Campaign source changed while archiving: {source / name}")
                    shutil.copy2(source / name, snapshot / name)
                for name, records in plan["entries"].items():
                    old, new = source / name, destination / name
                    os.rename(old, new)
                    moved.append((old, new))
                    old.symlink_to(os.path.relpath(new, old.parent), target_is_directory=new.is_dir())
                    if file_records(new) != records:
                        raise ValueError(f"Artifact contents changed during migration: {new}")
                write_json(destination / "migration.json", {
                    "schema_version": 1, "status": "historical_archive",
                    "measurement_status": "unchanged; migration is not a new validation",
                    **plan,
                })
        except BaseException:
            # Only new directories are removed, after every moved original is restored.
            for old, new in reversed(moved):
                if old.is_symlink():
                    old.unlink()
                os.rename(new, old)
            for destination in reversed(created):
                shutil.rmtree(destination)
            raise
    return _summary(plans, "migrated")


def restore_campaign_links():
    """Restore local history aliases from versioned manifests after a fresh clone."""
    campaign_root = ROOT / "campaigns"
    if campaign_root.is_symlink():
        raise ValueError("Cannot restore aliases through a redirected campaign directory")
    pending = []
    for manifest in sorted((ROOT / "runs").glob("campaign-*/migration.json")):
        record = read_json(manifest)
        source = Path(record["source"])
        if len(source.parts) != 2 or source.parts[0] != "campaigns" or source.parts[1] in {".", ".."}:
            raise ValueError("Invalid campaign migration source")
        if (ROOT / source).is_symlink() or (ROOT / source).resolve().parent != campaign_root.resolve():
            raise ValueError("Cannot restore aliases through a redirected campaign directory")
        destination = run_directory(ROOT / record["destination"])
        if destination != manifest.parent.resolve():
            raise ValueError("Migration destination does not match its manifest")
        for name in record["entries"]:
            if Path(name).name != name or name in {"", ".", ".."}:
                raise ValueError("Invalid campaign artifact name")
            old, new = ROOT / source / name, destination / name
            if not new.exists() and not new.is_symlink():
                continue  # Verbose artifacts are intentionally absent from a new clone.
            if old.is_symlink() and old.resolve() == new.resolve():
                continue
            if old.exists() or old.is_symlink():
                raise ValueError(f"Compatibility alias conflict: {old}")
            pending.append((old, new))
    for old, new in pending:
        old.parent.mkdir(parents=True, exist_ok=True)
        old.symlink_to(os.path.relpath(new, old.parent), target_is_directory=new.is_dir())
    return {"status": "links_restored", "restored": len(pending)}
