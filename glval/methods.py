"""Git export and explicit candidate review. No candidate Python is imported here."""
import ast
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

from .common import ROOT, VENDOR, digest, git, identifier, inside, inventory, read_json, write_json

RUNTIME_ROOTS = {"comm_hooks", "optimizer", "c4", "glue", "pytorch-cifar"}
PROTECTED = {"optimizer", "c4", "glue", "pytorch-cifar", "glval"}


def export_source(repo, commit, target):
    """Export tracked source blobs only. Never inspect working tree or tensor payloads."""
    sha = git(repo, "rev-parse", "--verify", commit + "^{commit}")
    raw = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-rlz", sha])
    paths, size = [], 0
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, kind, _, length = meta.split()
        name = name.decode()
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Invalid Git source path")
        if p.suffix not in {".py", ".json", ".toml", ".yaml", ".yml"}:
            continue
        if any(part.startswith(".") or part in {"output", "outputs", "results", "runs", "tests"}
               for part in p.parts[:-1]):
            continue
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            raise ValueError(f"Source must be a regular blob: {name}")
        size += int(length)
        if int(length) > 4_000_000 or size > 64_000_000 or len(paths) >= 2000:
            raise ValueError("Source export exceeds limits; narrow the candidate repository")
        paths.append(name)
    if not paths:
        raise ValueError("No tracked source files at candidate SHA")
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", sha, "--", *paths])
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            destination = inside(target, member.name)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(tar.extractfile(member).read())
            else:
                raise ValueError(f"Nonregular archive entry: {member.name}")
    return sha


def changes(source):
    base, candidate = inventory(VENDOR), inventory(source)
    names = {p for p in set(base) | set(candidate)
             if Path(p).parts[0] in RUNTIME_ROOTS and Path(p).suffix in {".py", ".json"}}
    return {p: {"base": base.get(p), "candidate": candidate.get(p)}
            for p in sorted(names) if base.get(p) != candidate.get(p)}


def create_method(name, repo, commit):
    identifier(name)
    target = ROOT / "methods" / name
    if target.exists():
        raise ValueError(f"Method already exists: {name}")
    target.mkdir(parents=True)
    try:
        sha = export_source(Path(repo).resolve(), commit, target / "source")
        shutil.copyfile(ROOT / "methods" / "original_greedylore" / "adapter.py", target / "adapter.py")
        diff = changes(target / "source")
        write_json(target / "source_manifest.json", {
            "repo": str(Path(repo).resolve()), "commit": sha,
            "files": inventory(target / "source"), "changes_from_base": diff,
        })
        write_json(target / "method.json", {
            "schema_version": 1, "id": name, "kind": "ddp_comm_hook", "source": "source",
            "source_commit": sha, "adapter": "adapter.py", "status": "draft",
            "supported_workloads": [], "support_notes": {},
            "parameters": {"compressor": "top_subspace", "compress_rank": 32,
                           "update_proj_gap": 200, "use_error_feedback": "ef14",
                           "beta_ef": 0.0, "error_inherit": 0},
            "overlay": ["comm_hooks"], "reviewed_changes": {},
        })
        (target / "adaptation.md").write_text(
            f"# {name}\n\nSource: `{sha}`\n\n状态：待适配。\n"
            "\n请记录方法语义、依赖、训练入口改动、各 workload 接入依据、预检及未覆盖行为。\n")
        return target
    except BaseException:
        shutil.rmtree(target)
        raise


def check_method(path, workload=None):
    path = Path(path).resolve()
    method = read_json(path)
    identifier(method["id"])
    if method.get("schema_version") != 1 or method.get("kind") != "ddp_comm_hook":
        raise ValueError("Only schema v1 / ddp_comm_hook is supported")
    adapter = inside(path.parent, method["adapter"])
    ast.parse(adapter.read_text(), filename=str(adapter))
    builtin = method.get("builtin", False)
    if builtin and path.parent not in {(ROOT / "methods" / "dense").resolve(),
                                        (ROOT / "methods" / "original_greedylore").resolve()}:
        raise ValueError("Only shipped baseline methods may declare builtin")
    source = VENDOR if builtin else inside(path.parent, method["source"])
    if builtin:
        if inventory(source) != read_json(ROOT / "vendor" / "provenance.json")["files"]:
            raise ValueError("Pinned baseline files changed")
    else:
        manifest = read_json(path.parent / "source_manifest.json")
        if inventory(source) != manifest["files"] or method["source_commit"] != manifest["commit"]:
            raise ValueError("Candidate snapshot changed; import a new method version")
        diff = changes(source)
        if diff != manifest["changes_from_base"]:
            raise ValueError("Baseline changed since candidate import")
        review = method.get("reviewed_changes", {})
        for name in diff:
            item = review.get(name, {})
            if item.get("disposition") not in {"overlay", "not_required"} or not item.get("reason", "").strip():
                raise ValueError(f"Unreviewed source difference: {name}")
            selected = any(name == p or name.startswith(p.rstrip("/") + "/") for p in method["overlay"])
            if (item["disposition"] == "overlay") != selected:
                raise ValueError(f"Review disagrees with overlay: {name}")
    if method.get("status") != "adapted":
        raise ValueError("Method remains draft; review and adapt before planning")
    if workload and (workload not in method["supported_workloads"] or
                     not method.get("support_notes", {}).get(workload, "").strip()):
        raise ValueError(f"Method has no documented support for {workload}")
    for name in method["overlay"]:
        p = Path(name)
        if not p.parts or p.parts[0] in PROTECTED or p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Protected or invalid overlay: {name}")
        if not inside(source, name).exists():
            raise ValueError(f"Missing overlay source: {name}")
    for name, value in method["parameters"].items():
        if not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"Invalid method parameter: {name}")
        if name in {"optimizer", "lr", "learning_rate", "seed", "epochs", "num_train_epochs",
                    "num_training_steps", "stop_after_steps", "model_config", "dataset_path",
                    "output_dir", "batch_size", "total_batch_size", "gradient_accumulation",
                    "start_compress_iter"}:
            raise ValueError(f"Training parameter belongs in a recipe: {name}")
    return method, source, adapter
