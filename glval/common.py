import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "greedy_lore"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(path):
    path = Path(path)
    result = {}
    for f in sorted(path.rglob("*")):
        if "__pycache__" in f.parts or f.suffix == ".pyc":
            continue
        if f.is_symlink():
            raise ValueError(f"Symlinks are not allowed in frozen code: {f}")
        if f.is_file():
            result[f.relative_to(path).as_posix()] = file_hash(f)
    return result


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def identifier(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value):
        raise ValueError(f"Invalid identifier: {value}")
    return value


def inside(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes root: {name}")
    return path


def load_ref(value, directory):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    return ROOT / directory / identifier(value) / "method.json" if directory == "methods" else ROOT / directory / (identifier(value) + ".json")
