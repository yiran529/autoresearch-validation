"""Locations for formal experiment artifacts; diagnostics remain independent."""
import os
from pathlib import Path

from .common import ROOT


def run_directory(path):
    """Validate a run location before any read, write, or GPU operation."""
    project = ROOT.resolve()
    base = project / "runs"
    lexical = Path(os.path.abspath(path))
    if not lexical.is_relative_to(project):
        # The checkout has a documented legacy project-name symlink. Accept that
        # alias, while still rejecting aliases rooted in campaigns/ or elsewhere.
        for parent in lexical.parents:
            if parent.resolve() == project:
                lexical = project / lexical.relative_to(parent)
                break
    target = lexical.resolve()
    if (not lexical.is_relative_to(base) or not target.is_relative_to(base)
            or not base.resolve().is_relative_to(project)):
        raise ValueError("Formal experiment artifacts must stay inside this project's runs/")
    if target == base:
        raise ValueError("Select a run directory below runs/, not runs/ itself")
    return target


def new_run_directory(run_id, output=None):
    """Select a fresh run; directory creation belongs to the plan transaction."""
    from .common import identifier
    if output is not None and Path(output).is_symlink() and not Path(output).exists():
        raise ValueError(f"Run already exists as a dangling symlink: {output}")
    target = run_directory(output if output is not None else ROOT / "runs" / identifier(run_id))
    if target.exists() or target.is_symlink():
        raise ValueError(f"Run already exists: {target}")
    return target
