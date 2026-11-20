"""Load the method adapter's exact execution wrapper for diagnostic tests."""
import importlib.util
from pathlib import Path
_path = Path(__file__).resolve().parents[2] / "methods/powersgd-r32-graph-v1/adapter.py"
_spec = importlib.util.spec_from_file_location("powersgd_graph_adapter", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
GraphOrthogonalize = _module.GraphOrthogonalize
