"""Build CPU-only campaign fixtures instead of reading historical experiment outputs."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import shutil
import tempfile

def build_fixture(script_path, reuse_variant=False):
    temporary = tempfile.TemporaryDirectory(prefix="glval-campaign-")
    destination = Path(temporary.name)
    namespace = runpy.run_path(str(script_path))
    main = namespace["main"]
    scope = main.__globals__
    scope["HERE"] = destination
    for name in ("worker.py", "run.py"):
        source = script_path.parent / name
        if source.exists():
            shutil.copyfile(source, destination / name)
    if reuse_variant:
        tree = ast.parse(script_path.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "main")
        # The fixed method variant is already committed. Rebuild only the cases.
        start = next(index for index, node in enumerate(function.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "cases"
                             for target in node.targets))
        function.body = function.body[start:]
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     str(script_path), "exec"), scope)
        main = scope["main"]
    try:
        with redirect_stdout(io.StringIO()):
            main()
    except BaseException:
        temporary.cleanup()
        raise
    return temporary, destination

