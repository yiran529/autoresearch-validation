"""torchrun target; never imported by plan/check."""
import argparse
import os
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    driver = root / options.driver
    if not driver.resolve().is_relative_to(root):
        raise ValueError("Driver escapes runtime")
    sys.path.insert(0, str(driver.parent))
    sys.path.insert(0, str(root))
    os.chdir(root)
    sys.argv = [str(driver), *([*options.args][1:] if options.args[:1] == ["--"] else options.args)]
    runpy.run_path(str(driver), run_name="__main__")
    from glval.bridge import finish_runtime
    finish_runtime()
    from glval.common import read_json
    recipe = read_json(root / "recipe.json")
    if recipe.get("kind") == "timing":
        from glval.bridge import upload_timing
        upload_timing()
    elif int(os.environ.get("RANK", "0")) == 0:
        import wandb
        if wandb.run is not None:
            from glval.common import write_json
            write_json(Path(os.environ["GLVAL_OUTPUT"]) / "tracking.json",
                       {"id": wandb.run.id, "url": wandb.run.url, "mode": os.environ.get("WANDB_MODE")})
            wandb.finish()


if __name__ == "__main__":
    main()
