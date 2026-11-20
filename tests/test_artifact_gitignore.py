"""The repository keeps core measurements, not runtime copies and verbose outputs."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from glval.common import ROOT


class ArtifactGitignoreTests(unittest.TestCase):
    def test_core_results_are_addable_while_verbose_payloads_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            shutil.copyfile(ROOT / ".gitignore", root / ".gitignore")
            wanted = ["runs/example/plan.json", "runs/example/report.json", "runs/example/state.json",
                      "runs/campaign-example/status.json", "runs/campaign-example/130m-status.json",
                      "runs/campaign-example/continuation-result.json",
                      "runs/example/jobs/method/output/all_results.json",
                      "runs/campaign-example/summary.json", "runs/campaign-example/migration.json",
                      "runs/campaign-example/cases/method/output/all_results.json",
                      "runs/campaign-example/legacy-source/prepare.py"]
            ignored = ["runs/example/jobs/method/runtime/train.py",
                       "runs/example/jobs/method/output/train.log",
                       "runs/example/jobs/method/output/metrics.jsonl",
                       "runs/example/jobs/method/output/trace.json",
                       "runs/example/jobs/method/output/checkpoint.pt",
                       "runs/campaign-example/cases/method/runtime/train.py",
                       "runs/campaign-example/cases/method/output/trace.json"]
            for name in wanted + ignored:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                result = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", name])
                with self.subTest(path=name):
                    self.assertEqual(result.returncode == 0, name in ignored)


if __name__ == "__main__":
    unittest.main()
