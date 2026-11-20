"""Read historical metrics without rewriting their reports or validating new code."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from glval import results
from glval.artifacts import new_run_directory, run_directory
from glval.common import ROOT, write_json


class SavedArtifactTests(unittest.TestCase):
    def test_project_alias_still_uses_the_real_runs_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            project = parent / "project"
            project.mkdir()
            alias = parent / "legacy-project-name"
            alias.symlink_to(project, target_is_directory=True)
            with patch("glval.artifacts.ROOT", project):
                self.assertEqual(run_directory(alias / "runs/example"), project / "runs/example")

    def test_dangling_output_symlink_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "runs").mkdir()
            alias = project / "runs/claimed"
            alias.symlink_to("not-created-yet", target_is_directory=True)
            with patch("glval.artifacts.ROOT", project):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    new_run_directory("example", alias)

    def test_reader_exposes_ppl_and_time_without_modifying_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            run = project / "runs/history"
            write_json(run / "summary.json", {"status": "completed", "mean_iteration_seconds": 0.5})
            write_json(run / "130m-status.json", {"status": "completed", "final_eval_ppl": 12.5})
            output = run / "cases/method/output/all_results.json"
            write_json(output, {"final_eval_ppl": 12.5, "rank_losses": [[1, 2, 3]]})
            original = output.read_bytes()
            with patch("glval.artifacts.ROOT", project):
                payload = results.saved_results(run)
            self.assertEqual(payload["status"], "saved_results")
            files = {item["path"]: item for item in payload["files"]}
            self.assertEqual(files["cases/method/output/all_results.json"]["metrics"], {"final_eval_ppl": 12.5})
            self.assertEqual(files["summary.json"]["metrics"], {"mean_iteration_seconds": 0.5})
            self.assertEqual(files["130m-status.json"]["metrics"], {"final_eval_ppl": 12.5})
            self.assertEqual(output.read_bytes(), original)
            self.assertFalse((run / "report.json").exists())

    def test_reader_refuses_symlinked_results_outside_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            run = project / "runs/history"
            run.mkdir(parents=True)
            write_json(project / "external.json", {"final_eval_ppl": 12.5})
            (run / "summary.json").symlink_to(project / "external.json")
            with patch("glval.artifacts.ROOT", project):
                with self.assertRaisesRegex(ValueError, "escapes root"):
                    results.saved_results(run)

    def test_report_rejects_escaped_markdown_before_writing_either_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            run = project / "runs/new"
            write_json(run / "state.json", {"status": "planned", "jobs": {}})
            outside = project / "original.md"
            outside.write_text("preserve this file\n")
            (run / "report.md").symlink_to(outside)
            plan = {"id": "new", "plan_hash": "hash", "jobs": [],
                    "recipe": {"id": "recipe", "evaluation_split": "validation"}}
            with patch("glval.artifacts.ROOT", project):
                with self.assertRaisesRegex(ValueError, "escapes root"):
                    results.report(run, plan)
            self.assertEqual(outside.read_text(), "preserve this file\n")
            self.assertFalse((run / "report.json").exists())

    def test_retired_campaign_cli_stops_before_gpu_execution(self):
        command = [sys.executable, "-B", str(ROOT / "campaigns/powersgd-graph-timing-20261118/controller.py"), "--help"]
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
        self.assertEqual(process.returncode, 2)
        self.assertIn("glval", process.stderr)
        self.assertIn("archived", process.stderr)


if __name__ == "__main__":
    unittest.main()
