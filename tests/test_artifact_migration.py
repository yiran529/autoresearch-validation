"""Migration moves only campaign artifacts and leaves old consumers usable."""
import fcntl
import importlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class CampaignMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / ".gitignore").write_text("/campaigns/*/*.log\n/campaigns/*/status.json\n")
        self.campaign = self.root / "campaigns" / "example"
        self.campaign.mkdir(parents=True)
        (self.campaign / "prepare.py").write_text("# original experiment source\n")
        (self.campaign / "experiment.json").write_text('{"schema_version": 1}\n')
        (self.campaign / "helpers").mkdir()
        (self.campaign / "helpers/helper.py").write_text("# source, not an artifact\n")
        (self.campaign / "summary.json").write_text('{"ppl": 12.5}\n')
        (self.campaign / "controller.log").write_text("verbose log\n")
        (self.campaign / "queue.json").write_text('["runs/already-here"]\n')
        (self.campaign / "cases" / "one" / "output").mkdir(parents=True)
        (self.campaign / "cases/one/output/all_results.json").write_text('{"mean_iteration_seconds": 0.5}\n')
        (self.campaign / "cases/one/output/latest").symlink_to("all_results.json")
        (self.root / "runs/already-here").mkdir(parents=True)
        (self.root / "runs/already-here/result.json").write_text("original run\n")
        (self.root / "diagnostics/check").mkdir(parents=True)
        (self.root / "diagnostics/check/result.json").write_text("original diagnostic\n")

    def module(self):
        module = importlib.import_module("glval.migration")
        self.scope = patch.object(module, "ROOT", self.root)
        self.scope.start()
        self.addCleanup(self.scope.stop)
        self.paths = patch("glval.artifacts.ROOT", self.root)
        self.paths.start()
        self.addCleanup(self.paths.stop)
        return module

    def test_dry_run_does_not_change_files_or_create_runs(self):
        module = self.module()
        plan = module.migrate_campaigns()
        self.assertEqual(plan["status"], "dry_run")
        self.assertFalse((self.root / "runs/campaign-example").exists())
        self.assertFalse((self.campaign / "summary.json").is_symlink())
        self.assertEqual(set(plan["campaigns"][0]["entries"]),
                         {"summary.json", "controller.log", "queue.json", "cases"})

    def test_move_keeps_bytes_and_old_paths_but_touches_no_other_results(self):
        module = self.module()
        original = (self.campaign / "summary.json").read_bytes()
        result = module.migrate_campaigns(apply=True)
        self.assertEqual(result["status"], "migrated")
        destination = self.root / "runs/campaign-example"
        self.assertEqual((destination / "summary.json").read_bytes(), original)
        self.assertTrue((self.campaign / "cases").is_symlink())
        self.assertEqual((self.campaign / "cases/one/output/latest").read_text(),
                         '{"mean_iteration_seconds": 0.5}\n')
        self.assertEqual(os.readlink(destination / "cases/one/output/latest"), "all_results.json")
        self.assertEqual((destination / "legacy-source/prepare.py").read_text(),
                         "# original experiment source\n")
        self.assertEqual((self.campaign / "experiment.json").read_text(), '{"schema_version": 1}\n')
        self.assertFalse((self.campaign / "helpers").is_symlink())
        self.assertEqual((self.campaign / "helpers/helper.py").read_text(), "# source, not an artifact\n")
        self.assertEqual((self.root / "runs/already-here/result.json").read_text(), "original run\n")
        self.assertEqual((self.root / "diagnostics/check/result.json").read_text(), "original diagnostic\n")
        self.assertTrue((destination / "migration.json").is_file())
        self.assertEqual(module.migrate_campaigns()["campaigns"], [])

    def test_target_collision_blocks_entire_migration(self):
        module = self.module()
        (self.root / "runs/campaign-example").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            module.migrate_campaigns(apply=True)
        self.assertFalse((self.campaign / "cases").is_symlink())
        self.assertTrue((self.campaign / "summary.json").is_file())

    def test_busy_controller_lock_blocks_before_any_move(self):
        module = self.module()
        with (self.campaign / "controller.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "active lock"):
                module.migrate_campaigns(apply=True)
        self.assertFalse((self.root / "runs/campaign-example").exists())
        self.assertFalse((self.campaign / "cases").is_symlink())

    def test_move_failure_rolls_back_preceding_moves(self):
        module = self.module()
        rename = os.rename
        def fail_later(source, destination):
            if Path(source) == self.campaign / "summary.json":
                raise OSError("synthetic disk failure")
            return rename(source, destination)
        with patch.object(module.os, "rename", side_effect=fail_later):
            with self.assertRaisesRegex(OSError, "synthetic disk failure"):
                module.migrate_campaigns(apply=True)
        self.assertFalse((self.root / "runs/campaign-example").exists())
        self.assertFalse((self.campaign / "cases").is_symlink())
        self.assertTrue((self.campaign / "controller.log").is_file())

    def test_clone_can_restore_missing_aliases_without_overwriting_files(self):
        module = self.module()
        module.migrate_campaigns(apply=True)
        (self.campaign / "cases").unlink()
        self.assertGreater(module.restore_campaign_links()["restored"], 0)
        self.assertTrue((self.campaign / "cases").is_symlink())
        (self.campaign / "summary.json").unlink()
        (self.campaign / "summary.json").write_text("conflicting local result\n")
        with self.assertRaisesRegex(ValueError, "conflict"):
            module.restore_campaign_links()
        self.assertEqual((self.campaign / "summary.json").read_text(), "conflicting local result\n")

    def test_alias_restoration_cannot_write_through_a_redirected_campaign(self):
        module = self.module()
        module.migrate_campaigns(apply=True)
        (self.campaign / "cases").unlink()
        redirected = self.root / "other-location"
        self.campaign.rename(redirected)
        self.campaign.symlink_to(redirected, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "campaign directory"):
            module.restore_campaign_links()
        self.assertFalse((redirected / "cases").exists())


if __name__ == "__main__":
    unittest.main()
