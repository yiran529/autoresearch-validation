"""CPU orchestration tests only. No torch import, data loading or training."""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from glval.build import instrument
from glval.common import ROOT, VENDOR, file_hash, inventory, read_json, write_json
from glval.methods import check_method, create_method
from glval.plans import make_plan, preflight, validate_recipe, verify_plan
from glval.results import inspect_job, report
from glval.runner import run_plan, probe_plan


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="_test-", dir=ROOT / "runs")
        self.path = Path(self.temp.name)
        self.method_dirs = []

    def tearDown(self):
        for path in self.method_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.temp.cleanup()

    def plan(self, workload="llama", output=None, method="original_greedylore", model_architecture=None):
        name = {"llama": "llama130m-muon-full-v1", "resnet": "resnet18-cifar10-muon-full-v1",
                "glue": "glue-sst2-bert-full-v1"}[workload]
        recipe = read_json(ROOT / "recipes" / (name + ".json"))
        recipe["required_paths"] = [str(self.path)]
        if model_architecture is not None:
            recipe["model_architecture"] = model_architecture
        write_json(self.path / "recipe.json", recipe)
        write_json(self.path / "experiment.json", {
            "schema_version": 1, "id": "unit-example", "method": method,
            "baselines": [] if method == "dense" else ["dense"], "recipe": str(self.path / "recipe.json"),
            "seeds": [1243], "execution": {"python": sys.executable}})
        return make_plan(self.path / "experiment.json", output or self.path / "run")

    def test_plan_rejects_outputs_outside_runs_without_creating_them(self):
        destination = ROOT / "campaigns" / ("test-" + uuid.uuid4().hex)
        try:
            with self.assertRaisesRegex(ValueError, "runs/"):
                self.plan(output=destination)
            self.assertFalse(destination.exists())
        finally:
            if destination.exists():
                shutil.rmtree(destination)

    def test_plan_rejects_run_root_and_duplicate_before_building(self):
        with self.assertRaisesRegex(ValueError, "run directory"):
            self.plan(output=ROOT / "runs")
        run = self.plan()
        original = inventory(run)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.plan(output=run)
        self.assertEqual(inventory(run), original)

    def test_plan_rejects_symlink_to_non_run_project_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as outside:
            link = self.path / "escape"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "runs/"):
                self.plan(output=link / "new-run")
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_baseline_checks_do_not_import_torch(self):
        before = set(sys.modules)
        check_method(ROOT / "methods/dense/method.json", "llama")
        check_method(ROOT / "methods/original_greedylore/method.json", "resnet")
        self.assertNotIn("torch", set(sys.modules) - before)

    def test_plan_is_static_and_preserves_sources(self):
        before = inventory(VENDOR)
        run = self.plan()
        plan = verify_plan(run)
        self.assertEqual(preflight(run)["status"], "static_pass")
        self.assertFalse(preflight(run)["training_executed"])
        self.assertEqual(len(plan["jobs"]), 2)
        self.assertEqual(inventory(VENDOR), before)
        self.assertEqual(read_json(run / "state.json")["status"], "planned")
        self.assertFalse(any((run / j["output"]).exists() for j in plan["jobs"]))

    def test_runtime_tamper_blocks_execution(self):
        run = self.plan()
        plan = verify_plan(run)
        (run / plan["jobs"][0]["runtime"] / "candidate_adapter.py").write_text("raise RuntimeError('changed')")
        with self.assertRaisesRegex(ValueError, "Runtime changed"):
            verify_plan(run)

    def test_formal_powersgd_preserves_the_verified_dense_exclusion_bridge(self):
        run = self.plan(method="powersgd-paper-r32")
        plan = verify_plan(run)
        hook = run / plan["jobs"][0]["runtime"] / "comm_hooks/powerSGD_hook.py"
        # Frozen hash from the completed 130M validation, independent of the new builder.
        self.assertEqual(file_hash(hook), "ba49b4f33dcc921d634302f8817d570c8605dd44c8bd4b856a6a33536f83f857")

    def test_formal_resnet50_builds_requested_architecture(self):
        run = self.plan(workload="resnet", method="dense", model_architecture="ResNet50")
        job = verify_plan(run)["jobs"][0]
        tree = ast.parse((run / job["runtime"] / "pytorch-cifar/train_ddp.py").read_text())
        build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model")
        namespace = {"ResNet": lambda block, sizes, num_classes: (block, sizes, num_classes),
                     "BasicBlock": "basic", "Bottleneck": "bottleneck"}
        exec(compile(ast.Module(body=[build], type_ignores=[]), "model-fixture", "exec"), namespace)
        self.assertEqual(namespace["build_model"]("cifar10"), ("bottleneck", [3, 4, 6, 3], 10))

    def test_plan_tamper_is_detected(self):
        run = self.plan()
        plan = read_json(run / "plan.json")
        plan["recipe"]["args"]["num_training_steps"] = 2
        write_json(run / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "plan was modified"):
            verify_plan(run)

    def test_training_requires_explicit_flag_without_gpu_probe(self):
        with patch("glval.runner.query_gpus", side_effect=AssertionError("GPU probe forbidden")):
            with self.assertRaisesRegex(ValueError, "--execute"):
                run_plan(self.path, "0,1,2,3", execute=False)
            with self.assertRaisesRegex(ValueError, "--execute"):
                probe_plan(self.path, "method-s1243", "0,1,2,3", 8, execute=False)

    def test_full_recipe_rejects_early_stop(self):
        recipe = read_json(ROOT / "recipes/llama130m-muon-full-v1.json")
        recipe["args"]["stop_after_steps"] = 5
        with self.assertRaisesRegex(ValueError, "Reserved"):
            validate_recipe(recipe)

    def test_glue_accumulation_is_explicitly_limited(self):
        recipe = read_json(ROOT / "recipes/glue-sst2-bert-full-v1.json")
        recipe["args"]["gradient_accumulation_steps"] = 4
        with self.assertRaisesRegex(ValueError, "accumulation=1"):
            validate_recipe(recipe)

    def test_new_checkpoint_policy_rejects_external_or_traversing_root(self):
        recipe = read_json(ROOT / "recipes/llama130m-muon-full-v1.json")
        for destination in ("/dev/shm/wyr_tmp/old", "../elsewhere", "/tmp/checkpoints"):
            recipe["checkpoint_policy"] = {"mode": "rolling", "every_updates": 1000, "root": destination}
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "checkpoint"):
                validate_recipe(recipe)
        recipe["checkpoint_policy"]["root"] = "checkpoints"
        validate_recipe(recipe)

    def test_passive_bucket_recording_is_supported_by_formal_timing_plan(self):
        recipe = read_json(ROOT / "recipes/llama60m-adamw-timing-bsz1-local-v1.json")
        recipe["required_paths"] = [str(self.path)]
        recipe["args"]["record_bucket_layout"] = True
        write_json(self.path / "recording-recipe.json", recipe)
        write_json(self.path / "recording.json", {"schema_version": 1, "id": "recording-unit",
                   "recipe": str(self.path / "recording-recipe.json"), "methods": ["dense"],
                   "seeds": [1243], "execution": {"python": sys.executable}})
        run = make_plan(self.path / "recording.json", self.path / "recording-run")
        plan = verify_plan(run)
        job = plan["jobs"][0]
        self.assertIn("--record_bucket_layout", job["command"])
        tree = ast.parse((run / job["runtime"] / "c4/table_v_timing.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register_timing_comm_hook")
        namespace = {}
        state_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BucketRecordingHookState")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[state_class, function], type_ignores=[])), "timing-fixture", "exec"), namespace)
        from types import SimpleNamespace
        from glval import bridge
        class Model:
            def register_comm_hook(self, state, hook): self.state, self.hook = state, hook
        model = Model()
        future = object()
        def attach(model, process_group, args, optimizer=None):
            model.register_comm_hook("method-state", lambda state, bucket: future)
        namespace["bucket_recording_hook"] = lambda state, bucket: state.inner_hook(state.inner_state, bucket)
        with patch.object(bridge, "register_comm_hook_for_ddp_model", side_effect=attach):
            observer = namespace["register_timing_comm_hook"](model, object(), SimpleNamespace(record_bucket_layout=True, strict_blocking_communication=False), object())
        self.assertEqual(observer.inner_state, "method-state")
        self.assertIs(model.hook(model.state, object()), future)
        self.assertEqual(model.register_comm_hook.__func__, Model.register_comm_hook)

    def test_all_driver_bridges_compile(self):
        for workload in ("llama", "resnet", "glue", "timing"):
            runtime = self.path / workload
            shutil.copytree(VENDOR, runtime)
            before = inventory(VENDOR)
            relative = instrument(runtime, workload)
            code = (runtime / relative).read_text()
            compile(code, relative, "exec")
            self.assertIn("glval.bridge", code)
            if workload == "resnet":
                self.assertIn("download=False", code)
                self.assertIn("attach_hook(model, args, optimizer)", code)
            if workload == "llama":
                self.assertIn("GLVAL_TOKENIZER_PATH", code)
            if workload == "glue":
                self.assertIn("cache_file_names=", code)
            if workload == "timing":
                self.assertIn("register_comm_hook_for_ddp_model(model, process_group, args, optimizer)", code)
                self.assertNotIn("['git', 'rev-parse'", code)
            self.assertEqual(inventory(VENDOR), before)

    def test_timing_plan_preserves_order_tracking_and_repetitions(self):
        recipe = read_json(ROOT / "recipes/llama130m-muon-full-v1.json")
        recipe.update(kind="timing", evaluation_split="none", required_paths=[str(self.path)],
                      completion={"status": "completed", "measured_steps": 800}, metrics=["mean_iteration_seconds"])
        recipe["args"] = {"model_config": "c4/configs/llama_130m.json", "batch_size": 1,
                          "warmup_iterations": 1201, "measured_iterations": 800,
                          "scheduler_steps": 8393, "lr_warmup_steps": 1000}
        write_json(self.path / "timing.json", recipe)
        write_json(self.path / "experiment.json", {"schema_version": 1, "id": "timing-unit",
                   "methods": ["original_greedylore", "dense"], "reference_methods": ["dense"],
                   "recipe": str(self.path / "timing.json"), "repetitions": 3, "seeds": [1243],
                   "execution": {"python": sys.executable, "tracking": {"mode": "online", "project": "unit"}}})
        run = make_plan(self.path / "experiment.json", self.path / "run")
        plan = verify_plan(run)
        self.assertEqual([j["method_id"] for j in plan["jobs"]], ["original_greedylore", "dense"] * 3)
        self.assertTrue(all(j["tracking"]["mode"] == "online" for j in plan["jobs"]))
        self.assertTrue(all("c4/table_v_timing.py" in j["command"] for j in plan["jobs"]))

    def test_report_does_not_invent_missing_results(self):
        run = self.plan()
        result = report(run, verify_plan(run))
        self.assertTrue(all(j["status"] == "unverified" for j in result["jobs"]))
        self.assertEqual(result["comparisons"], [])

    def test_result_rejects_missing_rank_and_wrong_budget(self):
        run = self.plan()
        plan = verify_plan(run)
        job = plan["jobs"][0]
        write_json(run / job["output"] / "all_results.json", {
            "status": "completed", "update_step": 2, "final_eval_loss": 3.,
            "final_eval_ppl": 20., "final_eval_tokens": 1000})
        result = inspect_job(run, job, plan["recipe"])
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("rank" in e for e in result["errors"]))
        self.assertTrue(any("Completion mismatch" in e for e in result["errors"]))

    def test_matching_results_compare_only_after_process_completion(self):
        run = self.plan()
        plan = verify_plan(run)
        state = read_json(run / "state.json")
        for index, job in enumerate(plan["jobs"]):
            output = run / job["output"]
            write_json(output / "all_results.json", {
                **plan["recipe"]["completion"], "final_eval_loss": 3. + index * .2,
                "final_eval_ppl": 20. + index, "final_eval_tokens": 1000})
            for rank in range(4):
                source = "comm_hooks/default_hooks.py"
                write_json(output / f"runtime-rank-{rank}.json", {
                    "rank": rank, "method_id": job["method_id"], "hook_registrations": 1,
                    "hook_calls": 20000, "last_bucket_calls": 20000, "hook_source": source,
                    "hook_sha256": job["runtime_files"][source], "peak_memory_bytes": 100})
            self.assertEqual(inspect_job(run, job, plan["recipe"])["status"], "verified")
            state["jobs"][job["id"]] = {"status": "completed", "elapsed_seconds": 2. + index}
        self.assertEqual(report(run, plan)["comparisons"], [])
        state["status"] = "completed"
        write_json(run / "state.json", state)
        result = report(run, plan)
        self.assertEqual(result["comparisons"][0]["total_wall_speedup"], 1.5)
        job = plan["jobs"][0]
        evidence = run / job["output"] / "runtime-rank-0.json"
        data = read_json(evidence)
        data["hook_sha256"] = "wrong"
        write_json(evidence, data)
        self.assertEqual(inspect_job(run, job, plan["recipe"])["status"], "invalid")

    def test_import_uses_commit_not_dirty_worktree_and_never_executes(self):
        repo = self.path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "comm_hooks").mkdir()
        (repo / "comm_hooks/hook.py").write_text("raise RuntimeError('must never execute during import')\n")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Scaffold Test", "-c",
                        "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)
        (repo / "comm_hooks/hook.py").write_text("dirty uncommitted text\n")
        name = "test-" + uuid.uuid4().hex[:10]
        target = create_method(name, repo, "HEAD")
        self.method_dirs.append(target)
        self.assertIn("must never execute", (target / "source/comm_hooks/hook.py").read_text())
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            check_method(target / "method.json", "llama")

    def test_existing_run_is_never_overwritten(self):
        self.plan()
        with self.assertRaisesRegex(ValueError, "already exists"):
            make_plan(self.path / "experiment.json", self.path / "run")


if __name__ == "__main__":
    unittest.main()
