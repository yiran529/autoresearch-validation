"""Protect this campaign's approved training and output constraints, without torch."""
import copy
from pathlib import Path
import runpy
import unittest

from glval.common import read_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "campaigns/powersgd-350m-full-20261118/prepare.py"


class PowerSGD350MContract(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), "The approved 350M campaign is not implemented")
        self.validate = runpy.run_path(str(SCRIPT))["validate_contract"]
        self.recipe = read_json(ROOT / "recipes/llama350m-muon-full-offline-v5.json")
        self.execution = read_json(
            ROOT / "campaigns/powersgd-validation-20261109/llama130m.json"
        )["execution"]
        self.execution["tracking"]["mode"] = "disabled"

    def test_accepts_approved_full_and_smoke_protocol(self):
        self.validate(self.recipe, self.execution, "full")
        smoke = copy.deepcopy(self.recipe)
        smoke["args"]["num_training_steps"] = 8
        smoke["compression_start_step"] = 2
        smoke["completion"].update(update_step=8, completed_steps=8, expected_steps=8)
        self.validate(smoke, self.execution, "smoke")

    def test_rejects_tracking_and_checkpoint_reactivation(self):
        for mode in ("offline", "online"):
            execution = copy.deepcopy(self.execution)
            execution["tracking"]["mode"] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "W&B"):
                self.validate(self.recipe, execution, "full")
        recipe = copy.deepcopy(self.recipe)
        recipe["checkpoint_policy"] = {"mode": "rolling", "every_updates": 20000}
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.validate(recipe, self.execution, "full")

    def test_rejects_accidental_training_drift(self):
        for key, value in (("lr", .01), ("batch_size", 1), ("optimizer", "adamw"),
                           ("dtype", "float32"), ("num_training_steps", 8)):
            recipe = copy.deepcopy(self.recipe)
            recipe["args"][key] = value
            with self.subTest(argument=key), self.assertRaisesRegex(ValueError, "Training settings"):
                self.validate(recipe, self.execution, "full")
        recipe = copy.deepcopy(self.recipe)
        recipe["compression_start_step"] = 2
        with self.assertRaisesRegex(ValueError, "compression"):
            self.validate(recipe, self.execution, "full")


if __name__ == "__main__":
    unittest.main()
