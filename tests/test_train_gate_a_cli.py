from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "journal" / "scripts" / "train_gate_a.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_gate_a", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainGateACliTests(unittest.TestCase):
    def test_help_runs_without_loading_transformers(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mode", result.stdout)
        self.assertIn("--dataset", result.stdout)

    def test_only_full_mode_evaluates_test_split(self) -> None:
        module = load_module()
        self.assertFalse(module.mode_evaluates_test("overfit"))
        self.assertFalse(module.mode_evaluates_test("smoke"))
        self.assertTrue(module.mode_evaluates_test("full"))

    def test_mode_epoch_budget_uses_training_config(self) -> None:
        module = load_module()

        class Config:
            max_epochs = 12
            smoke_epochs = 2

        self.assertEqual(module.mode_epoch_budget("smoke", Config()), 2)
        self.assertEqual(module.mode_epoch_budget("full", Config()), 12)
        self.assertEqual(
            module.mode_epoch_budget("overfit", Config(), overfit_epochs=20),
            20,
        )


if __name__ == "__main__":
    unittest.main()
