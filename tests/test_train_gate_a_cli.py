from __future__ import annotations

import importlib.util
import inspect
import os
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
        self.assertFalse(module.mode_evaluates_test("dev"))
        self.assertTrue(module.mode_evaluates_test("full"))

    def test_mode_epoch_budget_uses_training_config(self) -> None:
        module = load_module()

        class Config:
            max_epochs = 12
            smoke_epochs = 2

        self.assertEqual(module.mode_epoch_budget("smoke", Config()), 2)
        self.assertEqual(module.mode_epoch_budget("dev", Config()), 12)
        self.assertEqual(module.mode_epoch_budget("full", Config()), 12)
        self.assertEqual(
            module.mode_epoch_budget("overfit", Config(), overfit_epochs=20),
            20,
        )

    def test_history_row_records_candidate_recall_curve(self) -> None:
        module = load_module()
        diagnostics = {
            "span_proposal": {"recall": 0.98},
            "span_post_pruning_typed": {"recall": 0.61},
            "pair_pre_distance": {"recall": 0.42},
            "pair_post_distance": {"recall": 0.40},
        }
        row = module.build_history_row(
            epoch=3,
            seconds=12.5,
            train_metrics={"total_loss": 1.25},
            selected={
                "relation_f1": 0.2,
                "primary_entity_f1": 0.3,
                "threshold": 0.5,
            },
            candidate_diagnostics=diagnostics,
        )
        self.assertEqual(row["epoch"], 3)
        self.assertEqual(
            row["candidate_recall"],
            {
                "span_proposal": 0.98,
                "span_post_pruning_typed": 0.61,
                "pair_pre_distance": 0.42,
                "pair_post_distance": 0.40,
            },
        )

    def test_seed_setup_enables_strict_deterministic_torch_runtime(self) -> None:
        import torch

        module = load_module()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_benchmark = torch.backends.cudnn.benchmark
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        old_cudnn_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
        old_matmul_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
        old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        try:
            module._set_seed(42)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                self.assertFalse(torch.backends.cudnn.allow_tf32)
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.use_deterministic_algorithms(old_deterministic)
            torch.backends.cudnn.benchmark = old_benchmark
            torch.backends.cudnn.deterministic = old_cudnn_deterministic
            if old_cudnn_tf32 is not None:
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            if old_matmul_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            if old_cublas is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas

    def test_driver_wires_split_relation_head_diagnostic_artifacts(self) -> None:
        module = load_module()
        self.assertTrue(
            hasattr(module, "_relation_head_diagnostics_for_split"),
            "driver must aggregate gold-span relation diagnostics per split",
        )
        run_source = inspect.getsource(module.run)
        self.assertIn(
            "validation_relation_head_diagnostics.json",
            run_source,
            "validation diagnostics artifact must always be written after checkpoint freeze",
        )
        self.assertIn(
            "test_relation_head_diagnostics.json",
            run_source,
            "full mode must write test diagnostics with the frozen validation threshold",
        )


if __name__ == "__main__":
    unittest.main()
