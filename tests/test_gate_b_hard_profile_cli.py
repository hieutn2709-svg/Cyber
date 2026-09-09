from __future__ import annotations

import copy
import unittest

from journal.scripts.evaluate_gate_b_hard_profile import (
    mode_evaluates_test,
    requested_evaluation_splits,
    validate_gate_a_provenance,
)


class GateBHardProfileCliTests(unittest.TestCase):
    def test_mode_guard_keeps_test_out_of_dev(self):
        self.assertFalse(mode_evaluates_test("dev"))
        self.assertTrue(mode_evaluates_test("full"))
        self.assertEqual(requested_evaluation_splits("dev"), ("validation",))
        self.assertEqual(
            requested_evaluation_splits("full"), ("validation", "test")
        )
        with self.assertRaises(ValueError):
            mode_evaluates_test("smoke")

    def _valid_provenance(self):
        dataset_sha = "a" * 64
        config_sha = "b" * 64
        checkpoint = {
            "model_state_dict": {},
            "git_commit": "b4033edbaf2150605c286a36e4b0564d75b0ac91",
            "dataset_sha256": dataset_sha,
            "config_sha256": config_sha,
            "width_cap": 8,
        }
        run_metadata = {
            "fold": 1,
            "seed": 42,
            "dataset_sha256": dataset_sha,
            "combined_config_sha256": config_sha,
        }
        return checkpoint, run_metadata, dataset_sha, config_sha

    def test_provenance_accepts_matching_frozen_gate_a_checkpoint(self):
        checkpoint, run_metadata, dataset_sha, config_sha = self._valid_provenance()
        validate_gate_a_provenance(
            checkpoint,
            run_metadata,
            dataset_sha256=dataset_sha,
            config_sha256=config_sha,
            fold=1,
            seed=42,
        )

    def test_provenance_rejects_dataset_config_fold_and_seed_mismatch(self):
        checkpoint, run_metadata, dataset_sha, config_sha = self._valid_provenance()
        cases = (
            ("checkpoint_dataset", {"dataset_sha256": "c" * 64}, {}),
            ("checkpoint_config", {"config_sha256": "d" * 64}, {}),
            ("fold", {}, {"fold": 2}),
            ("seed", {}, {"seed": 123}),
        )
        for name, checkpoint_patch, metadata_patch in cases:
            with self.subTest(name=name):
                bad_checkpoint = copy.deepcopy(checkpoint)
                bad_metadata = copy.deepcopy(run_metadata)
                bad_checkpoint.update(checkpoint_patch)
                bad_metadata.update(metadata_patch)
                with self.assertRaises(ValueError):
                    validate_gate_a_provenance(
                        bad_checkpoint,
                        bad_metadata,
                        dataset_sha256=dataset_sha,
                        config_sha256=config_sha,
                        fold=1,
                        seed=42,
                    )

    def test_provenance_rejects_wrong_gate_a_lineage(self):
        checkpoint, run_metadata, dataset_sha, config_sha = self._valid_provenance()
        checkpoint["git_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "lineage mismatch"):
            validate_gate_a_provenance(
                checkpoint,
                run_metadata,
                dataset_sha256=dataset_sha,
                config_sha256=config_sha,
                fold=1,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
