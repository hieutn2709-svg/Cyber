from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileRunArtifactTests(unittest.TestCase):
    def test_validation_run_artifacts_cover_frozen_journal_bundle(self) -> None:
        builder = getattr(gate_c_cli, "_build_validation_run_artifacts", None)
        self.assertIsNotNone(
            builder,
            "Gate C validation run-level artifact builder must exist",
        )

        inventory = SimpleNamespace(relation_types=("uses", "targets"))
        prepared = SimpleNamespace(
            inventory=inventory,
            canonicalization="canonicalization",
            profile="profile",
            device="cpu",
            run_id="gate-c-artifact-test",
            fold=1,
            seed=42,
            git_commit="c" * 40,
            dataset_sha256="a" * 64,
            config_sha256="b" * 64,
            canonicalization_sha256="d" * 64,
            task_profile_sha256="e" * 64,
            gate_a_checkpoint_sha256="f" * 64,
            checkpoint={
                "git_commit": gate_c_cli._FROZEN_GATE_A_COMMIT,
                "dataset_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "epoch": 10,
                "threshold": 0.96,
                "width_cap": 8,
                "model_state_dict": {"weight": "must-not-be-serialized"},
            },
            run_metadata={
                "fold": 1,
                "seed": 42,
                "dataset_sha256": "a" * 64,
                "combined_config_sha256": "b" * 64,
            },
        )
        inferences = ("inference",)
        records = ("record",)
        selection = {
            "best": {
                "beta": 0.25,
                "threshold": 0.96,
                "relation_f1": 0.55,
                "primary_entity_f1": 0.66,
            },
            "grid": [],
        }
        metrics = {
            "all_relation": {"f1": 0.55},
            "primary_entity": {"f1": 0.66},
        }
        prediction_rows = ({"prediction": "row"},)
        posterior_rows = ({"posterior": "row"},)
        pair_rows = ({"pair": "row"},)
        diagnostics = {"argmax_change_count_vs_gate_a": 3}
        environment = {"device": "cpu"}

        with patch.object(
            gate_c_cli,
            "_record_to_dict",
            return_value=prediction_rows[0],
            create=True,
        ) as record_mock, patch.object(
            gate_c_cli,
            "build_gate_c_span_rows",
            return_value=posterior_rows,
            create=True,
        ) as span_mock, patch.object(
            gate_c_cli,
            "build_gate_c_scored_pair_rows",
            return_value=pair_rows,
            create=True,
        ) as pair_mock, patch.object(
            gate_c_cli,
            "build_gate_c_diagnostics",
            return_value=diagnostics,
            create=True,
        ) as diagnostics_mock, patch.object(
            gate_c_cli,
            "_environment",
            return_value=environment,
            create=True,
        ) as environment_mock:
            artifacts = builder(
                prepared,
                inferences,
                records,
                selection,
                metrics,
                beta=0.25,
                threshold=0.96,
                mode="dev",
            )

        self.assertEqual(
            set(artifacts),
            {
                "validation_predictions.jsonl",
                "validation_entity_posteriors.jsonl",
                "validation_scored_pairs.jsonl",
                "validation_decoder_selection.json",
                "validation_metrics.json",
                "validation_profile_diagnostics.json",
                "environment.json",
                "hashes.json",
                "checkpoint_metadata.json",
                "run_summary.json",
            },
        )
        self.assertEqual(artifacts["validation_predictions.jsonl"], prediction_rows)
        self.assertEqual(
            artifacts["validation_entity_posteriors.jsonl"], posterior_rows
        )
        self.assertEqual(artifacts["validation_scored_pairs.jsonl"], pair_rows)
        self.assertEqual(artifacts["validation_decoder_selection.json"], selection)
        self.assertEqual(artifacts["validation_metrics.json"], metrics)
        self.assertEqual(
            artifacts["validation_profile_diagnostics.json"], diagnostics
        )
        self.assertEqual(artifacts["environment.json"], environment)

        record_mock.assert_called_once_with("record")
        span_mock.assert_called_once_with(inferences)
        pair_mock.assert_called_once_with(
            inferences,
            prepared.inventory.relation_types,
            beta=0.25,
            canonicalization=prepared.canonicalization,
            profile=prepared.profile,
            epsilon=gate_c_cli._EPSILON,
        )
        diagnostics_mock.assert_called_once_with(
            inferences,
            prepared.inventory.relation_types,
            beta=0.25,
            threshold=0.96,
            canonicalization=prepared.canonicalization,
            profile=prepared.profile,
            epsilon=gate_c_cli._EPSILON,
        )
        environment_mock.assert_called_once_with(prepared.device)

        hashes = artifacts["hashes.json"]
        self.assertEqual(hashes["dataset_sha256"], prepared.dataset_sha256)
        self.assertEqual(
            hashes["gate_a_combined_config_sha256"], prepared.config_sha256
        )
        self.assertEqual(
            hashes["canonicalization_sha256"], prepared.canonicalization_sha256
        )
        self.assertEqual(
            hashes["task_profile_sha256"], prepared.task_profile_sha256
        )
        self.assertEqual(
            hashes["gate_a_checkpoint_sha256"], prepared.gate_a_checkpoint_sha256
        )

        checkpoint_metadata = artifacts["checkpoint_metadata.json"]
        self.assertNotIn(
            "model_state_dict",
            checkpoint_metadata["checkpoint"],
        )
        self.assertEqual(
            checkpoint_metadata["companion_training_run_config"],
            prepared.run_metadata,
        )

        summary = artifacts["run_summary.json"]
        required_summary = {
            "status": "dev_complete",
            "mode": "dev",
            "fold": 1,
            "seed": 42,
            "selection_scope": "validation-only",
            "beta": 0.25,
            "relation_threshold": 0.96,
            "validation": selection["best"],
            "test_evaluated": False,
            "gate_a_parent_commit": gate_c_cli._FROZEN_GATE_A_COMMIT,
            "gate_b_parent_commit": "2e74e98231c3c5bb1b0db4d826602b61b71ba07d",
            "git_commit": prepared.git_commit,
            "dataset_sha256": prepared.dataset_sha256,
            "config_sha256": prepared.config_sha256,
            "canonicalization_sha256": prepared.canonicalization_sha256,
            "task_profile_sha256": prepared.task_profile_sha256,
            "epsilon": gate_c_cli._EPSILON,
            "beta_grid": list(gate_c_cli._EXPECTED_BETA_GRID),
        }
        for key, expected in required_summary.items():
            self.assertEqual(summary[key], expected, key)


if __name__ == "__main__":
    unittest.main()
