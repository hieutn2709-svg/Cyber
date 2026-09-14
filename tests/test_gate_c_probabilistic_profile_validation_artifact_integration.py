from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileValidationArtifactIntegrationTests(unittest.TestCase):
    def test_validation_runtime_returns_full_frozen_artifact_bundle(self) -> None:
        inventory = SimpleNamespace(relation_types=("uses", "targets"))
        prepared = SimpleNamespace(
            model="model",
            inventory=inventory,
            width_cap=8,
            base_config="base-config",
            training_config="training-config",
            device="cpu",
            canonicalization="canonicalization",
            profile="profile",
            run_id="gate-c-validation-artifact-integration",
            git_commit="commit",
            dataset_sha256="dataset-sha",
            config_sha256="config-sha",
            fold=1,
            seed=42,
        )
        validation_windows = ("validation-window",)
        inferences = ("gate-c-inference",)
        selection = {
            "best": {
                "beta": 0.25,
                "threshold": 0.96,
                "relation_f1": 0.55,
                "primary_entity_f1": 0.66,
            },
            "grid": [],
        }
        records = ("validation-record",)
        metrics = {
            "all_relation": {"f1": 0.55},
            "primary_entity": {"f1": 0.66},
        }
        artifact_bundle = {
            "validation_predictions.jsonl": ({"prediction": "row"},),
            "validation_entity_posteriors.jsonl": ({"posterior": "row"},),
            "validation_scored_pairs.jsonl": ({"pair": "row"},),
            "validation_decoder_selection.json": selection,
            "validation_metrics.json": metrics,
            "validation_profile_diagnostics.json": {"diagnostics": True},
            "environment.json": {"device": "cpu"},
            "hashes.json": {"dataset_sha256": "dataset-sha"},
            "checkpoint_metadata.json": {"checkpoint": {}},
            "run_summary.json": {"test_evaluated": False},
        }
        parity_report = {
            "checked_thresholds": list(gate_c_cli._EXPECTED_THRESHOLD_GRID),
            "prediction_parity": True,
            "mismatch_count": 0,
        }

        with patch.object(
            gate_c_cli,
            "infer_gate_c_split",
            return_value=inferences,
        ), patch.object(
            gate_c_cli,
            "_check_beta_zero_parity",
            return_value=parity_report,
        ) as parity_mock, patch.object(
            gate_c_cli,
            "_select_probabilistic_decoder",
            return_value=selection,
        ), patch.object(
            gate_c_cli,
            "build_probabilistic_prediction_records",
            return_value=records,
        ), patch.object(
            gate_c_cli,
            "_score_records",
            return_value=metrics,
        ), patch.object(
            gate_c_cli,
            "_build_validation_run_artifacts",
            return_value=artifact_bundle,
        ) as artifact_mock:
            result = gate_c_cli._evaluate_validation(prepared, validation_windows)

        parity_mock.assert_called_once_with(prepared, inferences)
        artifact_mock.assert_called_once_with(
            prepared,
            inferences,
            records,
            selection,
            metrics,
            beta=0.25,
            threshold=0.96,
            mode="dev",
        )
        self.assertIs(result["artifacts"], artifact_bundle)
        self.assertEqual(result["beta"], 0.25)
        self.assertEqual(result["relation_threshold"], 0.96)
        self.assertEqual(result["validation"], selection["best"])
        self.assertEqual(result["validation_metrics"], metrics)
        self.assertFalse(result["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
