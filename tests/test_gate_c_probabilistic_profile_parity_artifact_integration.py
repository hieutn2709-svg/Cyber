from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileParityArtifactIntegrationTests(unittest.TestCase):
    def test_validation_runtime_passes_verified_parity_report_to_artifact_builder(self) -> None:
        prepared = SimpleNamespace(
            model="model",
            inventory=SimpleNamespace(relation_types=("uses", "targets")),
            width_cap=8,
            base_config="base-config",
            training_config="training-config",
            device="cpu",
            canonicalization="canonicalization",
            profile="profile",
            run_id="gate-c-parity-artifact-integration",
            git_commit="commit",
            dataset_sha256="dataset",
            config_sha256="config",
            fold=1,
            seed=42,
        )
        validation_windows = ("validation-window",)
        inferences = ("gate-c-inference",)
        parity = {
            "checked_thresholds": list(gate_c_cli._EXPECTED_THRESHOLD_GRID),
            "prediction_parity": True,
            "mismatch_count": 0,
        }
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
        artifact_bundle = {"validation_beta_zero_parity.json": parity}

        with patch.object(
            gate_c_cli,
            "infer_gate_c_split",
            return_value=inferences,
        ), patch.object(
            gate_c_cli,
            "_check_beta_zero_parity",
            return_value=parity,
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
            parity=parity,
        )
        self.assertIs(result["artifacts"], artifact_bundle)


if __name__ == "__main__":
    unittest.main()
