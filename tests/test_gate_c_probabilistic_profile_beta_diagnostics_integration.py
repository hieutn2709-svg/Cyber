from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileBetaDiagnosticsIntegrationTests(unittest.TestCase):
    def test_validation_runtime_builds_and_forwards_diagnostics_by_beta(self) -> None:
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
            run_id="gate-c-beta-diagnostics-integration",
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
        diagnostics_by_beta = {
            "selected_threshold": 0.96,
            "by_beta": [
                {
                    "beta": beta,
                    "argmax_change_count_vs_gate_a": 0,
                    "argmax_change_rate_vs_gate_a": 0.0,
                    "above_selected_threshold_change_count": 0,
                    "transition_counts": {},
                }
                for beta in gate_c_cli._EXPECTED_BETA_GRID
            ],
        }
        artifact_bundle = {"validation_bundle": True}

        with patch.object(
            gate_c_cli,
            "infer_gate_c_split",
            return_value=inferences,
        ), patch.object(
            gate_c_cli,
            "_check_beta_zero_parity",
            return_value=parity,
        ), patch.object(
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
            "_build_diagnostics_by_beta",
            return_value=diagnostics_by_beta,
        ) as diagnostics_mock, patch.object(
            gate_c_cli,
            "_build_validation_run_artifacts",
            return_value=artifact_bundle,
        ) as artifact_mock:
            result = gate_c_cli._evaluate_validation(prepared, validation_windows)

        diagnostics_mock.assert_called_once_with(
            inferences,
            prepared.inventory.relation_types,
            gate_c_cli._EXPECTED_BETA_GRID,
            selected_threshold=0.96,
            canonicalization=prepared.canonicalization,
            profile=prepared.profile,
            epsilon=gate_c_cli._EPSILON,
        )
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
            diagnostics_by_beta=diagnostics_by_beta,
        )
        self.assertIs(result["artifacts"], artifact_bundle)
        self.assertFalse(result["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
