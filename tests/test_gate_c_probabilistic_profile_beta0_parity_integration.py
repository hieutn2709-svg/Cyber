from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileBetaZeroParityIntegrationTests(unittest.TestCase):
    def test_validation_hard_fails_before_selection_on_beta_zero_mismatch(self) -> None:
        prepared = SimpleNamespace(
            model="model",
            inventory=SimpleNamespace(relation_types=("uses", "targets")),
            width_cap=8,
            base_config="base-config",
            training_config="training-config",
            device="cpu",
            canonicalization="canonicalization",
            profile="profile",
            run_id="gate-c-beta0-hard-gate",
            git_commit="commit",
            dataset_sha256="dataset",
            config_sha256="config",
            fold=1,
            seed=42,
        )
        validation_windows = ("validation-window",)
        inferences = ("gate-c-inference",)
        mismatch_report = {
            "checked_thresholds": list(gate_c_cli._EXPECTED_THRESHOLD_GRID),
            "prediction_parity": False,
            "mismatch_count": 1,
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

        with patch.object(
            gate_c_cli,
            "infer_gate_c_split",
            return_value=inferences,
        ), patch.object(
            gate_c_cli,
            "_check_beta_zero_parity",
            return_value=mismatch_report,
        ) as parity_mock, patch.object(
            gate_c_cli,
            "_select_probabilistic_decoder",
            return_value=selection,
        ) as selector_mock, patch.object(
            gate_c_cli,
            "build_probabilistic_prediction_records",
            return_value=("validation-record",),
        ), patch.object(
            gate_c_cli,
            "_score_records",
            return_value={
                "all_relation": {"f1": 0.55},
                "primary_entity": {"f1": 0.66},
            },
        ), patch.object(
            gate_c_cli,
            "_build_validation_run_artifacts",
            return_value={"bundle": True},
        ):
            try:
                gate_c_cli._evaluate_validation(prepared, validation_windows)
            except Exception as exc:
                self.assertIs(type(exc), RuntimeError)
                self.assertEqual(str(exc), "Gate C beta=0 parity failure")
            else:
                self.fail("beta=0 parity mismatch must hard-fail")

        parity_mock.assert_called_once_with(prepared, inferences)
        selector_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
