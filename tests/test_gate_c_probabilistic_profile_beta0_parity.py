from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileBetaZeroParityTests(unittest.TestCase):
    def test_beta_zero_runtime_parity_checks_every_frozen_threshold(self) -> None:
        checker = getattr(gate_c_cli, "_check_beta_zero_parity", None)
        self.assertIsNotNone(
            checker,
            "Gate C beta=0 runtime parity helper must exist",
        )

        inferences = (
            SimpleNamespace(base="gate-a-base-1"),
            SimpleNamespace(base="gate-a-base-2"),
        )
        prepared = SimpleNamespace(
            inventory=SimpleNamespace(relation_types=("uses", "targets")),
            canonicalization="canonicalization",
            profile="profile",
            run_id="gate-c-beta0-parity",
            git_commit="commit",
            dataset_sha256="dataset",
            config_sha256="config",
            fold=1,
            seed=42,
        )

        def gate_a_records(base_inferences, relation_types, threshold, **kwargs):
            self.assertEqual(base_inferences, tuple(item.base for item in inferences))
            self.assertEqual(relation_types, prepared.inventory.relation_types)
            self.assertEqual(kwargs["split"], "validation")
            return (("same-records", float(threshold)),)

        def gate_c_records(gate_c_inferences, relation_types, **kwargs):
            self.assertIs(gate_c_inferences, inferences)
            self.assertEqual(relation_types, prepared.inventory.relation_types)
            self.assertEqual(kwargs["beta"], 0.0)
            self.assertEqual(kwargs["split"], "validation")
            return (("same-records", float(kwargs["threshold"])),)

        with patch.object(
            gate_c_cli,
            "build_gate_a_prediction_records",
            side_effect=gate_a_records,
            create=True,
        ) as gate_a_mock, patch.object(
            gate_c_cli,
            "build_probabilistic_prediction_records",
            side_effect=gate_c_records,
        ) as gate_c_mock:
            report = checker(prepared, inferences)

        expected_thresholds = list(gate_c_cli._EXPECTED_THRESHOLD_GRID)
        self.assertEqual(gate_a_mock.call_count, len(expected_thresholds))
        self.assertEqual(gate_c_mock.call_count, len(expected_thresholds))
        self.assertEqual(
            [float(call.args[2]) for call in gate_a_mock.call_args_list],
            expected_thresholds,
        )
        self.assertEqual(
            [float(call.kwargs["threshold"]) for call in gate_c_mock.call_args_list],
            expected_thresholds,
        )
        self.assertEqual(
            report,
            {
                "checked_thresholds": expected_thresholds,
                "prediction_parity": True,
                "mismatch_count": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
