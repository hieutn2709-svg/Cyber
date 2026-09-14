from __future__ import annotations

import unittest
from unittest.mock import call, patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileBetaDiagnosticsTests(unittest.TestCase):
    def test_builds_prediction_side_diagnostics_for_every_frozen_beta(self) -> None:
        builder = getattr(gate_c_cli, "_build_diagnostics_by_beta", None)
        self.assertIsNotNone(
            builder,
            "Gate C diagnostics-by-beta helper must exist",
        )

        inferences = ("inference",)
        relation_types = ("uses", "targets")
        selected_threshold = 0.96
        canonicalization = "canonicalization"
        profile = "profile"

        diagnostics_by_beta = {
            0.0: {
                "argmax_change_count_vs_gate_a": 0,
                "argmax_change_rate_vs_gate_a": 0.0,
                "above_threshold_argmax_change_count_vs_gate_a": 0,
                "transition_counts": {},
            },
            0.25: {
                "argmax_change_count_vs_gate_a": 3,
                "argmax_change_rate_vs_gate_a": 0.03,
                "above_threshold_argmax_change_count_vs_gate_a": 1,
                "transition_counts": {"uses->targets": 3},
            },
            0.5: {
                "argmax_change_count_vs_gate_a": 5,
                "argmax_change_rate_vs_gate_a": 0.05,
                "above_threshold_argmax_change_count_vs_gate_a": 2,
                "transition_counts": {"uses->targets": 4, "targets->uses": 1},
            },
            1.0: {
                "argmax_change_count_vs_gate_a": 8,
                "argmax_change_rate_vs_gate_a": 0.08,
                "above_threshold_argmax_change_count_vs_gate_a": 3,
                "transition_counts": {"uses->targets": 6, "targets->uses": 2},
            },
            2.0: {
                "argmax_change_count_vs_gate_a": 12,
                "argmax_change_rate_vs_gate_a": 0.12,
                "above_threshold_argmax_change_count_vs_gate_a": 4,
                "transition_counts": {"uses->targets": 9, "targets->uses": 3},
            },
        }

        def fake_diagnostics(*args, **kwargs):
            return diagnostics_by_beta[float(kwargs["beta"])]

        with patch.object(
            gate_c_cli,
            "build_gate_c_diagnostics",
            side_effect=fake_diagnostics,
        ) as diagnostics_mock:
            report = builder(
                inferences,
                relation_types,
                gate_c_cli._EXPECTED_BETA_GRID,
                selected_threshold=selected_threshold,
                canonicalization=canonicalization,
                profile=profile,
                epsilon=gate_c_cli._EPSILON,
            )

        self.assertEqual(
            diagnostics_mock.call_args_list,
            [
                call(
                    inferences,
                    relation_types,
                    beta=beta,
                    threshold=selected_threshold,
                    canonicalization=canonicalization,
                    profile=profile,
                    epsilon=gate_c_cli._EPSILON,
                )
                for beta in gate_c_cli._EXPECTED_BETA_GRID
            ],
        )
        self.assertEqual(report["selected_threshold"], selected_threshold)
        self.assertEqual(
            [row["beta"] for row in report["by_beta"]],
            list(gate_c_cli._EXPECTED_BETA_GRID),
        )
        self.assertEqual(
            report["by_beta"][0],
            {
                "beta": 0.0,
                "argmax_change_count_vs_gate_a": 0,
                "argmax_change_rate_vs_gate_a": 0.0,
                "above_selected_threshold_change_count": 0,
                "transition_counts": {},
            },
        )
        self.assertEqual(
            report["by_beta"][1],
            {
                "beta": 0.25,
                "argmax_change_count_vs_gate_a": 3,
                "argmax_change_rate_vs_gate_a": 0.03,
                "above_selected_threshold_change_count": 1,
                "transition_counts": {"uses->targets": 3},
            },
        )
        for row in report["by_beta"]:
            self.assertEqual(
                set(row),
                {
                    "beta",
                    "argmax_change_count_vs_gate_a",
                    "argmax_change_rate_vs_gate_a",
                    "above_selected_threshold_change_count",
                    "transition_counts",
                },
            )


if __name__ == "__main__":
    unittest.main()
