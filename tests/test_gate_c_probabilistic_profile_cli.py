from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli
except ImportError:
    gate_c_cli = None


class GateCProbabilisticProfileCliTests(unittest.TestCase):
    EXPECTED_BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
    EXPECTED_THRESHOLDS = tuple(round(value / 100, 2) for value in range(85, 100))

    def _assert_module(self) -> None:
        self.assertIsNotNone(
            gate_c_cli,
            "Gate C probabilistic-profile evaluator module must exist",
        )

    def _selector(self, score_map):
        self._assert_module()
        selector = getattr(gate_c_cli, "_select_probabilistic_decoder", None)
        self.assertIsNotNone(selector, "Gate C validation selector must exist")

        def fake_build_records(
            inferences,
            relation_types,
            *,
            beta,
            threshold,
            **kwargs,
        ):
            return ({"beta": float(beta), "threshold": float(threshold)},)

        def fake_score_records(records, inventory):
            token = records[0]
            relation_f1, entity_f1 = score_map.get(
                (token["beta"], token["threshold"]),
                (0.0, 0.0),
            )
            return {
                "all_relation": {"f1": float(relation_f1)},
                "primary_entity": {"f1": float(entity_f1)},
            }

        inventory = SimpleNamespace(relation_types=("uses",))
        with patch.object(
            gate_c_cli,
            "build_probabilistic_prediction_records",
            side_effect=fake_build_records,
        ), patch.object(
            gate_c_cli,
            "_score_records",
            side_effect=fake_score_records,
        ):
            return selector(
                (),
                self.EXPECTED_BETAS,
                self.EXPECTED_THRESHOLDS,
                inventory,
                canonicalization=object(),
                profile=object(),
                run_id="gate-c-selector-test",
                git_commit="commit",
                dataset_sha256="dataset",
                config_sha256="config",
                fold=1,
                seed=42,
            )

    def test_mode_guard_and_frozen_grid_constants(self) -> None:
        self._assert_module()
        self.assertFalse(gate_c_cli.mode_evaluates_test("dev"))
        self.assertTrue(gate_c_cli.mode_evaluates_test("full"))
        self.assertEqual(
            gate_c_cli.requested_evaluation_splits("dev"),
            ("validation",),
        )
        self.assertEqual(
            gate_c_cli.requested_evaluation_splits("full"),
            ("validation", "test"),
        )
        with self.assertRaises(ValueError):
            gate_c_cli.mode_evaluates_test("smoke")
        self.assertEqual(gate_c_cli._VALID_MODES, ("dev", "full"))
        self.assertEqual(
            gate_c_cli._FROZEN_GATE_A_COMMIT,
            "b4033edbaf2150605c286a36e4b0564d75b0ac91",
        )
        self.assertEqual(gate_c_cli._EXPECTED_BETA_GRID, self.EXPECTED_BETAS)
        self.assertEqual(
            gate_c_cli._EXPECTED_THRESHOLD_GRID,
            self.EXPECTED_THRESHOLDS,
        )
        self.assertEqual(gate_c_cli._EPSILON, 1e-8)

    def test_selector_evaluates_all_75_rows_in_beta_major_order(self) -> None:
        result = self._selector({})
        grid = result["grid"]
        self.assertEqual(len(grid), 75)
        expected_order = [
            (beta, threshold)
            for beta in self.EXPECTED_BETAS
            for threshold in self.EXPECTED_THRESHOLDS
        ]
        self.assertEqual(
            [(row["beta"], row["threshold"]) for row in grid],
            expected_order,
        )
        self.assertEqual(result["best"]["beta"], 0.0)
        self.assertEqual(result["best"]["threshold"], 0.85)

    def test_selector_prioritizes_relation_f1_then_primary_entity_f1(self) -> None:
        relation_priority = self._selector(
            {
                (0.0, 0.85): (0.70, 0.99),
                (2.0, 0.99): (0.80, 0.10),
            }
        )
        self.assertEqual(
            (relation_priority["best"]["beta"], relation_priority["best"]["threshold"]),
            (2.0, 0.99),
        )

        entity_priority = self._selector(
            {
                (0.0, 0.85): (0.80, 0.60),
                (2.0, 0.99): (0.80, 0.70),
            }
        )
        self.assertEqual(
            (entity_priority["best"]["beta"], entity_priority["best"]["threshold"]),
            (2.0, 0.99),
        )

    def test_selector_breaks_metric_ties_by_lower_beta_then_threshold(self) -> None:
        result = self._selector(
            {
                (0.5, 0.85): (0.80, 0.70),
                (0.5, 0.99): (0.80, 0.70),
                (1.0, 0.85): (0.80, 0.70),
            }
        )
        self.assertEqual(
            (result["best"]["beta"], result["best"]["threshold"]),
            (0.5, 0.85),
        )


if __name__ == "__main__":
    unittest.main()
