from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileTestRuntimeTests(unittest.TestCase):
    def test_test_runtime_reuses_validation_decoder_without_reselection(self) -> None:
        evaluate = getattr(gate_c_cli, "_evaluate_test", None)
        self.assertIsNotNone(evaluate, "Gate C frozen test runtime helper must exist")

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
            run_id="gate-c-test-runtime",
            git_commit="commit",
            dataset_sha256="dataset-sha",
            config_sha256="config-sha",
            fold=1,
            seed=42,
        )
        test_windows = ("test-window",)
        validation_result = {
            "beta": 0.25,
            "relation_threshold": 0.96,
            "validation": {
                "beta": 0.25,
                "threshold": 0.96,
                "relation_f1": 0.55,
                "primary_entity_f1": 0.66,
            },
        }
        inferences = ("test-inference",)
        records = ("test-record",)
        metrics = {
            "all_relation": {"f1": 0.44},
            "primary_entity": {"f1": 0.61},
        }

        with patch.object(
            gate_c_cli,
            "infer_gate_c_split",
            return_value=inferences,
            create=True,
        ) as infer_mock, patch.object(
            gate_c_cli,
            "_select_probabilistic_decoder",
            side_effect=AssertionError("test split must never reselect beta/threshold"),
        ), patch.object(
            gate_c_cli,
            "build_probabilistic_prediction_records",
            return_value=records,
        ) as records_mock, patch.object(
            gate_c_cli,
            "_score_records",
            return_value=metrics,
        ) as score_mock:
            result = evaluate(prepared, test_windows, validation_result)

        infer_mock.assert_called_once_with(
            prepared.model,
            test_windows,
            prepared.inventory,
            width_cap=prepared.width_cap,
            base_config=prepared.base_config,
            training_config=prepared.training_config,
            device=prepared.device,
        )
        records_mock.assert_called_once_with(
            inferences,
            prepared.inventory.relation_types,
            beta=0.25,
            threshold=0.96,
            canonicalization=prepared.canonicalization,
            profile=prepared.profile,
            run_id=prepared.run_id,
            git_commit=prepared.git_commit,
            dataset_sha256=prepared.dataset_sha256,
            config_sha256=prepared.config_sha256,
            fold=prepared.fold,
            seed=prepared.seed,
            split="test",
            epsilon=gate_c_cli._EPSILON,
        )
        score_mock.assert_called_once_with(records, prepared.inventory)

        self.assertTrue(result["test_evaluated"])
        self.assertEqual(result["mode"], "full")
        self.assertEqual(result["test_metrics"], metrics)
        self.assertEqual(
            result["artifacts"],
            {"test_metrics.json": metrics},
        )


if __name__ == "__main__":
    unittest.main()
