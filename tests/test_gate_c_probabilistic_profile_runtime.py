from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileRuntimeTests(unittest.TestCase):
    def test_validation_runtime_uses_gate_c_inference_and_selected_decoder(self) -> None:
        evaluate = getattr(gate_c_cli, "_evaluate_validation", None)
        self.assertIsNotNone(evaluate, "Gate C validation runtime helper must exist")

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
            run_id="gate-c-runtime-test",
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

        with patch.object(
            gate_c_cli,
            "infer_gate_c_split",
            return_value=inferences,
            create=True,
        ) as infer_mock, patch.object(
            gate_c_cli,
            "_select_probabilistic_decoder",
            return_value=selection,
        ) as selector_mock, patch.object(
            gate_c_cli,
            "build_probabilistic_prediction_records",
            return_value=records,
        ) as records_mock, patch.object(
            gate_c_cli,
            "_score_records",
            return_value=metrics,
        ) as score_mock:
            result = evaluate(prepared, validation_windows)

        infer_mock.assert_called_once_with(
            prepared.model,
            validation_windows,
            prepared.inventory,
            width_cap=prepared.width_cap,
            base_config=prepared.base_config,
            training_config=prepared.training_config,
            device=prepared.device,
        )
        selector_mock.assert_called_once_with(
            inferences,
            gate_c_cli._EXPECTED_BETA_GRID,
            gate_c_cli._EXPECTED_THRESHOLD_GRID,
            prepared.inventory,
            canonicalization=prepared.canonicalization,
            profile=prepared.profile,
            run_id=prepared.run_id,
            git_commit=prepared.git_commit,
            dataset_sha256=prepared.dataset_sha256,
            config_sha256=prepared.config_sha256,
            fold=prepared.fold,
            seed=prepared.seed,
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
            split="validation",
            epsilon=gate_c_cli._EPSILON,
        )
        score_mock.assert_called_once_with(records, prepared.inventory)

        self.assertEqual(result["beta"], 0.25)
        self.assertEqual(result["relation_threshold"], 0.96)
        self.assertEqual(result["validation"], selection["best"])
        self.assertEqual(result["validation_metrics"], metrics)
        self.assertFalse(result["test_evaluated"])
        self.assertEqual(
            result["artifacts"],
            {
                "validation_decoder_selection.json": selection,
                "validation_metrics.json": metrics,
            },
        )


if __name__ == "__main__":
    unittest.main()
