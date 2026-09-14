from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

    def test_prepare_context_loads_frozen_checkpoint_and_uses_train_only_for_width(self) -> None:
        prepare = getattr(gate_c_cli, "_prepare_evaluation_context", None)
        self.assertIsNotNone(prepare, "Gate C runtime preparation helper must exist")

        dataset_sha = "a" * 64
        config_sha = "b" * 64
        canonicalization_sha = "c" * 64
        profile_sha = "d" * 64
        checkpoint_sha = "e" * 64
        commit = "f" * 40

        base_config = SimpleNamespace(
            seed=42,
            span_width_coverage=0.995,
            max_span_candidates=128,
            max_relation_token_distance=96,
        )
        training_config = SimpleNamespace(
            max_epochs=12,
            threshold_grid=gate_c_cli._EXPECTED_THRESHOLD_GRID,
        )
        inventory = SimpleNamespace(
            primary_entity_types=("intrusion-set", "identity"),
            auxiliary_entity_types=("file-paths",),
            trainable_entity_types=("intrusion-set", "identity", "file-paths"),
            relation_types=("uses", "targets", "used-in"),
        )
        canonicalization = SimpleNamespace(
            by_project_label={
                "uses": object(),
                "targets": object(),
                "used-in": object(),
            }
        )
        profile = SimpleNamespace(
            resolved_entity_types=frozenset({"intrusion-set", "identity"}),
            unresolved_entity_types=frozenset({"file-paths"}),
        )
        partition = SimpleNamespace(
            train_document_ids=("train-doc",),
            validation_document_ids=("val-doc",),
            test_document_ids=("test-doc",),
        )
        windows = ("all-window",)
        train_windows = ("train-window",)
        preflight = {"status": "ok"}
        checkpoint = {
            "model_state_dict": {"weight": "frozen"},
            "git_commit": gate_c_cli._FROZEN_GATE_A_COMMIT,
            "dataset_sha256": dataset_sha,
            "config_sha256": config_sha,
            "width_cap": 8,
        }
        run_metadata = {
            "fold": 1,
            "seed": 42,
            "dataset_sha256": dataset_sha,
            "combined_config_sha256": config_sha,
        }
        model = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "best_model.pt"
            checkpoint_path.write_bytes(b"checkpoint-placeholder")
            (root / "training_run_config.json").write_text(
                json.dumps(run_metadata),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                mode="dev",
                gate_a_checkpoint=str(checkpoint_path),
                config="gate_a.json",
                training_config="training.json",
                inventory="inventory.json",
                canonicalization="canonicalization.json",
                profile="profile.json",
                manifest="manifest.json",
                dataset=str(root / "dataset.json"),
                fold=1,
                seed=42,
                device="cpu",
                output_dir=str(root / "out"),
            )

            def fake_sha256(path):
                name = Path(path).name
                values = {
                    "dataset.json": dataset_sha,
                    "canonicalization.json": canonicalization_sha,
                    "profile.json": profile_sha,
                    "best_model.pt": checkpoint_sha,
                }
                return values[name]

            fake_torch = SimpleNamespace(
                load=MagicMock(return_value=checkpoint),
            )
            gate_a_config_loader = SimpleNamespace(
                from_json=MagicMock(return_value=base_config)
            )
            training_config_loader = SimpleNamespace(
                from_json=MagicMock(return_value=training_config)
            )
            inventory_loader = SimpleNamespace(
                from_json=MagicMock(return_value=inventory)
            )

            with patch.object(
                gate_c_cli, "GateAConfig", gate_a_config_loader, create=True
            ), patch.object(
                gate_c_cli,
                "GateATrainingConfig",
                training_config_loader,
                create=True,
            ), patch.object(
                gate_c_cli, "LabelInventory", inventory_loader, create=True
            ), patch.object(
                gate_c_cli,
                "load_relation_canonicalization",
                return_value=canonicalization,
                create=True,
            ), patch.object(
                gate_c_cli,
                "load_task_relationship_profile",
                return_value=profile,
                create=True,
            ), patch.object(
                gate_c_cli,
                "build_preflight",
                return_value=preflight,
                create=True,
            ) as preflight_mock, patch.object(
                gate_c_cli,
                "load_fold_partition",
                return_value=partition,
                create=True,
            ), patch.object(
                gate_c_cli,
                "load_clean_windows",
                return_value=windows,
                create=True,
            ), patch.object(
                gate_c_cli,
                "_windows_for_ids",
                return_value=train_windows,
                create=True,
            ) as windows_mock, patch.object(
                gate_c_cli,
                "_derive_width_cap",
                return_value=8,
                create=True,
            ) as width_mock, patch.object(
                gate_c_cli,
                "_sha256_file",
                side_effect=fake_sha256,
                create=True,
            ), patch.object(
                gate_c_cli,
                "_combined_config_hash",
                return_value=config_sha,
                create=True,
            ), patch.object(
                gate_c_cli,
                "_set_seed",
                create=True,
            ) as seed_mock, patch.object(
                gate_c_cli,
                "torch",
                fake_torch,
                create=True,
            ), patch.object(
                gate_c_cli,
                "_resolve_device",
                return_value="cpu",
                create=True,
            ), patch.object(
                gate_c_cli,
                "_make_model",
                return_value=model,
                create=True,
            ), patch.object(
                gate_c_cli,
                "_git_commit",
                return_value=commit,
                create=True,
            ), patch.object(
                gate_c_cli,
                "validate_gate_c_frozen_contract",
                return_value={
                    "canonicalization_sha256": canonicalization_sha,
                    "task_profile_sha256": profile_sha,
                },
            ) as contract_mock, patch.object(
                gate_c_cli,
                "validate_gate_a_provenance",
            ) as provenance_mock:
                context = prepare(args)

        preflight_mock.assert_called_once()
        windows_mock.assert_called_once_with(windows, partition.train_document_ids)
        width_mock.assert_called_once_with(train_windows, base_config.span_width_coverage)
        seed_mock.assert_called_once_with(42)
        contract_mock.assert_called_once_with(
            base_config,
            training_config,
            inventory,
            canonicalization,
            profile,
            canonicalization_sha256=canonicalization_sha,
            task_profile_sha256=profile_sha,
        )
        provenance_mock.assert_called_once_with(
            checkpoint,
            run_metadata,
            dataset_sha256=dataset_sha,
            config_sha256=config_sha,
            fold=1,
            seed=42,
            width_cap=8,
        )
        model.load_state_dict.assert_called_once_with(checkpoint["model_state_dict"])
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()

        self.assertIs(context.base_config, base_config)
        self.assertIs(context.training_config, training_config)
        self.assertIs(context.inventory, inventory)
        self.assertIs(context.canonicalization, canonicalization)
        self.assertIs(context.profile, profile)
        self.assertIs(context.partition, partition)
        self.assertIs(context.windows, windows)
        self.assertIs(context.model, model)
        self.assertEqual(context.width_cap, 8)
        self.assertEqual(context.device, "cpu")
        self.assertEqual(context.fold, 1)
        self.assertEqual(context.seed, 42)
        self.assertEqual(context.dataset_sha256, dataset_sha)
        self.assertEqual(context.config_sha256, config_sha)
        self.assertEqual(context.canonicalization_sha256, canonicalization_sha)
        self.assertEqual(context.task_profile_sha256, profile_sha)
        self.assertEqual(context.gate_a_checkpoint_sha256, checkpoint_sha)
        self.assertEqual(context.git_commit, commit)
        self.assertIs(context.preflight, preflight)
        self.assertIs(context.checkpoint, checkpoint)
        self.assertEqual(context.run_metadata, run_metadata)
        self.assertIn("gate-c", context.run_id)
        self.assertIn(commit[:8], context.run_id)


if __name__ == "__main__":
    unittest.main()
