from __future__ import annotations

import copy
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

    def _provenance_api(self):
        self._assert_module()
        api = getattr(gate_c_cli, "validate_gate_a_provenance", None)
        self.assertIsNotNone(api, "Gate C Gate-A provenance guard must exist")
        return api

    def _valid_provenance(self):
        dataset_sha = "a" * 64
        config_sha = "b" * 64
        checkpoint = {
            "model_state_dict": {},
            "git_commit": "b4033edbaf2150605c286a36e4b0564d75b0ac91",
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
        return checkpoint, run_metadata, dataset_sha, config_sha

    def _validate_provenance(self, checkpoint, run_metadata, dataset_sha, config_sha):
        api = self._provenance_api()
        return api(
            checkpoint,
            run_metadata,
            dataset_sha256=dataset_sha,
            config_sha256=config_sha,
            fold=1,
            seed=42,
            width_cap=8,
        )

    def _contract_api(self):
        self._assert_module()
        api = getattr(gate_c_cli, "validate_gate_c_frozen_contract", None)
        self.assertIsNotNone(api, "Gate C frozen config/profile guard must exist")
        return api

    def _valid_contract(self):
        base_config = SimpleNamespace(
            max_span_candidates=128,
            max_relation_token_distance=96,
        )
        training_config = SimpleNamespace(
            max_epochs=12,
            threshold_grid=self.EXPECTED_THRESHOLDS,
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
        return base_config, training_config, inventory, canonicalization, profile

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

    def test_parser_exposes_frozen_gate_c_cli_surface(self) -> None:
        self._assert_module()
        parser_factory = getattr(gate_c_cli, "_parser", None)
        self.assertIsNotNone(parser_factory, "Gate C CLI parser must exist")

        args = parser_factory().parse_args(
            [
                "--gate-a-checkpoint",
                "/tmp/best_model.pt",
                "--dataset",
                "/tmp/dataset.json",
                "--output-dir",
                "/tmp/gate-c-out",
            ]
        )

        self.assertEqual(args.mode, "dev")
        self.assertEqual(args.gate_a_checkpoint, "/tmp/best_model.pt")
        self.assertEqual(args.config, "journal/configs/gate_a_plain_spanpair.json")
        self.assertEqual(
            args.training_config,
            "journal/configs/gate_a_training_threshold_refine.json",
        )
        self.assertEqual(args.inventory, "journal/configs/gate_a_label_inventory.json")
        self.assertEqual(
            args.canonicalization,
            "journal/configs/stix/relation_canonicalization_v1.json",
        )
        self.assertEqual(
            args.profile,
            "journal/configs/stix/task_relationship_profile_v1.json",
        )
        self.assertEqual(
            args.manifest,
            "experiments/cv_manifest/run_partitions_seed_42.json",
        )
        self.assertEqual(args.dataset, "/tmp/dataset.json")
        self.assertEqual(args.fold, 1)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.output_dir, "/tmp/gate-c-out")

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

    def test_provenance_accepts_matching_frozen_gate_a_checkpoint(self) -> None:
        checkpoint, run_metadata, dataset_sha, config_sha = self._valid_provenance()
        self._validate_provenance(checkpoint, run_metadata, dataset_sha, config_sha)

    def test_provenance_rejects_checkpoint_lineage_hash_and_width_mismatch(self) -> None:
        checkpoint, run_metadata, dataset_sha, config_sha = self._valid_provenance()
        cases = (
            ("lineage", {"git_commit": "0" * 40}),
            ("checkpoint_dataset", {"dataset_sha256": "c" * 64}),
            ("checkpoint_config", {"config_sha256": "d" * 64}),
            ("width_cap", {"width_cap": 9}),
        )
        for name, checkpoint_patch in cases:
            with self.subTest(name=name):
                bad_checkpoint = copy.deepcopy(checkpoint)
                bad_checkpoint.update(checkpoint_patch)
                with self.assertRaises(ValueError):
                    self._validate_provenance(
                        bad_checkpoint,
                        run_metadata,
                        dataset_sha,
                        config_sha,
                    )

    def test_provenance_rejects_companion_fold_seed_and_hash_mismatch(self) -> None:
        checkpoint, run_metadata, dataset_sha, config_sha = self._valid_provenance()
        cases = (
            ("fold", {"fold": 2}),
            ("seed", {"seed": 123}),
            ("run_dataset", {"dataset_sha256": "c" * 64}),
            ("run_config", {"combined_config_sha256": "d" * 64}),
        )
        for name, metadata_patch in cases:
            with self.subTest(name=name):
                bad_metadata = copy.deepcopy(run_metadata)
                bad_metadata.update(metadata_patch)
                with self.assertRaises(ValueError):
                    self._validate_provenance(
                        checkpoint,
                        bad_metadata,
                        dataset_sha,
                        config_sha,
                    )

    def test_frozen_contract_accepts_and_reports_profile_hashes(self) -> None:
        api = self._contract_api()
        base, training, inventory, canonicalization, profile = self._valid_contract()
        canonicalization_sha = "c" * 64
        profile_sha = "d" * 64
        reported = api(
            base,
            training,
            inventory,
            canonicalization,
            profile,
            canonicalization_sha256=canonicalization_sha,
            task_profile_sha256=profile_sha,
        )
        self.assertEqual(
            reported,
            {
                "canonicalization_sha256": canonicalization_sha,
                "task_profile_sha256": profile_sha,
            },
        )

    def test_frozen_contract_rejects_model_and_training_drift(self) -> None:
        api = self._contract_api()
        base, training, inventory, canonicalization, profile = self._valid_contract()
        cases = (
            ("max_span_candidates", {"max_span_candidates": 127}, {}),
            ("max_relation_token_distance", {"max_relation_token_distance": 95}, {}),
            ("max_epochs", {}, {"max_epochs": 11}),
            ("threshold_grid", {}, {"threshold_grid": self.EXPECTED_THRESHOLDS[:-1]}),
        )
        for name, base_patch, training_patch in cases:
            with self.subTest(name=name):
                bad_base = copy.deepcopy(base)
                bad_training = copy.deepcopy(training)
                for field, value in base_patch.items():
                    setattr(bad_base, field, value)
                for field, value in training_patch.items():
                    setattr(bad_training, field, value)
                with self.assertRaises(ValueError):
                    api(
                        bad_base,
                        bad_training,
                        inventory,
                        canonicalization,
                        profile,
                        canonicalization_sha256="c" * 64,
                        task_profile_sha256="d" * 64,
                    )

    def test_frozen_contract_rejects_decoder_constant_drift(self) -> None:
        api = self._contract_api()
        base, training, inventory, canonicalization, profile = self._valid_contract()
        kwargs = dict(
            canonicalization_sha256="c" * 64,
            task_profile_sha256="d" * 64,
        )
        with patch.object(gate_c_cli, "_EXPECTED_BETA_GRID", (0.0, 1.0)):
            with self.assertRaises(ValueError):
                api(base, training, inventory, canonicalization, profile, **kwargs)
        with patch.object(gate_c_cli, "_EPSILON", 1e-6):
            with self.assertRaises(ValueError):
                api(base, training, inventory, canonicalization, profile, **kwargs)

    def test_frozen_contract_rejects_canonicalization_and_entity_coverage_drift(self) -> None:
        api = self._contract_api()
        base, training, inventory, canonicalization, profile = self._valid_contract()
        kwargs = dict(
            canonicalization_sha256="c" * 64,
            task_profile_sha256="d" * 64,
        )

        missing_relation = SimpleNamespace(
            by_project_label={"uses": object(), "targets": object()}
        )
        with self.assertRaises(ValueError):
            api(base, training, inventory, missing_relation, profile, **kwargs)

        missing_entity = SimpleNamespace(
            resolved_entity_types=frozenset({"intrusion-set", "identity"}),
            unresolved_entity_types=frozenset(),
        )
        with self.assertRaises(ValueError):
            api(base, training, inventory, canonicalization, missing_entity, **kwargs)

    def test_dev_run_never_materializes_or_writes_test(self) -> None:
        self._assert_module()
        run = getattr(gate_c_cli, "run", None)
        self.assertIsNotNone(run, "Gate C evaluator run() must exist")

        partition = SimpleNamespace(
            validation_document_ids=("val-doc",),
            test_document_ids=("test-doc",),
        )
        context = SimpleNamespace(
            windows=("all-window",),
            partition=partition,
        )
        materialized: list[tuple[str, ...]] = []
        written: list[str] = []

        def fake_windows_for_ids(windows, document_ids):
            ids = tuple(document_ids)
            materialized.append(ids)
            if ids == partition.test_document_ids:
                raise AssertionError("dev mode must not materialize test windows")
            self.assertEqual(ids, partition.validation_document_ids)
            return ("validation-window",)

        def fake_evaluate_validation(prepared, validation_windows):
            self.assertIs(prepared, context)
            self.assertEqual(validation_windows, ("validation-window",))
            return {
                "status": "completed",
                "mode": "dev",
                "test_evaluated": False,
                "artifacts": {
                    "validation_metrics.json": {"all_relation": {"f1": 0.5}},
                    "run_summary.json": {"test_evaluated": False},
                },
            }

        def fake_write_artifacts(output_dir, artifacts):
            for name in artifacts:
                if str(name).startswith("test_"):
                    raise AssertionError("dev mode must not write test artifacts")
                written.append(str(name))

        args = SimpleNamespace(mode="dev", output_dir="/tmp/gate-c-firewall-test")
        with patch.object(
            gate_c_cli,
            "_prepare_evaluation_context",
            return_value=context,
            create=True,
        ), patch.object(
            gate_c_cli,
            "_windows_for_ids",
            side_effect=fake_windows_for_ids,
            create=True,
        ), patch.object(
            gate_c_cli,
            "_evaluate_validation",
            side_effect=fake_evaluate_validation,
            create=True,
        ), patch.object(
            gate_c_cli,
            "_evaluate_test",
            side_effect=AssertionError("dev mode must not evaluate test"),
            create=True,
        ), patch.object(
            gate_c_cli,
            "_write_artifacts",
            side_effect=fake_write_artifacts,
            create=True,
        ):
            result = run(args)

        self.assertEqual(materialized, [partition.validation_document_ids])
        self.assertFalse(result["test_evaluated"])
        self.assertTrue(written)
        self.assertFalse(any(name.startswith("test_") for name in written))


if __name__ == "__main__":
    unittest.main()
