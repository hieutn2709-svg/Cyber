"""Static preflight checks for reproducibility and provenance claims."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_TYPES = [
    "attack-pattern",
    "campaign",
    "domain-name",
    "identity",
    "indicator",
    "intrusion-set",
    "location",
    "malware",
    "tool",
    "vulnerability",
]
EXPECTED_RESULTS = {
    "same-split-rule-kb-baseline": (0.289, 0.011),
    "V10-contextual-neural": (0.698, 0.219),
    "V13-conservative-hybrid": (0.710, 0.222),
}
EXPECTED_RUN_FOLDS = [
    ["405", "387", "413", "417", "402", "23", "414", "415", "425", "401", "410"],
    ["420", "423", "406", "407", "390", "388", "424", "376", "371", "396", "378"],
    ["370", "386", "372", "419", "421", "399", "395", "412", "403", "389"],
    ["377", "374", "381", "422", "409", "397", "398", "400", "418", "416"],
    ["426", "373", "393", "392", "411", "408", "375", "391", "366", "365"],
]
EXPECTED_RAW_RELATIONS = [134, 106, 138, 111, 85]
EXPECTED_EVALUABLE_RELATIONS = [128, 106, 137, 110, 83]


class ReproducibilityPreflight(unittest.TestCase):
    def test_reviewer_configuration_is_canonical(self) -> None:
        config = json.loads(
            (ROOT / "Configs" / "reviewer_experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["encoder"], "roberta-base")
        self.assertEqual(config["bigru_hidden_size"], 250)
        self.assertEqual(config["bigru_layers"], 2)
        self.assertEqual(config["attention_heads"], 8)
        self.assertEqual(config["dropout"], 0.35)
        self.assertEqual(config["encoder_learning_rate"], 3e-5)
        self.assertEqual(config["head_learning_rate"], 3e-4)
        self.assertEqual(config["weight_decay"], 0.01)
        self.assertEqual(config["joint_max_epochs"], 12)
        self.assertEqual(config["joint_patience"], 5)
        self.assertEqual(config["relation_refinement_epochs"], 12)
        self.assertEqual(config["relation_refinement_patience"], 4)
        self.assertEqual(config.get("max_seq_length"), 512)
        self.assertEqual(config.get("physical_batch_size"), 2)
        self.assertEqual(config.get("gradient_accumulation_steps"), 2)
        self.assertEqual(config.get("emission_ce_weight"), 1.5)
        self.assertEqual(config.get("o_class_weight_cap"), 0.25)
        self.assertEqual(config.get("max_relation_distance"), 96)
        self.assertEqual(config.get("run_seed"), 42)
        self.assertEqual(
            config.get("task_loss_weights"),
            {"entity": 0.80, "role": 0.05, "relation": 0.15},
        )
        self.assertEqual(config["selection_data"], "validation_only")

    def test_tag_and_role_spaces_are_separate(self) -> None:
        module_path = ROOT / "Joint_model" / "tagging_scheme.py"
        spec = importlib.util.spec_from_file_location("tagging_scheme", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scheme = module.BIEOSScheme(ROOT / "Configs" / "stix_mapping.json")
        self.assertEqual(scheme.get_num_labels(), 61)
        self.assertEqual(scheme.get_num_role_labels(), 4)
        self.assertEqual(
            list(scheme.role_to_id), ["O", "ROLE_1", "ROLE_2", "ROLE_BOTH"]
        )
        self.assertTrue(all("ROLE_" not in tag and not re.search(r"_[12]$", tag)
                            for tag in scheme.tag_to_id))

    def test_model_exposes_token_role_head_not_combined_relation_labels(self) -> None:
        source = (ROOT / "Joint_model" / "joint_model.py").read_text(encoding="utf-8")
        self.assertIn("role_classifier", source)
        self.assertIn("num_roles", source)
        self.assertNotIn("rel_classifier", source)
        self.assertNotIn("num_rel_types", source)

    def test_inference_requires_a_provenanced_checkpoint(self) -> None:
        source = (ROOT / "Joint_model" / "inference.py").read_text(encoding="utf-8")
        self.assertIn("model_weights=None", source)
        self.assertIn("raise FileNotFoundError", source)
        self.assertNotIn("best_model (10).pt", source)
        self.assertNotIn('"confidence": 0.95', source)
        self.assertNotIn('"confidence": 0.85', source)

    def test_official_document_manifest(self) -> None:
        manifest_path = ROOT / "experiments" / "cv_manifest" / "document_folds.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["split_seed"], 11800)
        self.assertEqual(manifest.get("run_seed"), 42)
        self.assertEqual(manifest["core_entity_types"], CORE_TYPES)
        self.assertEqual([len(fold["documents"]) for fold in manifest["folds"]],
                         [11, 11, 10, 10, 10])
        self.assertEqual(
            [fold["documents"] for fold in manifest["folds"]], EXPECTED_RUN_FOLDS
        )
        with (ROOT / "experiments" / "cv_manifest" / "fold_summary.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            fold_summary = list(csv.DictReader(stream))
        self.assertEqual(
            [int(row["relation_instances"]) for row in fold_summary],
            EXPECTED_RAW_RELATIONS,
        )
        self.assertEqual(
            [int(row["evaluation_relation_instances"]) for row in fold_summary],
            EXPECTED_EVALUABLE_RELATIONS,
        )
        document_ids = {document["document_id"] for document in manifest["documents"]}
        held_out_ids: set[str] = set()
        for fold in range(5):
            split = json.loads(
                (ROOT / "experiments" / "cv_manifest" / "outer_splits"
                 / f"outer_fold_{fold}.json").read_text(encoding="utf-8")
            )
            train = set(split["train_document_ids"])
            validation = set(split["validation_document_ids"])
            test = set(split["test_document_ids"])
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(train | validation | test, document_ids)
            held_out_ids.update(test)
        self.assertEqual(held_out_ids, document_ids)

    def test_archived_run_manifest_maps_to_stable_document_ids(self) -> None:
        annotations = json.loads(
            (ROOT / "data" / "Annotations.json").read_text(encoding="utf-8-sig")
        )
        document_ids_by_index = [str(task["id"]) for task in annotations]
        run_manifest_path = (
            ROOT / "experiments" / "cv_manifest" / "fold_manifest_v4.json"
        )
        self.assertTrue(run_manifest_path.is_file())
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        mapped = [
            [document_ids_by_index[int(index)] for index in fold]
            for fold in run_manifest["folds"]
        ]
        self.assertEqual(run_manifest["split_seed"], 11800)
        self.assertEqual(mapped, EXPECTED_RUN_FOLDS)

        partitions_path = (
            ROOT / "experiments" / "cv_manifest" / "run_partitions_seed_42.json"
        )
        self.assertTrue(partitions_path.is_file())
        partitions = json.loads(partitions_path.read_text(encoding="utf-8"))
        self.assertEqual(partitions["run_seed"], 42)
        self.assertEqual(
            partitions["dataset_sha256"],
            "190d3136edba33d89ee58f533e2d12cc6cac2842323e3168f6e3e0e71af72c48",
        )
        self.assertEqual(
            [fold["test_document_ids"] for fold in partitions["outer_folds"]],
            EXPECTED_RUN_FOLDS,
        )
        self.assertEqual(
            [fold["evaluation_relation_instances"] for fold in partitions["outer_folds"]],
            EXPECTED_EVALUABLE_RELATIONS,
        )

    def test_manifest_builder_check_mode_matches_checked_in_files(self) -> None:
        result = subprocess.run(
            [sys.executable, "experiments/build_document_cv.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "matches archived run manifest and checked-in derived files",
            result.stdout,
        )

    def test_current_results_contain_only_supported_primary_claims(self) -> None:
        path = ROOT / "experiments" / "results" / "current_results.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        observed = {
            row["configuration"]: (float(row["entity_f1"]), float(row["relation_f1"]))
            for row in rows
        }
        self.assertEqual(observed, EXPECTED_RESULTS)
        self.assertTrue(all(row["selection_scope"] == "validation-only" for row in rows))

    def test_fold_level_evidence_is_complete_and_consistent(self) -> None:
        results_dir = ROOT / "experiments" / "results"
        required = [
            "rule_kb_baseline_per_fold.csv",
            "contextual_neural_per_fold.csv",
            "naive_hybrid_per_fold.csv",
            "per_type_calibrated_hybrid_per_fold.csv",
            "five_fold_results_conservative_hybrid.csv",
            "five_fold_summary.csv",
            "gold_span_diagnostic_per_fold.csv",
            "gold_pair_type_accuracy_per_fold.csv",
            "per_class/entity_per_class_by_fold.csv",
            "per_class/relation_per_class_by_fold.csv",
        ]
        for relative_path in required:
            self.assertTrue((results_dir / relative_path).is_file(), relative_path)
        self.assertTrue((ROOT / "corpus_label_stats.csv").is_file())

        with (results_dir / "five_fold_results_conservative_hybrid.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            v13_rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["fold"]) for row in v13_rows], [1, 2, 3, 4, 5])
        self.assertEqual(
            [int(row["entity_tp"]) + int(row["entity_fn"]) for row in v13_rows],
            [287, 262, 275, 250, 279],
        )
        self.assertEqual(
            [int(row["relation_tp"]) + int(row["relation_fn"]) for row in v13_rows],
            EXPECTED_EVALUABLE_RELATIONS,
        )

        for filename, prefix in [
            ("entity_per_class_by_fold.csv", "entity"),
            ("relation_per_class_by_fold.csv", "relation"),
        ]:
            with (results_dir / "per_class" / filename).open(
                newline="", encoding="utf-8"
            ) as stream:
                per_class_rows = list(csv.DictReader(stream))
            for fold_row in v13_rows:
                fold = int(fold_row["fold"])
                class_rows = [
                    row for row in per_class_rows if int(row["fold"]) == fold
                ]
                for count_name in ("tp", "fp", "fn"):
                    self.assertEqual(
                        sum(int(row[count_name]) for row in class_rows),
                        int(fold_row[f"{prefix}_{count_name}"]),
                    )

        prediction_root = results_dir / "predictions" / "conservative_hybrid"
        for fold, expected_windows in enumerate([16, 11, 10, 11, 19], start=1):
            prediction_path = (
                prediction_root
                / f"fold_{fold}_seed_42"
                / "test_predictions.jsonl"
            )
            self.assertTrue(prediction_path.is_file())
            rows = [
                json.loads(line)
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), expected_windows)

    def test_variant_lineage_preserves_v10_through_v13(self) -> None:
        path = ROOT / "experiments" / "results" / "variant_provenance.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["variant"] for row in rows], ["V10", "V11", "V12", "V13"])
        self.assertEqual([row["reporting_status"] for row in rows],
                         ["primary", "development-diagnostic",
                          "development-diagnostic", "primary"])
        self.assertTrue(all(row["selection_scope"] == "validation-only" for row in rows))

    def test_unsupported_artifacts_are_archived(self) -> None:
        self.assertFalse((ROOT / "experiments" / "cv_results.csv").exists())
        self.assertFalse((ROOT / "experiments" / "ultimate_run_fixed").exists())
        self.assertFalse((ROOT / "experiments" / "aggregate_cv_results.py").exists())
        self.assertFalse((ROOT / "experiments" / "cv_results_template.csv").exists())
        self.assertFalse((ROOT / "Joint_model" / "train_model.ipynb").exists())
        archive = ROOT / "archive" / "deprecated"
        self.assertTrue((archive / "README.md").is_file())
        self.assertTrue((archive / "unsupported_cv_results.csv").is_file())
        self.assertTrue((archive / "unsupported_ultimate_run_fixed").is_dir())
        self.assertTrue((archive / "legacy_train_model.ipynb").is_file())
        self.assertTrue((archive / "legacy_aggregate_cv_results.py").is_file())
        self.assertTrue((archive / "legacy_cv_results_template.csv").is_file())

    def test_active_claims_exclude_legacy_metrics_and_label_counts(self) -> None:
        active_roots = [ROOT / "README.md", ROOT / "Configs", ROOT / "experiments"]
        violations: list[str] = []
        for active_root in active_roots:
            paths = [active_root] if active_root.is_file() else active_root.rglob("*")
            for path in paths:
                if not path.is_file() or "archive" in path.parts:
                    continue
                if path.suffix.lower() not in {".md", ".tex", ".csv", ".json", ".yaml", ".yml"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if re.search(r"\b(?:0\.922|0\.927|0\.763)\b", text):
                    violations.append(str(path.relative_to(ROOT)))
                if re.search(r"\b(?:81|82|146)[ -](?:label|tag)", text, re.IGNORECASE):
                    violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_literature_and_reproduction_are_explicitly_distinguished(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("literature-reported", readme)
        self.assertIn("same-split reproduction", readme)
        self.assertIn("0.916", readme)
        self.assertIn("0.724", readme)
        self.assertIn("0.289", readme)
        self.assertIn("0.011", readme)

    def test_readme_links_large_artifacts_and_explains_fold_counts(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Large artifacts", readme)
        self.assertIn(
            "https://drive.google.com/drive/folders/1karEftjWrFmWq7gdjLwD0BhVnOL44caY",
            readme,
        )
        self.assertIn("Anyone with the link — Viewer", readme)
        self.assertIn("134, 106, 138, 111, 85", readme)
        self.assertIn("128, 106, 137, 110, 83", readme)
        self.assertIn("61 sequence tags", readme)
        self.assertIn("separate four-class token head", readme)

    def test_readme_explains_old_and_run_fold_numbering(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        normalized = " ".join(readme.split())
        self.assertIn("same 52-document universe", normalized)
        self.assertIn(
            "no old fold has the same document set as any run fold", normalized
        )
        self.assertIn(
            "V13 fold `k` (1--5) is `fold_manifest_v4.json[\"folds\"][k-1]`",
            normalized,
        )
        self.assertIn("`outer_splits/outer_fold_{k-1}.json`", readme)

    def test_license_and_notice_are_present(self) -> None:
        self.assertTrue((ROOT / "LICENSE").is_file())
        notice_path = ROOT / "NOTICE.md"
        self.assertTrue(notice_path.is_file())
        notice = notice_path.read_text(encoding="utf-8")
        self.assertIn("© 2026 The MITRE Corporation", notice)
        self.assertIn("https://attack.mitre.org/resources/terms-of-use/", notice)


if __name__ == "__main__":
    unittest.main()
