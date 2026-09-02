from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from journal.scsp.config import GateAConfig
from journal.scsp.splits import FoldPartition, assert_disjoint_partition, load_fold_partition


class GateAConfigSplitTests(unittest.TestCase):
    def test_config_is_immutable_and_explicitly_disables_schema(self) -> None:
        payload = {
            "experiment_name": "gate-a-test",
            "schema_mode": "none",
            "encoder_model": "FacebookAI/roberta-base",
            "encoder_revision": "unit-test-revision",
            "split_seed": 11800,
            "seed": 42,
            "fold": 1,
            "span_width_coverage": 0.995,
            "max_span_candidates": 128,
            "min_entity_score": 0.05,
            "max_relation_token_distance": 96,
            "relation_negative_ratio": 6,
            "output_dir": "journal/runs/gate_a"
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = GateAConfig.from_json(path)

        self.assertEqual(config.schema_mode, "none")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.fold, 1)
        with self.assertRaises(FrozenInstanceError):
            config.seed = 123  # type: ignore[misc]

    def test_config_rejects_schema_enabled_gate_a(self) -> None:
        payload = {
            "experiment_name": "bad",
            "schema_mode": "hard_profile",
            "encoder_model": "FacebookAI/roberta-base",
            "encoder_revision": "unit-test-revision",
            "split_seed": 11800,
            "seed": 42,
            "fold": 1,
            "span_width_coverage": 0.995,
            "max_span_candidates": 128,
            "min_entity_score": 0.05,
            "max_relation_token_distance": 96,
            "relation_negative_ratio": 6,
            "output_dir": "journal/runs/gate_a"
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_mode"):
                GateAConfig.from_json(path)

    def test_manifest_loader_preserves_document_ids(self) -> None:
        manifest = {
            "split_seed": 11800,
            "run_seed": 42,
            "dataset_sha256": "abc",
            "outer_folds": [{
                "fold": 1,
                "train_document_ids": ["a", "b"],
                "validation_document_ids": ["c"],
                "test_document_ids": ["d"],
            }],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            partition = load_fold_partition(path, 1)

        self.assertEqual(partition.train_document_ids, ("a", "b"))
        self.assertEqual(partition.validation_document_ids, ("c",))
        self.assertEqual(partition.test_document_ids, ("d",))
        self.assertEqual(partition.dataset_sha256, "abc")
        self.assertEqual(partition.split_seed, 11800)

    def test_overlap_is_rejected(self) -> None:
        partition = FoldPartition(
            fold=1,
            split_seed=11800,
            run_seed=42,
            dataset_sha256="abc",
            train_document_ids=("a", "b"),
            validation_document_ids=("b", "c"),
            test_document_ids=("d",),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_partition(partition)


if __name__ == "__main__":
    unittest.main()
