from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from journal.scsp.structures import GoldSpan, SpanCandidate


@unittest.skipIf(torch is None, "PyTorch not installed")
class TrainingConfigUtilsTests(unittest.TestCase):
    def test_training_config_loads_and_rejects_unknown_fields(self) -> None:
        from journal.scsp.training_config import GateATrainingConfig

        payload = {
            "encoder_lr": 2e-5,
            "head_lr": 2e-4,
            "max_epochs": 12,
            "smoke_epochs": 2,
            "early_stopping_patience": 4,
            "gradient_accumulation_steps": 4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "width_embedding_dim": 32,
            "context_dim": 128,
            "distance_embedding_dim": 32,
            "entity_negative_ratio": 6,
            "relation_existence_pos_weight": 6.0,
            "focal_gamma": 1.0,
            "entity_loss_weight": 1.0,
            "relation_existence_loss_weight": 1.0,
            "relation_type_loss_weight": 1.0,
            "relation_inference_chunk_size": 512,
            "fp16": True,
            "threshold_grid": [0.25, 0.5, 0.75],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "training.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = GateATrainingConfig.from_json(path)
            self.assertEqual(cfg.max_epochs, 12)
            self.assertEqual(cfg.threshold_grid, (0.25, 0.5, 0.75))
            payload["test_threshold"] = 0.5
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                GateATrainingConfig.from_json(path)

    def test_entity_training_sampler_keeps_all_positives_and_is_deterministic(self) -> None:
        from journal.scsp.training_utils import sample_entity_training_indices

        candidates = tuple(SpanCandidate("d", i, i) for i in range(10))
        gold = (
            GoldSpan("d", 1, 1, "malware"),
            GoldSpan("d", 7, 7, "tool"),
        )
        first = sample_entity_training_indices(
            candidates, gold, negative_ratio=2, seed=5
        )
        second = sample_entity_training_indices(
            candidates, gold, negative_ratio=2, seed=5
        )
        self.assertEqual(first, second)
        self.assertIn(1, first)
        self.assertIn(7, first)
        self.assertLessEqual(len(first), 6)

    def test_entity_targets_use_none_zero_and_inventory_order(self) -> None:
        from journal.scsp.training_utils import build_entity_targets

        candidates = (
            SpanCandidate("d", 1, 1),
            SpanCandidate("d", 2, 2),
            SpanCandidate("d", 3, 3),
        )
        gold = (
            GoldSpan("d", 1, 1, "malware"),
            GoldSpan("d", 3, 3, "tool"),
        )
        labels = build_entity_targets(candidates, gold, ("malware", "tool"))
        self.assertEqual(labels, (1, 0, 2))

    def test_scored_candidates_take_best_non_none_class(self) -> None:
        from journal.scsp.training_utils import scored_entity_candidates

        candidates = (
            SpanCandidate("d", 1, 1),
            SpanCandidate("d", 2, 2),
        )
        logits = torch.tensor(
            [
                [4.0, 1.0, 3.0],
                [0.0, 2.0, 1.0],
            ]
        )
        probabilities = torch.softmax(logits, dim=-1)
        scored = scored_entity_candidates(
            candidates, logits, ("malware", "tool")
        )
        self.assertEqual(scored[0].label, "tool")
        self.assertEqual(scored[1].label, "malware")
        self.assertAlmostEqual(
            scored[0].entity_score,
            float(1.0 - probabilities[0, 0]),
            places=6,
        )
        self.assertAlmostEqual(
            scored[1].entity_score,
            float(1.0 - probabilities[1, 0]),
            places=6,
        )
        self.assertGreater(scored[1].entity_score, scored[0].entity_score)


if __name__ == "__main__":
    unittest.main()
