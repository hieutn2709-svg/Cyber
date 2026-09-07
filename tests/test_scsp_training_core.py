from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from journal.scsp.data import LabelInventory, WindowExample
from journal.scsp.pairs import GoldRelation
from journal.scsp.runtime_model import GateASpanPairModel
from journal.scsp.structures import GoldSpan


@unittest.skipIf(torch is None, "PyTorch not installed")
class TrainingCoreTests(unittest.TestCase):
    class DummyEncoder(nn.Module):
        def __init__(self, hidden_size: int = 8) -> None:
            super().__init__()
            self.embedding = nn.Embedding(100, hidden_size)
            self.config = SimpleNamespace(hidden_size=hidden_size)

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class BaseConfig:
        max_span_candidates = 8
        min_entity_score = 0.0
        max_relation_token_distance = 2
        relation_negative_ratio = 2

    class TrainConfig:
        entity_negative_ratio = 2
        relation_existence_pos_weight = 2.0
        focal_gamma = 1.0
        entity_loss_weight = 1.0
        relation_existence_loss_weight = 1.0
        relation_type_loss_weight = 1.0
        relation_inference_chunk_size = 4

    def _inventory(self):
        return LabelInventory.from_dict(
            {
                "primary_entity_types": ["intrusion-set", "malware"],
                "auxiliary_entity_types": ["tactic"],
                "relation_types": ["uses", "targets"],
            }
        )

    def _window(self):
        a = GoldSpan("d", 1, 1, "intrusion-set")
        b = GoldSpan("d", 5, 5, "malware")
        return WindowExample(
            doc_seq_index=0,
            doc_id="d",
            window_index=0,
            token_start_global=0,
            token_end_global=6,
            input_ids=(0, 10, 11, 12, 13, 14, 2),
            attention_mask=(1, 1, 1, 1, 1, 1, 1),
            label_mask=(False, True, True, True, True, True, False),
            content_start=1,
            content_end=5,
            gold_spans=(a, b),
            gold_relations=(GoldRelation(a, b, "uses"),),
            primary_entity_types=("intrusion-set", "malware"),
            auxiliary_entity_types=("tactic",),
        )

    def _model(self):
        return GateASpanPairModel(
            encoder=self.DummyEncoder(8),
            hidden_size=8,
            num_entity_classes=4,
            num_relation_types=2,
            max_width=2,
            width_embedding_dim=4,
            context_dim=5,
            distance_embedding_dim=3,
            max_distance=2,
        )

    def test_training_window_loss_is_finite_and_keeps_distant_gold_positive(self):
        from journal.scsp.training import compute_training_window_loss

        model = self._model()
        result = compute_training_window_loss(
            model,
            self._window(),
            self._inventory(),
            width_cap=2,
            base_config=self.BaseConfig(),
            train_config=self.TrainConfig(),
            seed=42,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(result.total_loss))
        self.assertEqual(result.relation_positive_count, 1)
        result.total_loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_inference_never_injects_gold_spans(self):
        from journal.scsp.training import infer_window

        model = self._model()
        inference = infer_window(
            model,
            self._window(),
            self._inventory(),
            width_cap=2,
            base_config=self.BaseConfig(),
            relation_chunk_size=4,
            device=torch.device("cpu"),
        )
        self.assertTrue(
            all(
                span.proposal_source == "predicted"
                for span in inference.predicted_spans
            )
        )
        self.assertLessEqual(
            len(inference.predicted_spans),
            self.BaseConfig.max_span_candidates,
        )

    def test_threshold_selection_is_validation_only(self):
        from journal.scsp.training import select_relation_threshold

        with self.assertRaisesRegex(ValueError, "validation"):
            select_relation_threshold((), (0.5,), split="test")


if __name__ == "__main__":
    unittest.main()
