from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class RuntimeModelTests(unittest.TestCase):
    class DummyEncoder(nn.Module):
        def __init__(self, hidden_size: int = 8) -> None:
            super().__init__()
            self.embedding = nn.Embedding(50, hidden_size)
            self.config = SimpleNamespace(hidden_size=hidden_size)

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(
                last_hidden_state=self.embedding(input_ids)
            )

    def test_runtime_model_encodes_and_exposes_poolers_heads(self) -> None:
        from journal.scsp.runtime_model import GateASpanPairModel

        model = GateASpanPairModel(
            encoder=self.DummyEncoder(8),
            hidden_size=8,
            num_entity_classes=4,
            num_relation_types=3,
            max_width=6,
            width_embedding_dim=4,
            context_dim=5,
            distance_embedding_dim=3,
            max_distance=96,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        mask = torch.ones_like(input_ids)
        states = model.encode(input_ids, mask)
        self.assertEqual(tuple(states.shape), (1, 4, 8))
        self.assertEqual(model.span_pooler.output_dim, 28)
        self.assertEqual(model.context_pooler.output_dim, 5)

    def test_parameter_groups_separate_encoder_from_new_heads(self) -> None:
        from journal.scsp.runtime_model import GateASpanPairModel

        model = GateASpanPairModel(
            encoder=self.DummyEncoder(8),
            hidden_size=8,
            num_entity_classes=4,
            num_relation_types=3,
            max_width=6,
            width_embedding_dim=4,
            context_dim=5,
            distance_embedding_dim=3,
            max_distance=96,
        )
        encoder_ids = {id(p) for p in model.encoder_parameters()}
        head_ids = {id(p) for p in model.head_parameters()}
        self.assertTrue(encoder_ids)
        self.assertTrue(head_ids)
        self.assertFalse(encoder_ids & head_ids)

    def test_from_pretrained_rejects_mutable_revision_before_loading(self) -> None:
        from journal.scsp.runtime_model import GateASpanPairModel

        with self.assertRaisesRegex(ValueError, "40-character"):
            GateASpanPairModel.from_pretrained(
                model_name="FacebookAI/roberta-base",
                revision="main",
                num_entity_classes=16,
                num_relation_types=13,
                max_width=6,
                width_embedding_dim=32,
                context_dim=128,
                distance_embedding_dim=32,
                max_distance=96,
            )


if __name__ == "__main__":
    unittest.main()
