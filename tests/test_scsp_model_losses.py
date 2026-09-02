from __future__ import annotations

import inspect
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class ModelLossTests(unittest.TestCase):
    def test_span_pooler_and_entity_head_shapes(self) -> None:
        from journal.scsp.model import SpanEntityHead, SpanPooler

        torch.manual_seed(0)
        token_states = torch.randn(7, 8)
        spans = torch.tensor([[0, 0], [1, 3], [5, 6]], dtype=torch.long)
        pooler = SpanPooler(hidden_size=8, width_embedding_dim=4, max_width=4)
        span_reps = pooler(token_states, spans)
        self.assertEqual(tuple(span_reps.shape), (3, 28))
        logits = SpanEntityHead(span_dim=28, num_entity_types=11)(span_reps)
        self.assertEqual(tuple(logits.shape), (3, 11))

    def test_pair_representation_is_directional(self) -> None:
        from journal.scsp.model import PairRepresentation

        torch.manual_seed(1)
        span_a = torch.randn(1, 6)
        span_b = torch.randn(1, 6)
        context = torch.randn(1, 5)
        distance = torch.tensor([4], dtype=torch.long)
        pair = PairRepresentation(
            span_dim=6,
            context_dim=5,
            distance_embedding_dim=3,
            max_distance=10,
        )
        ab = pair(span_a, span_b, context, distance)
        ba = pair(span_b, span_a, context, distance)
        self.assertEqual(tuple(ab.shape), (1, 32))
        self.assertFalse(torch.allclose(ab, ba))

    def test_relation_heads_and_losses_are_finite(self) -> None:
        from journal.scsp.losses import (
            focal_binary_cross_entropy,
            positive_relation_type_loss,
        )
        from journal.scsp.model import RelationExistenceHead, RelationTypeHead

        features = torch.randn(4, 10)
        existence_logits = RelationExistenceHead(10)(features)
        type_logits = RelationTypeHead(10, 13)(features)
        self.assertEqual(tuple(existence_logits.shape), (4,))
        self.assertEqual(tuple(type_logits.shape), (4, 13))

        existence_targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
        existence_loss = focal_binary_cross_entropy(
            existence_logits, existence_targets, pos_weight=2.0, gamma=1.0
        )
        type_labels = torch.tensor([3, 0, 0, 5], dtype=torch.long)
        positive_mask = existence_targets.bool()
        type_loss = positive_relation_type_loss(type_logits, type_labels, positive_mask)
        self.assertTrue(torch.isfinite(existence_loss))
        self.assertTrue(torch.isfinite(type_loss))

    def test_zero_positive_relation_batch_has_differentiable_zero_type_loss(self) -> None:
        from journal.scsp.losses import positive_relation_type_loss

        logits = torch.randn(3, 4, requires_grad=True)
        labels = torch.zeros(3, dtype=torch.long)
        mask = torch.zeros(3, dtype=torch.bool)
        loss = positive_relation_type_loss(logits, labels, mask)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_plain_heads_forward_has_no_schema_argument(self) -> None:
        from journal.scsp.model import PlainSpanPairHeads

        parameters = inspect.signature(PlainSpanPairHeads.forward).parameters
        self.assertNotIn("schema", parameters)
        self.assertNotIn("compatibility", parameters)


if __name__ == "__main__":
    unittest.main()
