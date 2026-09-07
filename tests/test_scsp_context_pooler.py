from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class ContextPoolerTests(unittest.TestCase):
    def test_between_span_pooler_returns_requested_shape(self) -> None:
        from journal.scsp.context_pooler import BetweenSpanContextPooler

        torch.manual_seed(0)
        states = torch.randn(8, 6)
        endpoints = torch.tensor(
            [[1, 1, 4, 4], [5, 5, 2, 2]],
            dtype=torch.long,
        )
        pooler = BetweenSpanContextPooler(hidden_size=6, output_dim=5)
        output = pooler(states, endpoints)
        self.assertEqual(tuple(output.shape), (2, 5))

    def test_adjacent_or_overlapping_spans_use_learned_empty_context(self) -> None:
        from journal.scsp.context_pooler import BetweenSpanContextPooler

        torch.manual_seed(1)
        states = torch.randn(6, 4)
        endpoints = torch.tensor(
            [[1, 2, 3, 3], [1, 3, 2, 4]],
            dtype=torch.long,
        )
        pooler = BetweenSpanContextPooler(hidden_size=4, output_dim=3)
        output = pooler(states, endpoints)
        self.assertTrue(torch.allclose(output[0], pooler.empty_context))
        self.assertTrue(torch.allclose(output[1], pooler.empty_context))

    def test_nonempty_between_context_depends_on_between_tokens(self) -> None:
        from journal.scsp.context_pooler import BetweenSpanContextPooler

        torch.manual_seed(2)
        states = torch.zeros(6, 4)
        states[2] = 1.0
        states[3] = 2.0
        pooler = BetweenSpanContextPooler(hidden_size=4, output_dim=3)
        endpoints = torch.tensor([[1, 1, 4, 4]], dtype=torch.long)
        first = pooler(states, endpoints)
        changed = states.clone()
        changed[2] = 9.0
        second = pooler(changed, endpoints)
        self.assertFalse(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
