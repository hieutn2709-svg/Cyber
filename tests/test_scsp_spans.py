from __future__ import annotations

import unittest

from journal.scsp.spans import (
    derive_width_cap,
    enumerate_spans,
    prune_span_candidates,
    span_diagnostics,
    span_recall,
)
from journal.scsp.structures import GoldSpan, SpanCandidate


class SpanCandidateTests(unittest.TestCase):
    def test_width_cap_is_smallest_width_reaching_training_coverage(self) -> None:
        gold = [GoldSpan("d", i, i, "tool") for i in range(199)]
        gold.append(GoldSpan("d", 300, 309, "malware"))
        self.assertEqual(derive_width_cap(gold, coverage=0.995), 1)
        self.assertEqual(derive_width_cap(gold, coverage=1.0), 10)

    def test_enumerate_spans_uses_inclusive_boundaries(self) -> None:
        spans = enumerate_spans(token_count=3, max_width=2, document_id="d")
        coords = [(s.start, s.end) for s in spans]
        self.assertEqual(coords, [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])
        self.assertTrue(all(s.document_id == "d" for s in spans))

    def test_pruning_is_deterministic_for_equal_scores(self) -> None:
        candidates = [
            SpanCandidate("d", 2, 2, "tool", 0.8),
            SpanCandidate("d", 0, 0, "malware", 0.8),
            SpanCandidate("d", 1, 1, "identity", 0.3),
        ]
        kept = prune_span_candidates(candidates, max_candidates=2, min_entity_score=0.5)
        self.assertEqual([(x.start, x.label) for x in kept], [(0, "malware"), (2, "tool")])

    def test_typed_span_recall_requires_exact_boundaries_and_type(self) -> None:
        gold = [
            GoldSpan("d", 0, 0, "malware"),
            GoldSpan("d", 2, 2, "tool"),
        ]
        candidates = [
            SpanCandidate("d", 0, 0, "malware", 0.9),
            SpanCandidate("d", 2, 2, "malware", 0.9),
            SpanCandidate("d", 1, 2, "tool", 0.9),
        ]
        self.assertEqual(span_recall(gold, candidates), 0.5)
        diagnostics = span_diagnostics(gold, candidates)
        self.assertEqual(diagnostics.gold_count, 2)
        self.assertEqual(diagnostics.matched_count, 1)
        self.assertEqual(dict(diagnostics.recall_by_type)["malware"], 1.0)
        self.assertEqual(dict(diagnostics.recall_by_type)["tool"], 0.0)

    def test_invalid_gold_span_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GoldSpan("d", 4, 3, "tool")


if __name__ == "__main__":
    unittest.main()
