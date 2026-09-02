from __future__ import annotations

import unittest

from journal.scsp.diagnostics import pair_diagnostics, pair_recall
from journal.scsp.pairs import GoldRelation, generate_ordered_pairs, pair_distance
from journal.scsp.structures import GoldSpan, SpanCandidate


class PairDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = SpanCandidate("d", 0, 0, "intrusion-set", 0.9)
        self.b = SpanCandidate("d", 5, 5, "malware", 0.8)
        self.c = SpanCandidate("d", 8, 9, "tool", 0.7)

    def test_ordered_pairs_preserve_direction(self) -> None:
        pairs = generate_ordered_pairs([self.a, self.b])
        keys = [(p.source.start, p.target.start) for p in pairs]
        self.assertEqual(keys, [(0, 5), (5, 0)])
        self.assertNotEqual(pairs[0].ordered_key, pairs[1].ordered_key)

    def test_pair_distance_counts_tokens_between_spans(self) -> None:
        self.assertEqual(pair_distance(self.a, self.b), 4)
        self.assertEqual(pair_distance(self.b, self.a), 4)
        overlap = SpanCandidate("d", 0, 2, "tool", 0.7)
        touching = SpanCandidate("d", 3, 3, "malware", 0.7)
        self.assertEqual(pair_distance(overlap, touching), 0)

    def test_distance_filter_reduces_pair_recall_only_when_endpoint_pair_removed(self) -> None:
        gold = [
            GoldRelation(
                source=GoldSpan("d", 0, 0, "intrusion-set"),
                target=GoldSpan("d", 5, 5, "malware"),
                label="uses",
            )
        ]
        all_pairs = generate_ordered_pairs([self.a, self.b])
        filtered = generate_ordered_pairs([self.a, self.b], max_token_distance=3)
        self.assertEqual(pair_recall(gold, all_pairs), 1.0)
        self.assertEqual(pair_recall(gold, filtered), 0.0)

    def test_pair_diagnostics_reports_candidate_class_balance(self) -> None:
        gold = [
            GoldRelation(
                source=GoldSpan("d", 0, 0, "intrusion-set"),
                target=GoldSpan("d", 5, 5, "malware"),
                label="uses",
            )
        ]
        pairs = generate_ordered_pairs([self.a, self.b, self.c])
        diagnostics = pair_diagnostics(gold, pairs)
        self.assertEqual(diagnostics.candidate_count, 6)
        self.assertEqual(diagnostics.positive_candidate_count, 1)
        self.assertEqual(diagnostics.negative_candidate_count, 5)
        self.assertEqual(diagnostics.matched_relation_count, 1)
        self.assertEqual(dict(diagnostics.recall_by_relation_type)["uses"], 1.0)

    def test_spans_from_different_documents_are_never_paired(self) -> None:
        other = SpanCandidate("other", 1, 1, "malware", 0.8)
        pairs = generate_ordered_pairs([self.a, other])
        self.assertEqual(pairs, ())


if __name__ == "__main__":
    unittest.main()
