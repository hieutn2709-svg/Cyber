from __future__ import annotations

import unittest

from journal.scsp.data import WindowExample
from journal.scsp.pairs import GoldRelation
from journal.scsp.structures import GoldSpan, SpanCandidate


class CandidateBuilderTests(unittest.TestCase):
    def _window(self) -> WindowExample:
        a = GoldSpan("d", 2, 2, "intrusion-set")
        b = GoldSpan("d", 4, 4, "malware")
        return WindowExample(
            doc_seq_index=0,
            doc_id="d",
            window_index=3,
            token_start_global=100,
            token_end_global=105,
            input_ids=(0, 10, 11, 12, 13, 14, 2),
            attention_mask=(1, 1, 1, 1, 1, 1, 1),
            label_mask=(False, True, True, True, True, True, False),
            content_start=1,
            content_end=5,
            gold_spans=(a, b),
            gold_relations=(GoldRelation(a, b, "uses"),),
            primary_entity_types=("intrusion-set", "malware"),
            auxiliary_entity_types=(),
        )

    def test_enumeration_excludes_special_tokens(self) -> None:
        from journal.scsp.candidates import enumerate_content_spans

        spans = enumerate_content_spans(self._window(), max_width=2)
        coords = {(s.start, s.end) for s in spans}
        self.assertNotIn((0, 0), coords)
        self.assertNotIn((6, 6), coords)
        self.assertIn((1, 1), coords)
        self.assertIn((4, 5), coords)
        self.assertEqual(len(spans), 9)

    def test_local_to_global_uses_content_offset(self) -> None:
        from journal.scsp.candidates import local_to_global_candidate

        local = SpanCandidate("d", 2, 4, "malware", 0.8, "predicted")
        global_span = local_to_global_candidate(self._window(), local)
        self.assertEqual((global_span.start, global_span.end), (101, 103))
        self.assertEqual(global_span.label, "malware")
        self.assertEqual(global_span.entity_score, 0.8)

    def test_training_gold_injection_preserves_missing_typed_span(self) -> None:
        from journal.scsp.candidates import inject_gold_spans

        pruned = (
            SpanCandidate("d", 2, 2, "location", 0.9, "predicted"),
            SpanCandidate("d", 4, 4, "malware", 0.8, "predicted"),
        )
        injected = inject_gold_spans(pruned, self._window().gold_spans)
        keys = {s.typed_key for s in injected}
        self.assertIn(("d", 2, 2, "intrusion-set"), keys)
        self.assertIn(("d", 4, 4, "malware"), keys)
        gold_added = [s for s in injected if s.label == "intrusion-set"]
        self.assertEqual(len(gold_added), 1)
        self.assertEqual(gold_added[0].proposal_source, "gold_injected")

    def test_training_pair_sampler_preserves_positive_beyond_distance_cutoff(self) -> None:
        from journal.scsp.candidates import build_training_pairs

        source_gold = GoldSpan("d", 1, 1, "intrusion-set")
        target_gold = GoldSpan("d", 110, 110, "malware")
        relation = GoldRelation(source_gold, target_gold, "uses")
        spans = (
            SpanCandidate("d", 1, 1, "intrusion-set", 1.0, "gold_injected"),
            SpanCandidate("d", 110, 110, "malware", 1.0, "gold_injected"),
            SpanCandidate("d", 2, 2, "location", 0.8, "predicted"),
        )
        batch = build_training_pairs(
            spans,
            (relation,),
            max_token_distance=96,
            negative_ratio=2,
            seed=42,
        )
        positive_keys = {
            pair.ordered_key
            for pair, is_positive in zip(batch.pairs, batch.positive_mask)
            if is_positive
        }
        self.assertIn(relation.endpoint_key, positive_keys)
        self.assertEqual(sum(batch.positive_mask), 1)
        self.assertLessEqual(len(batch.pairs) - 1, 2)

    def test_pair_sampling_is_deterministic(self) -> None:
        from journal.scsp.candidates import build_training_pairs

        a = GoldSpan("d", 1, 1, "intrusion-set")
        b = GoldSpan("d", 4, 4, "malware")
        rel = GoldRelation(a, b, "uses")
        spans = tuple(
            SpanCandidate("d", i, i, label, 0.9, "predicted")
            for i, label in [
                (1, "intrusion-set"),
                (2, "location"),
                (3, "tool"),
                (4, "malware"),
                (5, "identity"),
            ]
        )
        first = build_training_pairs(spans, (rel,), 96, 3, seed=7)
        second = build_training_pairs(spans, (rel,), 96, 3, seed=7)
        self.assertEqual(first.pairs, second.pairs)
        self.assertEqual(first.positive_mask, second.positive_mask)


if __name__ == "__main__":
    unittest.main()
