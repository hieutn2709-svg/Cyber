from __future__ import annotations

import unittest

from journal.evaluation.rescore_gate_a import strict_micro_scores
from journal.scsp.pairs import GoldRelation
from journal.scsp.serialization import PredictedRelation, PredictionRecord
from journal.scsp.structures import GoldSpan, SpanCandidate


class RescoreScopeTests(unittest.TestCase):
    def test_primary_entity_and_core_relation_scope_filters_auxiliary_labels(self):
        core_a = GoldSpan("d", 1, 1, "malware")
        core_b = GoldSpan("d", 3, 3, "tool")
        auxiliary = GoldSpan("d", 5, 5, "tactic")
        pred_a = SpanCandidate("d", 1, 1, "malware", 0.9, "predicted")
        pred_b = SpanCandidate("d", 3, 3, "tool", 0.9, "predicted")
        record = PredictionRecord(
            "r",
            "g",
            "d",
            "c",
            1,
            42,
            "test",
            "d",
            (core_a, core_b, auxiliary),
            (pred_a, pred_b),
            (
                GoldRelation(core_a, core_b, "uses"),
                GoldRelation(core_a, auxiliary, "targets"),
            ),
            (PredictedRelation(pred_a, pred_b, "uses", 0.8),),
        )
        all_scores = strict_micro_scores((record,))
        scoped = strict_micro_scores(
            (record,),
            entity_labels={"malware", "tool"},
            relation_endpoint_labels={"malware", "tool"},
        )
        self.assertEqual(all_scores["entity"]["fn"], 1)
        self.assertEqual(scoped["entity"]["fn"], 0)
        self.assertEqual(all_scores["relation"]["fn"], 1)
        self.assertEqual(scoped["relation"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
