from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from journal.evaluation.rescore_gate_a import strict_micro_scores
from journal.scsp.pairs import GoldRelation
from journal.scsp.serialization import (
    PredictedRelation,
    PredictionRecord,
    read_prediction_jsonl,
    write_prediction_jsonl,
)
from journal.scsp.structures import GoldSpan, SpanCandidate


class SerializationRescoreTests(unittest.TestCase):
    def _records(self) -> tuple[PredictionRecord, ...]:
        g11 = GoldSpan("d1", 0, 0, "intrusion-set")
        g12 = GoldSpan("d1", 4, 4, "malware")
        p11 = SpanCandidate("d1", 0, 0, "intrusion-set", 0.9)
        p12 = SpanCandidate("d1", 4, 4, "malware", 0.8)
        p13 = SpanCandidate("d1", 8, 8, "tool", 0.6)
        r1 = PredictionRecord(
            "run",
            "abc",
            "datahash",
            "cfghash",
            1,
            42,
            "test",
            "d1",
            (g11, g12),
            (p11, p12, p13),
            (GoldRelation(g11, g12, "uses"),),
            (
                PredictedRelation(p11, p12, "uses", 0.9),
                PredictedRelation(p11, p13, "targets", 0.7),
            ),
        )

        g21 = GoldSpan("d2", 1, 1, "identity")
        g22 = GoldSpan("d2", 3, 3, "location")
        p22 = SpanCandidate("d2", 3, 3, "location", 0.9)
        r2 = PredictionRecord(
            "run",
            "abc",
            "datahash",
            "cfghash",
            1,
            42,
            "test",
            "d2",
            (g21, g22),
            (p22,),
            (GoldRelation(g21, g22, "located-at"),),
            (),
        )
        return (r1, r2)

    def test_jsonl_round_trip(self) -> None:
        records = self._records()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "predictions.jsonl"
            write_prediction_jsonl(records, path)
            loaded = read_prediction_jsonl(path)
        self.assertEqual(loaded, records)

    def test_strict_micro_rescore(self) -> None:
        scores = strict_micro_scores(self._records())
        self.assertEqual(scores["entity"]["tp"], 3)
        self.assertEqual(scores["entity"]["fp"], 1)
        self.assertEqual(scores["entity"]["fn"], 1)
        self.assertAlmostEqual(scores["entity"]["f1"], 0.75)
        self.assertEqual(scores["relation"]["tp"], 1)
        self.assertEqual(scores["relation"]["fp"], 1)
        self.assertEqual(scores["relation"]["fn"], 1)
        self.assertAlmostEqual(scores["relation"]["f1"], 0.5)

    def test_rejects_predicted_relation_endpoint_not_in_predicted_spans(self) -> None:
        g1 = GoldSpan("d", 0, 0, "intrusion-set")
        g2 = GoldSpan("d", 2, 2, "malware")
        p1 = SpanCandidate("d", 0, 0, "intrusion-set", 0.9)
        missing = SpanCandidate("d", 2, 2, "malware", 0.9)
        with self.assertRaisesRegex(ValueError, "predicted relation endpoint"):
            PredictionRecord(
                "run",
                "abc",
                "data",
                "cfg",
                1,
                42,
                "test",
                "d",
                (g1, g2),
                (p1,),
                (GoldRelation(g1, g2, "uses"),),
                (PredictedRelation(p1, missing, "uses", 0.8),),
            )


if __name__ == "__main__":
    unittest.main()
