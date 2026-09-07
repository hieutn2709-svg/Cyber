from __future__ import annotations

import unittest

from journal.scsp.data import WindowExample
from journal.scsp.pairs import GoldRelation, PairCandidate
from journal.scsp.structures import GoldSpan, SpanCandidate
from journal.scsp.training import ScoredRelationPair, WindowInference


class ArtifactTests(unittest.TestCase):
    def _inference(self, window_index, token_start_global, local_entity_start):
        gold_a = GoldSpan(
            "d", local_entity_start, local_entity_start, "intrusion-set"
        )
        gold_b = GoldSpan(
            "d", local_entity_start + 2, local_entity_start + 2, "malware"
        )
        rel = GoldRelation(gold_a, gold_b, "uses")
        window = WindowExample(
            doc_seq_index=0,
            doc_id="d",
            window_index=window_index,
            token_start_global=token_start_global,
            token_end_global=token_start_global + 5,
            input_ids=(0, 1, 2, 3, 4, 2),
            attention_mask=(1, 1, 1, 1, 1, 1),
            label_mask=(False, True, True, True, True, False),
            content_start=1,
            content_end=4,
            gold_spans=(gold_a, gold_b),
            gold_relations=(rel,),
            primary_entity_types=("intrusion-set", "malware"),
            auxiliary_entity_types=(),
        )
        pred_a = SpanCandidate(
            "d", gold_a.start, gold_a.end, "intrusion-set", 0.9, "predicted"
        )
        pred_b = SpanCandidate(
            "d", gold_b.start, gold_b.end, "malware", 0.8, "predicted"
        )
        pair = PairCandidate(pred_a, pred_b, 1)
        scored = ScoredRelationPair(pair, 2.0, (3.0, 0.0))
        return WindowInference(window, (pred_a, pred_b), (scored,), 2, 2, 2)

    def test_prediction_records_use_document_global_coordinates_and_aggregate_windows(self):
        from journal.scsp.artifacts import build_prediction_records

        first = self._inference(0, 0, 1)
        second = self._inference(1, 10, 1)
        records = build_prediction_records(
            (first, second),
            relation_types=("uses", "targets"),
            threshold=0.5,
            run_id="run",
            git_commit="abc",
            dataset_sha256="data",
            config_sha256="cfg",
            fold=1,
            seed=42,
            split="test",
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.document_id, "d")
        self.assertEqual(
            {span.start for span in record.gold_spans},
            {0, 2, 10, 12},
        )
        self.assertEqual(len(record.predicted_relations), 2)

    def test_threshold_controls_relation_decode_without_changing_entities(self):
        from journal.scsp.artifacts import build_prediction_records

        inference = self._inference(0, 0, 1)
        low = build_prediction_records(
            (inference,),
            ("uses", "targets"),
            0.5,
            run_id="r",
            git_commit="g",
            dataset_sha256="d",
            config_sha256="c",
            fold=1,
            seed=42,
            split="validation",
        )[0]
        high = build_prediction_records(
            (inference,),
            ("uses", "targets"),
            0.99,
            run_id="r",
            git_commit="g",
            dataset_sha256="d",
            config_sha256="c",
            fold=1,
            seed=42,
            split="validation",
        )[0]
        self.assertEqual(low.predicted_spans, high.predicted_spans)
        self.assertEqual(len(low.predicted_relations), 1)
        self.assertEqual(len(high.predicted_relations), 0)


if __name__ == "__main__":
    unittest.main()
