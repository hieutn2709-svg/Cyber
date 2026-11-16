from __future__ import annotations
import copy
import json
import random
import unittest
from journal.scsp.candidates import build_training_pairs, inject_gold_spans
from journal.scsp.pairs import GoldRelation
from journal.scsp.structures import GoldSpan, SpanCandidate

class TrainingPairDiagnosticsTests(unittest.TestCase):
    def fixture(self, document='d'):
        a = GoldSpan(document, 1, 1, 'intrusion-set')
        b = GoldSpan(document, 4, 4, 'malware')
        spans = inject_gold_spans((
            SpanCandidate(document, 1, 1, 'tool', .8, 'predicted'),
            SpanCandidate(document, 4, 4, 'malware', .9, 'predicted'),
        ), (a, b))
        return spans, (GoldRelation(a, b, 'uses'),)

    def observe(self, spans, gold, distance=96, ratio=6, seed=42):
        record = {}
        batch = build_training_pairs(spans, gold, distance, ratio, seed=seed, diagnostics=record)
        return batch, record

    def test_directed_collision_counts_and_examples(self):
        batch, record = self.observe(*self.fixture())
        self.assertEqual(record['eligible'], dict(positive_pairs=1, negative_pairs=3,
            boundary_groups=2, collision_groups=1, collision_positive_pairs=1, collision_negative_pairs=1))
        self.assertEqual(record['sampled'], record['eligible'])
        example = record['collision_examples'][0]
        self.assertEqual(example['boundary_key'], ['d', 1, 1, 4, 4])
        self.assertEqual(example['eligible_positive_type_pairs'], [['intrusion-set', 'malware']])
        self.assertEqual(example['eligible_negative_type_pairs'], [['tool', 'malware']])
        self.assertEqual(example['sampled_negative_pairs'], 1)
        self.assertEqual(len(record['sampled_batch_sha256']), 64)
        self.assertEqual(len(batch.pairs), 4)
        json.dumps(record, allow_nan=False)

    def test_zero_budget_keeps_eligible_collision_only(self):
        batch, record = self.observe(*self.fixture(), ratio=0)
        self.assertEqual(record['eligible']['collision_groups'], 1)
        self.assertEqual(record['sampled']['collision_groups'], 0)
        self.assertEqual(record['sampled']['negative_pairs'], 0)
        self.assertEqual(batch.positive_mask, (True,))

    def test_distant_positive_and_self_relation(self):
        spans, gold = self.fixture()
        gold += (GoldRelation(gold[0].source, gold[0].source, 'targets'),)
        _, record = self.observe(spans, gold, distance=0)
        self.assertEqual(record['eligible']['positive_pairs'], 1)
        self.assertEqual(record['eligible']['negative_pairs'], 0)
        self.assertEqual(record['gold_relation_instances'], 2)
        self.assertEqual(record['gold_self_relation_instances'], 1)
        self.assertEqual(record['eligible_long_positive_pairs'], 1)

    def test_empty_and_no_positive_windows(self):
        _, empty = self.observe((), ())
        self.assertEqual(empty['eligible']['boundary_groups'], 0)
        spans, _ = self.fixture()
        _, record = self.observe(spans, (), ratio=2)
        self.assertEqual(record['eligible']['negative_pairs'], 4)
        self.assertEqual(record['sampled']['negative_pairs'], 2)
        self.assertEqual(record['eligible']['collision_groups'], 0)

    def test_batch_order_labels_rng_inputs_parity(self):
        spans, gold = self.fixture()
        original = copy.deepcopy((spans, gold))
        for seed in (42, 43, 44):
            for ratio in (0, 1, 6):
                with self.subTest(seed=seed, ratio=ratio):
                    state = random.getstate()
                    baseline = build_training_pairs(spans, gold, 96, ratio, seed=seed)
                    observed, first = self.observe(spans, gold, ratio=ratio, seed=seed)
                    _, second = self.observe(spans, gold, ratio=ratio, seed=seed)
                    self.assertEqual(baseline, observed)
                    self.assertEqual(first, second)
                    self.assertEqual(state, random.getstate())
                    self.assertEqual((spans, gold), original)

    def test_examples_bounded_and_document_keys_separated(self):
        spans, gold = [], []
        for i in range(8):
            s, g = self.fixture(str(i))
            spans.extend(s)
            gold.extend(g)
        _, record = self.observe(spans, gold)
        self.assertEqual(record['eligible']['collision_groups'], 8)
        self.assertEqual(record['sampled']['collision_groups'], 8)
        self.assertEqual(len(record['collision_examples']), 5)
        self.assertEqual(record['collision_examples_omitted'], 3)

if __name__ == '__main__': unittest.main()
