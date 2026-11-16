"""Behavioral tests for the opt-in boundary-conflict replacement ablation."""
import random
import unittest
from journal.scsp.candidates import build_training_pairs
from journal.scsp.pairs import GoldRelation
from journal.scsp.structures import GoldSpan, SpanCandidate


def boundary(pair):
    return (pair.document_id, pair.source.start, pair.source.end,
            pair.target.start, pair.target.end)


class BoundaryConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.spans = tuple(SpanCandidate('d', start, start, label=label)
                           for start, label in ((0,'A'),(0,'B'),(3,'C'),(3,'D'),(6,'E')))
        self.gold = (GoldRelation(GoldSpan('d',0,0,'A'),GoldSpan('d',3,3,'C'),'r'),)

    def sample(self, policy='boundary_replace', *, seed=42, ratio=3, spans=None, gold=None):
        stats = {}
        batch = build_training_pairs(self.spans if spans is None else spans,
            self.gold if gold is None else gold, 96, ratio, seed=seed,
            negative_policy=policy, diagnostics=stats)
        return batch, stats

    def test_replacement_oracle_retains_safe_draws_and_budget(self):
        # Exhaustive production pool, then independent local-RNG draw/repair oracle.
        pool = build_training_pairs(self.spans,self.gold,96,10000,seed=0)
        positive=[p for p,m in zip(pool.pairs,pool.positive_mask) if m]
        negative=[p for p,m in zip(pool.pairs,pool.positive_mask) if not m]
        poskeys={boundary(p) for p in positive}
        saw_collision=False
        for seed in range(100):
            rng=random.Random(seed)
            drawn=sorted(rng.sample(range(len(negative)),3))
            kept=[i for i in drawn if boundary(negative[i]) not in poskeys]
            safe=[i for i,p in enumerate(negative) if i not in drawn and boundary(p) not in poskeys]
            replacements=rng.sample(safe,3-len(kept))
            expected=positive+[negative[i] for i in sorted(kept+replacements)]
            actual,stats=self.sample(seed=seed)
            self.assertEqual(actual.pairs,tuple(expected))
            self.assertEqual(stats['sampled']['collision_groups'],0)
            self.assertEqual(stats['negative_policy']['removed_sampled_negatives'],len(replacements))
            self.assertEqual(stats['negative_policy']['replacement_negatives'],len(replacements))
            self.assertEqual(stats['negative_policy']['negative_budget_shortfall'],0)
            baseline=build_training_pairs(self.spans,self.gold,96,3,seed=seed)
            for pair,mask in zip(baseline.pairs,baseline.positive_mask):
                if mask or boundary(pair) not in poskeys:self.assertIn(pair,actual.pairs)
            saw_collision |= bool(replacements)
        self.assertTrue(saw_collision)

    def test_exhaustion_keeps_reverse_and_other_documents(self):
        other=tuple(SpanCandidate('other',p.start,p.end,label=p.label) for p in self.spans)
        actual,stats=self.sample(spans=self.spans+other,ratio=10000)
        self.assertEqual(stats['negative_policy']['removed_sampled_negatives'],3)
        self.assertEqual(stats['negative_policy']['replacement_negatives'],0)
        self.assertEqual(stats['negative_policy']['negative_budget_shortfall'],3)
        negatives=[p for p,m in zip(actual.pairs,actual.positive_mask) if not m]
        self.assertTrue(any(boundary(p)==('d',3,3,0,0) for p in negatives))
        self.assertTrue(any(boundary(p)==('other',0,0,3,3) for p in negatives))
        self.assertFalse(any(boundary(p)==('d',0,0,3,3) for p in negatives))

    def test_no_positive_zero_budget_default_and_global_rng(self):
        for gold,ratio in (((),3),(self.gold,0)):
            actual,stats=self.sample(gold=gold,ratio=ratio)
            self.assertEqual(actual,build_training_pairs(self.spans,gold,96,ratio,seed=42))
            self.assertEqual(stats['negative_policy']['removed_sampled_negatives'],0)
        state=random.getstate()
        self.sample()
        self.assertEqual(state,random.getstate())
        default={}
        original=build_training_pairs(self.spans,self.gold,96,3,seed=42,diagnostics=default)
        explicit,stats=self.sample('typed')
        self.assertEqual(explicit,original)
        self.assertEqual(stats,default)
        self.assertNotIn('negative_policy',stats)

    def test_long_positive_self_edge_and_invalid_policy(self):
        gold=self.gold+(GoldRelation(GoldSpan('d',0,0,'A'),GoldSpan('d',0,0,'A'),'r'),)
        stats={}
        actual=build_training_pairs(self.spans,gold,0,100,seed=1,
            negative_policy='boundary_replace',diagnostics=stats)
        self.assertEqual(sum(actual.positive_mask),1)
        self.assertEqual(stats['gold_self_relation_instances'],1)
        self.assertEqual(stats['eligible_long_positive_pairs'],1)
        with self.assertRaisesRegex(ValueError,'negative policy'):
            self.sample('unknown')


if __name__=='__main__':unittest.main()
