from __future__ import annotations

import math
import unittest
from types import MappingProxyType

try:
    from journal.scsp.gate_c import build_probabilistic_prediction_records
except ImportError:
    build_probabilistic_prediction_records = None

from journal.scsp.data import WindowExample
from journal.scsp.gate_c import conditional_non_none_posterior
from journal.scsp.gate_c_inference import GateCWindowInference
from journal.scsp.pairs import PairCandidate
from journal.scsp.schema import (
    CanonicalRelation,
    CanonicalizationTable,
    TaskRelationshipProfile,
)
from journal.scsp.structures import SpanCandidate
from journal.scsp.training import ScoredRelationPair, WindowInference


class GateCDecoderTests(unittest.TestCase):
    RELATION_TYPES = ("uses", "targets")
    ENTITY_TYPES = ("intrusion-set", "identity")

    def setUp(self) -> None:
        self.window = WindowExample(
            doc_seq_index=0,
            doc_id="doc-1",
            window_index=0,
            token_start_global=0,
            token_end_global=4,
            input_ids=(101, 11, 12, 13, 102),
            attention_mask=(1, 1, 1, 1, 1),
            label_mask=(False, True, True, True, False),
            content_start=1,
            content_end=3,
            gold_spans=(),
            gold_relations=(),
            primary_entity_types=self.ENTITY_TYPES,
            auxiliary_entity_types=(),
        )
        self.source = SpanCandidate(
            document_id="doc-1",
            start=1,
            end=1,
            label="intrusion-set",
            entity_score=1.0,
            proposal_source="predicted",
        )
        self.target = SpanCandidate(
            document_id="doc-1",
            start=3,
            end=3,
            label="identity",
            entity_score=1.0,
            proposal_source="predicted",
        )
        pair = PairCandidate(self.source, self.target, token_distance=1)
        self.scored_pair = ScoredRelationPair(
            pair=pair,
            existence_logit=3.0,
            type_logits=(5.0, 4.0),
        )
        self.base = WindowInference(
            window=self.window,
            predicted_spans=(self.source, self.target),
            scored_pairs=(self.scored_pair,),
            proposal_gold_count=0,
            proposal_matched_count=0,
            post_pruning_typed_matched_count=0,
        )
        source_posterior = conditional_non_none_posterior(
            [0.0, 1.0, 0.0],
            self.ENTITY_TYPES,
            span_key=self.source.typed_key,
        )
        target_posterior = conditional_non_none_posterior(
            [0.0, 0.0, 1.0],
            self.ENTITY_TYPES,
            span_key=self.target.typed_key,
        )
        self.inference = GateCWindowInference(
            base=self.base,
            posterior_by_typed_key=MappingProxyType(
                {
                    self.source.typed_key: source_posterior,
                    self.target.typed_key: target_posterior,
                }
            ),
        )
        self.canonicalization = CanonicalizationTable(
            by_project_label={
                "uses": CanonicalRelation("uses", False, "direct"),
                "targets": CanonicalRelation("targets", False, "direct"),
            }
        )
        self.profile = TaskRelationshipProfile(
            allowed_triples=frozenset(
                {("intrusion-set", "targets", "identity")}
            ),
            resolved_entity_types=frozenset(self.ENTITY_TYPES),
            unresolved_entity_types=frozenset(),
        )

    def _records(self, *, beta: float):
        self.assertIsNotNone(
            build_probabilistic_prediction_records,
            "Gate C probabilistic document decoder must exist",
        )
        return build_probabilistic_prediction_records(
            (self.inference,),
            self.RELATION_TYPES,
            beta=beta,
            threshold=0.90,
            canonicalization=self.canonicalization,
            profile=self.profile,
            run_id="gate-c-test",
            git_commit="commit",
            dataset_sha256="dataset",
            config_sha256="config",
            fold=1,
            seed=42,
            split="validation",
        )

    def test_beta_zero_preserves_raw_argmax_and_gate_a_style_relation(self) -> None:
        records = self._records(beta=0.0)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].predicted_relations), 1)
        relation = records[0].predicted_relations[0]
        self.assertEqual(relation.label, "uses")
        self.assertAlmostEqual(
            relation.relation_score,
            1.0 / (1.0 + math.exp(-3.0)),
            places=12,
        )
        self.assertEqual(relation.source.typed_key, self.source.typed_key)
        self.assertEqual(relation.target.typed_key, self.target.typed_key)

    def test_positive_beta_changes_only_relation_type(self) -> None:
        baseline = self._records(beta=0.0)[0]
        adjusted = self._records(beta=2.0)[0]

        self.assertEqual(adjusted.predicted_spans, baseline.predicted_spans)
        self.assertEqual(len(adjusted.predicted_relations), 1)
        self.assertEqual(len(baseline.predicted_relations), 1)
        base_relation = baseline.predicted_relations[0]
        gate_c_relation = adjusted.predicted_relations[0]
        self.assertAlmostEqual(
            gate_c_relation.relation_score,
            base_relation.relation_score,
            places=12,
        )
        self.assertEqual(gate_c_relation.source, base_relation.source)
        self.assertEqual(gate_c_relation.target, base_relation.target)
        self.assertEqual(base_relation.label, "uses")
        self.assertEqual(gate_c_relation.label, "targets")


if __name__ == "__main__":
    unittest.main()
