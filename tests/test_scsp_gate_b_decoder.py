from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.artifacts import build_prediction_records
from journal.scsp.data import WindowExample
from journal.scsp.gate_b import build_hard_profile_prediction_records
from journal.scsp.pairs import PairCandidate
from journal.scsp.schema import (
    load_relation_canonicalization,
    load_task_relationship_profile,
)
from journal.scsp.structures import SpanCandidate
from journal.scsp.training import ScoredRelationPair, WindowInference


class GateBDecoderTests(unittest.TestCase):
    def _schema(self, rules, *, resolved, unresolved=(), triples=()):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        canonicalization_path = root / "canonicalization.json"
        profile_path = root / "profile.json"
        canonicalization_path.write_text(
            json.dumps({"version": 1, "rules": list(rules)}), encoding="utf-8"
        )
        profile_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "resolved_entity_types": list(resolved),
                    "unresolved_entity_types": list(unresolved),
                    "allowed_triples": [
                        {
                            "source_type": source,
                            "relation_type": relation,
                            "target_type": target,
                            "source_note": "test",
                        }
                        for source, relation, target in triples
                    ],
                }
            ),
            encoding="utf-8",
        )
        return (
            td,
            load_relation_canonicalization(canonicalization_path),
            load_task_relationship_profile(profile_path),
        )

    def _inference(self, source_label, target_label, logits):
        window = WindowExample(
            doc_seq_index=0,
            doc_id="doc-1",
            window_index=0,
            token_start_global=0,
            token_end_global=2,
            input_ids=(101, 11, 12, 102),
            attention_mask=(1, 1, 1, 1),
            label_mask=(False, True, True, False),
            content_start=1,
            content_end=2,
            gold_spans=(),
            gold_relations=(),
            primary_entity_types=(),
            auxiliary_entity_types=(),
        )
        source = SpanCandidate("doc-1", 1, 1, source_label, 0.9)
        target = SpanCandidate("doc-1", 2, 2, target_label, 0.8)
        scored = ScoredRelationPair(
            PairCandidate(source, target, 0),
            existence_logit=10.0,
            type_logits=tuple(logits),
        )
        return WindowInference(
            window=window,
            predicted_spans=(source, target),
            scored_pairs=(scored,),
            proposal_gold_count=0,
            proposal_matched_count=0,
            post_pruning_typed_matched_count=0,
        )

    def _kwargs(self):
        return {
            "run_id": "gate-b-test",
            "git_commit": "a" * 40,
            "dataset_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "fold": 1,
            "seed": 42,
            "split": "validation",
        }

    def test_hard_profile_changes_only_relation_type_choice(self):
        rules = (
            {"project_label":"targets","canonical_label":"targets","swap_endpoints":False,"status":"direct"},
            {"project_label":"uses","canonical_label":"uses","swap_endpoints":False,"status":"direct"},
        )
        td, canonicalization, profile = self._schema(
            rules,
            resolved=("intrusion-set", "malware"),
            triples=(("intrusion-set", "uses", "malware"),),
        )
        self.addCleanup(td.cleanup)
        inference = self._inference("intrusion-set", "malware", (9.0, 8.0))
        original_logits = inference.scored_pairs[0].type_logits
        baseline = build_prediction_records(
            (inference,), ("targets", "uses"), 0.5, **self._kwargs()
        )
        hard = build_hard_profile_prediction_records(
            (inference,),
            ("targets", "uses"),
            0.5,
            canonicalization=canonicalization,
            profile=profile,
            **self._kwargs(),
        )
        self.assertEqual(baseline[0].predicted_spans, hard[0].predicted_spans)
        self.assertEqual(baseline[0].predicted_relations[0].label, "targets")
        self.assertEqual(hard[0].predicted_relations[0].label, "uses")
        self.assertEqual(
            baseline[0].predicted_relations[0].relation_score,
            hard[0].predicted_relations[0].relation_score,
        )
        self.assertEqual(inference.scored_pairs[0].type_logits, original_logits)

    def test_inverse_lookup_preserves_emitted_project_direction_and_label(self):
        rules = (
            {"project_label":"used-by","canonical_label":"uses","swap_endpoints":True,"status":"inverse"},
        )
        td, canonicalization, profile = self._schema(
            rules,
            resolved=("intrusion-set", "malware"),
            triples=(("intrusion-set", "uses", "malware"),),
        )
        self.addCleanup(td.cleanup)
        inference = self._inference("malware", "intrusion-set", (7.0,))
        hard = build_hard_profile_prediction_records(
            (inference,), ("used-by",), 0.5,
            canonicalization=canonicalization, profile=profile, **self._kwargs()
        )
        relation = hard[0].predicted_relations[0]
        self.assertEqual(relation.label, "used-by")
        self.assertEqual(relation.source.label, "malware")
        self.assertEqual(relation.target.label, "intrusion-set")

    def test_unresolved_endpoint_preserves_baseline_type_argmax(self):
        rules = (
            {"project_label":"targets","canonical_label":"targets","swap_endpoints":False,"status":"direct"},
            {"project_label":"uses","canonical_label":"uses","swap_endpoints":False,"status":"direct"},
        )
        td, canonicalization, profile = self._schema(
            rules,
            resolved=("malware",),
            unresolved=("tactic",),
        )
        self.addCleanup(td.cleanup)
        inference = self._inference("tactic", "malware", (9.0, 8.0))
        hard = build_hard_profile_prediction_records(
            (inference,), ("targets", "uses"), 0.5,
            canonicalization=canonicalization, profile=profile, **self._kwargs()
        )
        self.assertEqual(hard[0].predicted_relations[0].label, "targets")

    def test_all_blocked_types_emit_no_relation(self):
        rules = (
            {"project_label":"targets","canonical_label":"targets","swap_endpoints":False,"status":"direct"},
            {"project_label":"uses","canonical_label":"uses","swap_endpoints":False,"status":"direct"},
        )
        td, canonicalization, profile = self._schema(
            rules,
            resolved=("identity", "malware"),
        )
        self.addCleanup(td.cleanup)
        inference = self._inference("identity", "malware", (9.0, 8.0))
        hard = build_hard_profile_prediction_records(
            (inference,), ("targets", "uses"), 0.5,
            canonicalization=canonicalization, profile=profile, **self._kwargs()
        )
        self.assertEqual(hard[0].predicted_relations, ())
        self.assertEqual(len(hard[0].predicted_spans), 2)


if __name__ == "__main__":
    unittest.main()
