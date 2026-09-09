from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.data import WindowExample
from journal.scsp.gate_b_artifacts import (
    build_hard_profile_diagnostics,
    write_hard_profile_scored_pair_jsonl,
)
from journal.scsp.pairs import PairCandidate
from journal.scsp.schema import (
    load_relation_canonicalization,
    load_task_relationship_profile,
)
from journal.scsp.structures import SpanCandidate
from journal.scsp.training import ScoredRelationPair, WindowInference


class GateBArtifactTests(unittest.TestCase):
    def _fixtures(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        canonicalization_payload = {
            "version": 1,
            "rules": [
                {"project_label":"targets","canonical_label":"targets","swap_endpoints":False,"status":"direct"},
                {"project_label":"used-in","canonical_label":None,"swap_endpoints":False,"status":"unresolved"},
                {"project_label":"uses","canonical_label":"uses","swap_endpoints":False,"status":"direct"},
            ],
        }
        profile_payload = {
            "version": 1,
            "resolved_entity_types": ["intrusion-set", "malware"],
            "unresolved_entity_types": [],
            "allowed_triples": [
                {"source_type":"intrusion-set","relation_type":"uses","target_type":"malware","source_note":"test"}
            ],
        }
        cpath = root / "c.json"
        ppath = root / "p.json"
        cpath.write_text(json.dumps(canonicalization_payload), encoding="utf-8")
        ppath.write_text(json.dumps(profile_payload), encoding="utf-8")
        canonicalization = load_relation_canonicalization(cpath)
        profile = load_task_relationship_profile(ppath)

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
        source = SpanCandidate("doc-1", 1, 1, "intrusion-set", 0.9)
        target = SpanCandidate("doc-1", 2, 2, "malware", 0.8)
        scored = ScoredRelationPair(
            PairCandidate(source, target, 0),
            existence_logit=10.0,
            type_logits=(9.0, 7.0, 8.0),
        )
        inference = WindowInference(
            window=window,
            predicted_spans=(source, target),
            scored_pairs=(scored,),
            proposal_gold_count=0,
            proposal_matched_count=0,
            post_pruning_typed_matched_count=0,
        )
        return td, root, canonicalization, profile, inference

    def test_scored_pair_artifact_is_complete_strict_json_and_byte_deterministic(self):
        td, root, canonicalization, profile, inference = self._fixtures()
        self.addCleanup(td.cleanup)
        labels = ("targets", "used-in", "uses")
        first = root / "first.jsonl"
        second = root / "second.jsonl"
        rows1 = write_hard_profile_scored_pair_jsonl(
            first, (inference,), labels,
            canonicalization=canonicalization, profile=profile,
        )
        rows2 = write_hard_profile_scored_pair_jsonl(
            second, (inference,), labels,
            canonicalization=canonicalization, profile=profile,
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(rows1, rows2)
        row = json.loads(first.read_text(encoding="utf-8").strip())
        required = {
            "document_id", "window_index", "source", "target", "token_distance",
            "existence_logit", "relation_type_logits", "relation_types",
            "compatibility", "masked_relation_type_logits", "selected_index",
            "selected_project_label",
        }
        self.assertEqual(set(row), required)
        self.assertNotIn("gold_spans", row)
        self.assertNotIn("gold_relations", row)
        self.assertEqual(row["compatibility"], [False, None, True])
        self.assertEqual(row["masked_relation_type_logits"][0], "-inf")
        self.assertEqual(row["selected_project_label"], "uses")

    def test_profile_diagnostics_report_only_prediction_side_decisions(self):
        td, _, canonicalization, profile, inference = self._fixtures()
        self.addCleanup(td.cleanup)
        diagnostics = build_hard_profile_diagnostics(
            (inference,),
            ("targets", "used-in", "uses"),
            0.5,
            canonicalization=canonicalization,
            profile=profile,
        )
        self.assertEqual(diagnostics["candidate_relation_count"], 1)
        self.assertEqual(diagnostics["resolved_compatible_class_opportunities"], 1)
        self.assertEqual(diagnostics["resolved_blocked_class_opportunities"], 1)
        self.assertEqual(diagnostics["unresolved_relation_label_opportunities"], 1)
        self.assertEqual(diagnostics["emitted_relation_count"], 1)
        self.assertEqual(diagnostics["emitted_task_profile_compatible_count"], 1)
        self.assertEqual(diagnostics["emitted_blocked_count"], 0)
        self.assertEqual(diagnostics["counts_by_project_relation_label"], {"uses": 1})
        self.assertEqual(diagnostics["counts_by_canonical_relation_label"], {"uses": 1})
        self.assertEqual(diagnostics["compatibility_rate_resolved_emitted"], 1.0)


if __name__ == "__main__":
    unittest.main()
