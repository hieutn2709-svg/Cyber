from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

try:
    from journal.scsp.gate_c_artifacts import (
        build_gate_c_scored_pair_rows,
        build_gate_c_span_rows,
        write_gate_c_scored_pair_jsonl,
        write_gate_c_span_jsonl,
    )
except ImportError:
    build_gate_c_scored_pair_rows = None
    build_gate_c_span_rows = None
    write_gate_c_scored_pair_jsonl = None
    write_gate_c_span_jsonl = None

try:
    from journal.scsp.gate_c_artifacts import build_gate_c_diagnostics
except ImportError:
    build_gate_c_diagnostics = None

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


class GateCArtifactTests(unittest.TestCase):
    RELATION_TYPES = ("uses", "targets")
    ENTITY_TYPES = ("intrusion-set", "identity")

    def setUp(self) -> None:
        self.window = WindowExample(
            doc_seq_index=0,
            doc_id="doc-1",
            window_index=2,
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
            entity_score=0.8,
            proposal_source="predicted",
        )
        self.target = SpanCandidate(
            document_id="doc-1",
            start=3,
            end=3,
            label="identity",
            entity_score=0.9,
            proposal_source="predicted",
        )
        pair = PairCandidate(self.source, self.target, token_distance=1)
        self.scored_pair = ScoredRelationPair(
            pair=pair,
            existence_logit=3.0,
            type_logits=(5.0, 4.0),
        )
        base = WindowInference(
            window=self.window,
            predicted_spans=(self.source, self.target),
            scored_pairs=(self.scored_pair,),
            proposal_gold_count=0,
            proposal_matched_count=0,
            post_pruning_typed_matched_count=0,
        )
        self.source_posterior = conditional_non_none_posterior(
            [0.2, 0.7, 0.1],
            self.ENTITY_TYPES,
            span_key=self.source.typed_key,
        )
        self.target_posterior = conditional_non_none_posterior(
            [0.1, 0.2, 0.7],
            self.ENTITY_TYPES,
            span_key=self.target.typed_key,
        )
        self.inference = GateCWindowInference(
            base=base,
            posterior_by_typed_key=MappingProxyType(
                {
                    self.source.typed_key: self.source_posterior,
                    self.target.typed_key: self.target_posterior,
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

    def _assert_api(self) -> None:
        for api in (
            build_gate_c_span_rows,
            write_gate_c_span_jsonl,
            build_gate_c_scored_pair_rows,
            write_gate_c_scored_pair_jsonl,
        ):
            self.assertIsNotNone(api, "Gate C artifact API must exist")

    def _diagnostics(self, inference=None):
        self.assertIsNotNone(
            build_gate_c_diagnostics,
            "Gate C prediction-side diagnostics API must exist",
        )
        selected = self.inference if inference is None else inference
        return build_gate_c_diagnostics(
            (selected,),
            self.RELATION_TYPES,
            beta=2.0,
            threshold=0.90,
            canonicalization=self.canonicalization,
            profile=self.profile,
        )

    def test_span_rows_have_exact_schema_and_normalized_posterior(self) -> None:
        self._assert_api()
        rows = build_gate_c_span_rows((self.inference,))
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            set(rows[0]),
            {
                "document_id",
                "window_index",
                "start",
                "end",
                "top1_entity_type",
                "entity_probability",
                "none_probability",
                "entity_types",
                "conditional_non_none_posterior",
            },
        )
        self.assertEqual(rows[0]["document_id"], "doc-1")
        self.assertEqual(rows[0]["window_index"], 2)
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (1, 1))
        self.assertAlmostEqual(
            sum(rows[0]["conditional_non_none_posterior"]), 1.0, places=12
        )
        self.assertEqual((rows[1]["start"], rows[1]["end"]), (3, 3))

    def test_pair_row_has_exact_schema_and_recomputable_adjusted_logits(self) -> None:
        self._assert_api()
        rows = build_gate_c_scored_pair_rows(
            (self.inference,),
            self.RELATION_TYPES,
            beta=2.0,
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(
            set(row),
            {
                "document_id",
                "window_index",
                "source",
                "target",
                "token_distance",
                "existence_logit",
                "relation_types",
                "raw_relation_type_logits",
                "compatibility_scores",
                "beta",
                "epsilon",
                "adjusted_relation_type_logits",
                "selected_index",
                "selected_project_label",
            },
        )
        self.assertEqual(row["relation_types"], list(self.RELATION_TYPES))
        self.assertEqual(row["raw_relation_type_logits"], [5.0, 4.0])
        self.assertEqual(row["beta"], 2.0)
        self.assertEqual(row["epsilon"], 1e-8)

        from journal.scsp.gate_c import adjust_relation_type_logits

        expected = adjust_relation_type_logits(
            row["raw_relation_type_logits"],
            row["compatibility_scores"],
            beta=row["beta"],
            epsilon=row["epsilon"],
        )
        self.assertEqual(row["adjusted_relation_type_logits"], list(expected))
        self.assertEqual(
            row["selected_project_label"],
            self.RELATION_TYPES[row["selected_index"]],
        )

    def test_jsonl_writers_are_byte_deterministic_and_strict_json(self) -> None:
        self._assert_api()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            span_a = root / "span-a.jsonl"
            span_b = root / "span-b.jsonl"
            pair_a = root / "pair-a.jsonl"
            pair_b = root / "pair-b.jsonl"

            write_gate_c_span_jsonl(span_a, (self.inference,))
            write_gate_c_span_jsonl(span_b, (self.inference,))
            self.assertEqual(span_a.read_bytes(), span_b.read_bytes())

            kwargs = dict(
                beta=2.0,
                canonicalization=self.canonicalization,
                profile=self.profile,
            )
            write_gate_c_scored_pair_jsonl(
                pair_a, (self.inference,), self.RELATION_TYPES, **kwargs
            )
            write_gate_c_scored_pair_jsonl(
                pair_b, (self.inference,), self.RELATION_TYPES, **kwargs
            )
            self.assertEqual(pair_a.read_bytes(), pair_b.read_bytes())

            for path in (span_a, pair_a):
                for line in path.read_text(encoding="utf-8").splitlines():
                    payload = json.loads(line)
                    self.assertIsInstance(payload, dict)
                    self.assertNotIn("NaN", line)
                    self.assertNotIn("Infinity", line)
                    self.assertEqual(
                        line,
                        json.dumps(
                            payload,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                    )

    def test_diagnostics_have_required_prediction_side_fields_and_counts(self) -> None:
        diagnostics = self._diagnostics()
        required = {
            "retained_span_count",
            "posterior_top1_parity_mismatch_count",
            "posterior_max_normalization_deviation",
            "entity_score_parity_max_abs_difference",
            "compatibility_summary_by_project_relation",
            "mean_unresolved_source_posterior_mass",
            "mean_unresolved_target_posterior_mass",
            "candidate_relation_count",
            "argmax_change_count_vs_gate_a",
            "argmax_change_rate_vs_gate_a",
            "above_threshold_pair_count",
            "above_threshold_argmax_change_count_vs_gate_a",
            "transition_counts",
            "emitted_relation_count",
        }
        self.assertTrue(required.issubset(diagnostics))
        self.assertEqual(diagnostics["retained_span_count"], 2)
        self.assertEqual(diagnostics["posterior_top1_parity_mismatch_count"], 0)
        self.assertAlmostEqual(
            diagnostics["posterior_max_normalization_deviation"], 0.0, places=12
        )
        self.assertAlmostEqual(
            diagnostics["entity_score_parity_max_abs_difference"], 0.0, places=12
        )
        self.assertEqual(diagnostics["candidate_relation_count"], 1)
        self.assertEqual(diagnostics["argmax_change_count_vs_gate_a"], 1)
        self.assertEqual(diagnostics["argmax_change_rate_vs_gate_a"], 1.0)
        self.assertEqual(diagnostics["above_threshold_pair_count"], 1)
        self.assertEqual(
            diagnostics["above_threshold_argmax_change_count_vs_gate_a"], 1
        )
        self.assertEqual(diagnostics["transition_counts"], {"uses->targets": 1})
        self.assertEqual(diagnostics["emitted_relation_count"], 1)
        self.assertEqual(
            set(diagnostics["compatibility_summary_by_project_relation"]),
            set(self.RELATION_TYPES),
        )
        self.assertEqual(
            diagnostics["compatibility_summary_by_project_relation"]["uses"]["count"],
            1,
        )
        self.assertAlmostEqual(
            diagnostics["compatibility_summary_by_project_relation"]["uses"]["mean"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            diagnostics["mean_unresolved_source_posterior_mass"], 0.0, places=12
        )
        self.assertAlmostEqual(
            diagnostics["mean_unresolved_target_posterior_mass"], 0.0, places=12
        )

    def test_diagnostics_do_not_read_gold_fields(self) -> None:
        class PredictionOnlyWindow:
            doc_id = "doc-1"
            window_index = 2

            @property
            def gold_spans(self):
                raise AssertionError("diagnostics must not read gold_spans")

            @property
            def gold_relations(self):
                raise AssertionError("diagnostics must not read gold_relations")

        prediction_only = SimpleNamespace(
            base=SimpleNamespace(
                window=PredictionOnlyWindow(),
                predicted_spans=(self.source, self.target),
                scored_pairs=(self.scored_pair,),
            ),
            posterior_by_typed_key=MappingProxyType(
                {
                    self.source.typed_key: self.source_posterior,
                    self.target.typed_key: self.target_posterior,
                }
            ),
        )
        diagnostics = self._diagnostics(prediction_only)
        self.assertEqual(diagnostics["retained_span_count"], 2)
        self.assertEqual(diagnostics["candidate_relation_count"], 1)


if __name__ == "__main__":
    unittest.main()
