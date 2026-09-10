from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    import journal.scsp.gate_c_inference as gate_c_inference
    from journal.scsp.gate_c_inference import infer_gate_c_window
except ImportError:
    gate_c_inference = None
    infer_gate_c_window = None

from journal.scsp.data import LabelInventory, WindowExample
from journal.scsp.structures import SpanCandidate
from journal.scsp.training import WindowInference


class _FakeModel:
    def __init__(self, probabilities: list[list[float]]) -> None:
        self._logits = torch.log(torch.tensor(probabilities, dtype=torch.float64))
        self.heads = SimpleNamespace(entity_head=self._entity_head)

    def encode(self, input_ids, attention_mask):
        length = int(input_ids.shape[1])
        return (torch.zeros((length, 4), dtype=torch.float64),)

    def span_pooler(self, token_states, span_tensor):
        return torch.zeros((span_tensor.shape[0], 4), dtype=torch.float64)

    def _entity_head(self, span_representations):
        self.asserted_span_count = int(span_representations.shape[0])
        return self._logits


class GateCInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = LabelInventory(
            primary_entity_types=("malware", "tool"),
            auxiliary_entity_types=(),
            relation_types=("uses",),
        )
        self.window = WindowExample(
            doc_seq_index=0,
            doc_id="doc-1",
            window_index=0,
            token_start_global=0,
            token_end_global=2,
            input_ids=(101, 11, 102),
            attention_mask=(1, 1, 1),
            label_mask=(False, True, False),
            content_start=1,
            content_end=1,
            gold_spans=(),
            gold_relations=(),
            primary_entity_types=("malware", "tool"),
            auxiliary_entity_types=(),
        )
        self.base_config = SimpleNamespace()
        self.training_config = SimpleNamespace(relation_chunk_size=32)
        self.device = torch.device("cpu")

    def _assert_api(self) -> None:
        self.assertIsNotNone(
            gate_c_inference,
            "Gate C posterior-extraction module must exist",
        )
        self.assertIsNotNone(
            infer_gate_c_window,
            "Gate C posterior-extraction API must exist",
        )

    def _base(self, *, label: str = "malware", entity_score: float = 0.8) -> WindowInference:
        span = SpanCandidate(
            document_id="doc-1",
            start=1,
            end=1,
            label=label,
            entity_score=entity_score,
            proposal_source="predicted",
        )
        return WindowInference(
            window=self.window,
            predicted_spans=(span,),
            scored_pairs=(),
            proposal_gold_count=0,
            proposal_matched_count=0,
            post_pruning_typed_matched_count=0,
        )

    def _run(self, base: WindowInference, probabilities: list[list[float]]):
        model = _FakeModel(probabilities)
        with patch.object(gate_c_inference, "infer_window", return_value=base) as delegated:
            result = infer_gate_c_window(
                model,
                self.window,
                self.inventory,
                width_cap=4,
                base_config=self.base_config,
                training_config=self.training_config,
                device=self.device,
            )
        return result, delegated, model

    def test_delegates_authoritative_candidates_to_gate_a_and_extracts_posteriors(self) -> None:
        self._assert_api()
        base = self._base()
        result, delegated, model = self._run(base, [[0.20, 0.70, 0.10]])

        self.assertIs(result.base, base)
        delegated.assert_called_once()
        self.assertEqual(
            tuple(result.posterior_by_typed_key),
            tuple(span.typed_key for span in base.predicted_spans),
        )
        posterior = result.posterior_by_typed_key[base.predicted_spans[0].typed_key]
        self.assertEqual(posterior.top1_entity_type, "malware")
        self.assertAlmostEqual(posterior.entity_probability, 0.8, places=12)
        self.assertEqual(model.asserted_span_count, 1)

    def test_rejects_top1_parity_mismatch(self) -> None:
        self._assert_api()
        base = self._base(label="tool", entity_score=0.8)
        with self.assertRaisesRegex(ValueError, "top-1 parity"):
            self._run(base, [[0.20, 0.70, 0.10]])

    def test_rejects_entity_score_parity_mismatch(self) -> None:
        self._assert_api()
        base = self._base(label="malware", entity_score=0.7)
        with self.assertRaisesRegex(ValueError, "entity-score parity"):
            self._run(base, [[0.20, 0.70, 0.10]])


if __name__ == "__main__":
    unittest.main()
