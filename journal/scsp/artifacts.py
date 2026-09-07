"""Decode and aggregate Gate A window inference into rescorable artifacts."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from .candidates import (
    local_to_global_candidate,
    local_to_global_gold_relation,
    local_to_global_gold_span,
)
from .serialization import PredictedRelation, PredictionRecord
from .training import WindowInference


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def build_prediction_records(
    inferences: Sequence[WindowInference],
    relation_types: Sequence[str],
    threshold: float,
    *,
    run_id: str,
    git_commit: str,
    dataset_sha256: str,
    config_sha256: str,
    fold: int,
    seed: int,
    split: str,
) -> tuple[PredictionRecord, ...]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("relation threshold must lie in [0, 1]")
    if not relation_types:
        raise ValueError("relation_types must be non-empty")

    gold_spans_by_doc = defaultdict(dict)
    pred_spans_by_doc = defaultdict(dict)
    gold_relations_by_doc = defaultdict(dict)
    pred_relations_by_doc = defaultdict(dict)

    for inference in inferences:
        window = inference.window
        doc_id = window.doc_id
        for span in window.gold_spans:
            global_span = local_to_global_gold_span(window, span)
            gold_spans_by_doc[doc_id][global_span.key] = global_span
        for span in inference.predicted_spans:
            global_span = local_to_global_candidate(window, span)
            existing = pred_spans_by_doc[doc_id].get(global_span.typed_key)
            if existing is None or global_span.entity_score > existing.entity_score:
                pred_spans_by_doc[doc_id][global_span.typed_key] = global_span
        for relation in window.gold_relations:
            global_relation = local_to_global_gold_relation(window, relation)
            strict_key = (
                global_relation.source.document_id,
                global_relation.source.start,
                global_relation.source.end,
                global_relation.source.label,
                global_relation.label,
                global_relation.target.start,
                global_relation.target.end,
                global_relation.target.label,
            )
            gold_relations_by_doc[doc_id][strict_key] = global_relation

        for scored_pair in inference.scored_pairs:
            probability = _sigmoid(scored_pair.existence_logit)
            if probability < threshold:
                continue
            if len(scored_pair.type_logits) != len(relation_types):
                raise ValueError("relation logit count does not match inventory")
            type_index = max(
                range(len(scored_pair.type_logits)),
                key=lambda index: scored_pair.type_logits[index],
            )
            source = local_to_global_candidate(window, scored_pair.pair.source)
            target = local_to_global_candidate(window, scored_pair.pair.target)
            predicted = PredictedRelation(
                source=source,
                target=target,
                label=str(relation_types[type_index]),
                relation_score=probability,
            )
            existing = pred_relations_by_doc[doc_id].get(predicted.strict_key)
            if existing is None or predicted.relation_score > existing.relation_score:
                pred_relations_by_doc[doc_id][predicted.strict_key] = predicted

    doc_ids = sorted(
        set(gold_spans_by_doc)
        | set(pred_spans_by_doc)
        | set(gold_relations_by_doc)
        | set(pred_relations_by_doc)
    )
    records: list[PredictionRecord] = []
    for doc_id in doc_ids:
        gold_spans = tuple(
            sorted(
                gold_spans_by_doc[doc_id].values(),
                key=lambda span: (span.start, span.end, span.label),
            )
        )
        predicted_spans = tuple(
            sorted(
                pred_spans_by_doc[doc_id].values(),
                key=lambda span: (span.start, span.end, span.label or ""),
            )
        )
        gold_relations = tuple(
            relation
            for _, relation in sorted(gold_relations_by_doc[doc_id].items())
        )
        predicted_relations = tuple(
            relation
            for _, relation in sorted(pred_relations_by_doc[doc_id].items())
        )
        records.append(
            PredictionRecord(
                run_id=run_id,
                git_commit=git_commit,
                dataset_sha256=dataset_sha256,
                config_sha256=config_sha256,
                fold=fold,
                seed=seed,
                split=split,
                document_id=doc_id,
                gold_spans=gold_spans,
                predicted_spans=predicted_spans,
                gold_relations=gold_relations,
                predicted_relations=predicted_relations,
            )
        )
    return tuple(records)
