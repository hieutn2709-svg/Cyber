"""Independent strict rescoring of saved Gate A prediction artifacts."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from journal.scsp.serialization import PredictionRecord


def strict_micro_scores(
    records: Iterable[PredictionRecord],
    *,
    entity_labels: set[str] | frozenset[str] | None = None,
    relation_endpoint_labels: set[str] | frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Strict micro metrics with optional primary/core reporting scopes.

    Defaults preserve the original all-label evaluation behavior. The optional
    entity scope is used for the 10 primary Gate A entity types; the optional
    relation endpoint scope supports a core-to-core relation report without
    changing the all-evaluable relation metric.
    """
    entity_tp = entity_fp = entity_fn = 0
    relation_tp = relation_fp = relation_fn = 0

    entity_scope = set(entity_labels) if entity_labels is not None else None
    relation_scope = (
        set(relation_endpoint_labels)
        if relation_endpoint_labels is not None
        else None
    )

    for record in records:
        gold_entities = {
            span.key
            for span in record.gold_spans
            if entity_scope is None or span.label in entity_scope
        }
        pred_entities = {
            span.typed_key
            for span in record.predicted_spans
            if span.label is not None
            and (entity_scope is None or span.label in entity_scope)
        }
        entity_tp += len(gold_entities & pred_entities)
        entity_fp += len(pred_entities - gold_entities)
        entity_fn += len(gold_entities - pred_entities)

        gold_relations = {
            _gold_relation_key(relation)
            for relation in record.gold_relations
            if relation_scope is None
            or (
                relation.source.label in relation_scope
                and relation.target.label in relation_scope
            )
        }
        pred_relations = {
            relation.strict_key
            for relation in record.predicted_relations
            if relation_scope is None
            or (
                relation.source.label in relation_scope
                and relation.target.label in relation_scope
            )
        }
        relation_tp += len(gold_relations & pred_relations)
        relation_fp += len(pred_relations - gold_relations)
        relation_fn += len(gold_relations - pred_relations)

    return {
        "entity": _metric_block(entity_tp, entity_fp, entity_fn),
        "relation": _metric_block(relation_tp, relation_fp, relation_fn),
    }


def _gold_relation_key(relation: Any) -> tuple[object, ...]:
    return (
        relation.source.document_id,
        relation.source.start,
        relation.source.end,
        relation.source.label,
        relation.label,
        relation.target.start,
        relation.target.end,
        relation.target.label,
    )


def _metric_block(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
