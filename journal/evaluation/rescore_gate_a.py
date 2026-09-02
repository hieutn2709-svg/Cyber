"""Independent strict rescoring of saved Gate A prediction artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from journal.scsp.serialization import PredictionRecord


def strict_micro_scores(
    records: Iterable[PredictionRecord],
) -> dict[str, dict[str, Any]]:
    entity_tp = entity_fp = entity_fn = 0
    relation_tp = relation_fp = relation_fn = 0

    for record in records:
        gold_entities = {span.key for span in record.gold_spans}
        pred_entities = {
            span.typed_key
            for span in record.predicted_spans
            if span.label is not None
        }
        entity_tp += len(gold_entities & pred_entities)
        entity_fp += len(pred_entities - gold_entities)
        entity_fn += len(gold_entities - pred_entities)

        gold_relations = {
            _gold_relation_key(relation) for relation in record.gold_relations
        }
        pred_relations = {
            relation.strict_key for relation in record.predicted_relations
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
