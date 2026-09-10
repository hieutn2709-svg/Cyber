"""Pure probabilistic-profile utilities for Gate C.

Gate C is decoding-only. This module contains no training, split discovery,
or test-set selection logic.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .candidates import (
    local_to_global_candidate,
    local_to_global_gold_relation,
    local_to_global_gold_span,
)
from .schema import (
    CanonicalizationTable,
    TaskRelationshipProfile,
    canonicalize_relation,
)
from .serialization import PredictedRelation, PredictionRecord


@dataclass(frozen=True, slots=True)
class EntityTypePosterior:
    span_key: tuple[object, ...]
    entity_types: tuple[str, ...]
    conditional_probabilities: tuple[float, ...]
    none_probability: float
    entity_probability: float
    top1_entity_type: str


def conditional_non_none_posterior(
    probabilities: Sequence[float],
    entity_types: Sequence[str],
    *,
    span_key: tuple[object, ...],
    tolerance: float = 1e-8,
) -> EntityTypePosterior:
    """Condition a NONE+types probability vector on being a non-NONE entity."""
    probs = tuple(float(value) for value in probabilities)
    labels = tuple(str(label) for label in entity_types)

    if not labels:
        raise ValueError("entity_types must be non-empty")
    if len(probs) != len(labels) + 1:
        raise ValueError("probability vector must be NONE + entity types")
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")
    if any(value < -tolerance or value > 1.0 + tolerance for value in probs):
        raise ValueError("entity probabilities must lie in [0, 1]")
    if abs(sum(probs) - 1.0) > tolerance:
        raise ValueError("entity probabilities must sum to one")

    none_probability = probs[0]
    entity_probability = 1.0 - none_probability
    if entity_probability <= tolerance:
        raise ValueError("non-NONE mass is numerically degenerate")

    non_none = tuple(max(0.0, value) for value in probs[1:])
    non_none_sum = sum(non_none)
    if non_none_sum <= tolerance:
        raise ValueError("non-NONE mass is numerically degenerate")

    conditional = tuple(value / non_none_sum for value in non_none)
    top1_index = max(
        range(len(labels)),
        key=lambda index: (conditional[index], -index),
    )

    return EntityTypePosterior(
        span_key=span_key,
        entity_types=labels,
        conditional_probabilities=conditional,
        none_probability=float(none_probability),
        entity_probability=float(entity_probability),
        top1_entity_type=labels[top1_index],
    )


def probabilistic_compatibility(
    source: EntityTypePosterior,
    target: EntityTypePosterior,
    relation_types: Sequence[str],
    *,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
    tolerance: float = 1e-12,
) -> tuple[float, ...]:
    """Return C_ijr for each project relation label.

    Resolved endpoint/relation combinations use the frozen Gate B task profile.
    Unresolved endpoint types and unresolved relation labels are neutral (M=1),
    representing unknown compatibility rather than incompatibility.
    """
    labels = tuple(str(label) for label in relation_types)
    if not labels:
        raise ValueError("relation_types must be non-empty")
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")
    if source.entity_types != target.entity_types:
        raise ValueError("source and target posterior entity inventories must match")

    entity_types = source.entity_types
    known_entity_types = profile.resolved_entity_types | profile.unresolved_entity_types
    if set(entity_types) != known_entity_types:
        raise ValueError("posterior entity inventory must match task profile")

    for posterior in (source, target):
        if len(posterior.conditional_probabilities) != len(entity_types):
            raise ValueError("posterior probability count must match entity inventory")
        total = sum(posterior.conditional_probabilities)
        if abs(total - 1.0) > tolerance:
            raise ValueError("conditional entity posterior must sum to one")
        if any(
            value < -tolerance or value > 1.0 + tolerance
            for value in posterior.conditional_probabilities
        ):
            raise ValueError("conditional entity posterior must lie in [0, 1]")

    scores: list[float] = []
    for relation_label in labels:
        rule = canonicalize_relation(relation_label, canonicalization)
        score = 0.0
        for source_type, q_source in zip(
            entity_types,
            source.conditional_probabilities,
        ):
            for target_type, q_target in zip(
                entity_types,
                target.conditional_probabilities,
            ):
                if rule.status == "unresolved":
                    compatible = 1.0
                elif (
                    source_type in profile.unresolved_entity_types
                    or target_type in profile.unresolved_entity_types
                ):
                    compatible = 1.0
                else:
                    lookup_source, lookup_target = (
                        (target_type, source_type)
                        if rule.swap_endpoints
                        else (source_type, target_type)
                    )
                    compatible = float(
                        (lookup_source, rule.label, lookup_target)
                        in profile.allowed_triples
                    )
                score += float(q_source) * float(q_target) * compatible

        if score < -tolerance or score > 1.0 + tolerance:
            raise ValueError("probabilistic compatibility must lie in [0, 1]")
        scores.append(min(1.0, max(0.0, float(score))))

    return tuple(scores)


def adjust_relation_type_logits(
    raw_logits: Sequence[float],
    compatibility_scores: Sequence[float],
    *,
    beta: float,
    epsilon: float = 1e-8,
) -> tuple[float, ...]:
    """Apply the frozen probabilistic-profile penalty to relation-type logits."""
    logits = tuple(float(value) for value in raw_logits)
    scores = tuple(float(value) for value in compatibility_scores)

    if not logits or len(logits) != len(scores):
        raise ValueError(
            "relation logits and compatibility scores must align and be non-empty"
        )
    if beta < 0.0:
        raise ValueError("beta must be >= 0")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("compatibility scores must lie in [0, 1]")

    if beta == 0.0:
        return logits

    return tuple(
        logit + float(beta) * math.log(max(score, epsilon))
        for logit, score in zip(logits, scores)
    )


def select_relation_type(adjusted_logits: Sequence[float]) -> int:
    """Select the largest adjusted logit, breaking exact ties by inventory order."""
    values = tuple(float(value) for value in adjusted_logits)
    if not values:
        raise ValueError("adjusted logits must be non-empty")
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def build_probabilistic_prediction_records(
    inferences: Sequence[object],
    relation_types: Sequence[str],
    *,
    beta: float,
    threshold: float,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
    run_id: str,
    git_commit: str,
    dataset_sha256: str,
    config_sha256: str,
    fold: int,
    seed: int,
    split: str,
    epsilon: float = 1e-8,
) -> tuple[PredictionRecord, ...]:
    """Aggregate Gate C predictions while changing only relation-type decoding."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("relation threshold must lie in [0, 1]")
    relation_labels = tuple(str(label) for label in relation_types)
    if not relation_labels:
        raise ValueError("relation_types must be non-empty")

    gold_spans_by_doc = defaultdict(dict)
    pred_spans_by_doc = defaultdict(dict)
    gold_relations_by_doc = defaultdict(dict)
    pred_relations_by_doc = defaultdict(dict)

    for inference in inferences:
        base = inference.base
        window = base.window
        doc_id = window.doc_id

        for span in window.gold_spans:
            global_span = local_to_global_gold_span(window, span)
            gold_spans_by_doc[doc_id][global_span.key] = global_span

        for span in base.predicted_spans:
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

        for scored_pair in base.scored_pairs:
            probability = _sigmoid(scored_pair.existence_logit)
            if probability < threshold:
                continue
            if len(scored_pair.type_logits) != len(relation_labels):
                raise ValueError("relation logit count does not match inventory")

            source_key = scored_pair.pair.source.typed_key
            target_key = scored_pair.pair.target.typed_key
            missing_keys = tuple(
                key
                for key in (source_key, target_key)
                if key not in inference.posterior_by_typed_key
            )
            if missing_keys:
                raise ValueError(
                    "missing posterior for Gate C relation endpoint: "
                    f"{missing_keys!r}"
                )

            source_posterior = inference.posterior_by_typed_key[source_key]
            target_posterior = inference.posterior_by_typed_key[target_key]
            compatibility = probabilistic_compatibility(
                source_posterior,
                target_posterior,
                relation_labels,
                canonicalization=canonicalization,
                profile=profile,
            )
            adjusted = adjust_relation_type_logits(
                scored_pair.type_logits,
                compatibility,
                beta=beta,
                epsilon=epsilon,
            )
            selected_index = select_relation_type(adjusted)

            source = local_to_global_candidate(window, scored_pair.pair.source)
            target = local_to_global_candidate(window, scored_pair.pair.target)
            predicted = PredictedRelation(
                source=source,
                target=target,
                label=relation_labels[selected_index],
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
