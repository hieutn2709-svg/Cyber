"""Pure probabilistic-profile utilities for Gate C.

Gate C is decoding-only. This module contains no training, split discovery,
or test-set selection logic.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .schema import (
    CanonicalizationTable,
    TaskRelationshipProfile,
    canonicalize_relation,
)


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
