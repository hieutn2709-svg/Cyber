"""Pure probabilistic-profile utilities for Gate C.

Gate C is decoding-only. This module contains no training, split discovery,
or test-set selection logic.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


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
