"""Pure utilities used by the Gate A training loop."""
from __future__ import annotations

import random
from collections.abc import Sequence

import torch

from .structures import GoldSpan, SpanCandidate


def build_entity_targets(
    candidates: Sequence[SpanCandidate],
    gold_spans: Sequence[GoldSpan],
    entity_types: Sequence[str],
) -> tuple[int, ...]:
    label_to_id = {
        label: index + 1 for index, label in enumerate(entity_types)
    }
    gold_by_coords = {
        (gold.document_id, gold.start, gold.end): gold.label
        for gold in gold_spans
    }
    targets: list[int] = []
    for candidate in candidates:
        label = gold_by_coords.get(
            (candidate.document_id, candidate.start, candidate.end)
        )
        if label is None:
            targets.append(0)
            continue
        try:
            targets.append(label_to_id[label])
        except KeyError as exc:
            raise ValueError(
                f"gold entity label absent from inventory: {label}"
            ) from exc
    return tuple(targets)


def sample_entity_training_indices(
    candidates: Sequence[SpanCandidate],
    gold_spans: Sequence[GoldSpan],
    *,
    negative_ratio: int,
    seed: int,
) -> tuple[int, ...]:
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be >= 0")
    gold_coords = {
        (gold.document_id, gold.start, gold.end) for gold in gold_spans
    }
    positives = [
        index
        for index, candidate in enumerate(candidates)
        if (
            candidate.document_id,
            candidate.start,
            candidate.end,
        )
        in gold_coords
    ]
    negatives = [
        index
        for index, candidate in enumerate(candidates)
        if (
            candidate.document_id,
            candidate.start,
            candidate.end,
        )
        not in gold_coords
    ]
    limit = len(positives) * negative_ratio if positives else negative_ratio
    if len(negatives) > limit:
        rng = random.Random(seed)
        negatives = sorted(rng.sample(negatives, limit))
    return tuple(sorted(positives + negatives))


def scored_entity_candidates(
    candidates: Sequence[SpanCandidate],
    logits: torch.Tensor,
    entity_types: Sequence[str],
) -> tuple[SpanCandidate, ...]:
    if logits.ndim != 2 or logits.shape[0] != len(candidates):
        raise ValueError("entity logits must have shape [candidates, classes]")
    if logits.shape[1] != len(entity_types) + 1:
        raise ValueError(
            "entity logits class count must be NONE + entity types"
        )
    if not candidates:
        return ()
    probabilities = torch.softmax(logits.detach(), dim=-1)
    non_none = probabilities[:, 1:]
    best_scores, best_indices = torch.max(non_none, dim=-1)
    result: list[SpanCandidate] = []
    for candidate, score, type_index in zip(
        candidates,
        best_scores.tolist(),
        best_indices.tolist(),
    ):
        result.append(
            SpanCandidate(
                document_id=candidate.document_id,
                start=candidate.start,
                end=candidate.end,
                label=str(entity_types[type_index]),
                entity_score=float(score),
                proposal_source="predicted",
            )
        )
    return tuple(result)
