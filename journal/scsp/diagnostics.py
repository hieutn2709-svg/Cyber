"""Candidate-pair bottleneck diagnostics for Gate A."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .pairs import GoldRelation, PairCandidate, pair_distance


@dataclass(frozen=True, slots=True)
class PairDiagnostics:
    gold_relation_count: int
    matched_relation_count: int
    recall: float
    candidate_count: int
    positive_candidate_count: int
    negative_candidate_count: int
    positive_to_negative_ratio: float | None
    recall_by_relation_type: tuple[tuple[str, float], ...]
    recall_by_distance_bucket: tuple[tuple[str, float], ...]


def pair_recall(
    gold_relations: Sequence[GoldRelation],
    pairs: Iterable[PairCandidate],
) -> float:
    if not gold_relations:
        return 1.0
    available = {pair.ordered_key for pair in pairs}
    matched = sum(relation.endpoint_key in available for relation in gold_relations)
    return matched / len(gold_relations)


def pair_diagnostics(
    gold_relations: Sequence[GoldRelation],
    pairs: Iterable[PairCandidate],
) -> PairDiagnostics:
    pair_list = tuple(pairs)
    available = {pair.ordered_key for pair in pair_list}
    gold_endpoint_keys = {relation.endpoint_key for relation in gold_relations}

    matched = sum(relation.endpoint_key in available for relation in gold_relations)
    positive_candidate_count = sum(pair.ordered_key in gold_endpoint_keys for pair in pair_list)
    negative_candidate_count = len(pair_list) - positive_candidate_count

    totals_by_type: Counter[str] = Counter()
    matches_by_type: Counter[str] = Counter()
    totals_by_distance: Counter[str] = Counter()
    matches_by_distance: Counter[str] = Counter()

    for relation in gold_relations:
        totals_by_type[relation.label] += 1
        bucket = _distance_bucket(pair_distance(relation.source, relation.target))
        totals_by_distance[bucket] += 1
        if relation.endpoint_key in available:
            matches_by_type[relation.label] += 1
            matches_by_distance[bucket] += 1

    by_type = tuple(
        (label, matches_by_type[label] / total)
        for label, total in sorted(totals_by_type.items())
    )
    bucket_order = ("0-16", "17-32", "33-64", "65-96", "97+")
    by_distance = tuple(
        (bucket, matches_by_distance[bucket] / totals_by_distance[bucket])
        for bucket in bucket_order
        if totals_by_distance[bucket]
    )

    ratio = (
        positive_candidate_count / negative_candidate_count
        if negative_candidate_count
        else None
    )
    gold_count = len(gold_relations)
    return PairDiagnostics(
        gold_relation_count=gold_count,
        matched_relation_count=matched,
        recall=(matched / gold_count) if gold_count else 1.0,
        candidate_count=len(pair_list),
        positive_candidate_count=positive_candidate_count,
        negative_candidate_count=negative_candidate_count,
        positive_to_negative_ratio=ratio,
        recall_by_relation_type=by_type,
        recall_by_distance_bucket=by_distance,
    )


def _distance_bucket(distance: int) -> str:
    if distance <= 16:
        return "0-16"
    if distance <= 32:
        return "17-32"
    if distance <= 64:
        return "33-64"
    if distance <= 96:
        return "65-96"
    return "97+"
