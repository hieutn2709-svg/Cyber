"""Ordered entity-pair candidates for Gate A relation modeling."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .structures import GoldSpan, SpanCandidate


@dataclass(frozen=True, slots=True)
class GoldRelation:
    source: GoldSpan
    target: GoldSpan
    label: str

    def __post_init__(self) -> None:
        if self.source.document_id != self.target.document_id:
            raise ValueError("relation endpoints must belong to the same document")
        if not self.label:
            raise ValueError("relation label must be non-empty")

    @property
    def document_id(self) -> str:
        return self.source.document_id

    @property
    def endpoint_key(self) -> tuple[object, ...]:
        return (
            self.document_id,
            self.source.start,
            self.source.end,
            self.source.label,
            self.target.start,
            self.target.end,
            self.target.label,
        )


@dataclass(frozen=True, slots=True)
class PairCandidate:
    source: SpanCandidate
    target: SpanCandidate
    token_distance: int

    def __post_init__(self) -> None:
        if self.source.document_id != self.target.document_id:
            raise ValueError("pair endpoints must belong to the same document")
        if self.token_distance < 0:
            raise ValueError("token_distance must be >= 0")

    @property
    def document_id(self) -> str:
        return self.source.document_id

    @property
    def ordered_key(self) -> tuple[object, ...]:
        return (
            self.document_id,
            self.source.start,
            self.source.end,
            self.source.label,
            self.target.start,
            self.target.end,
            self.target.label,
        )


def pair_distance(a: SpanCandidate | GoldSpan, b: SpanCandidate | GoldSpan) -> int:
    if a.document_id != b.document_id:
        raise ValueError("cannot compute pair distance across documents")
    if a.end < b.start:
        return b.start - a.end - 1
    if b.end < a.start:
        return a.start - b.end - 1
    return 0


def generate_ordered_pairs(
    spans: Sequence[SpanCandidate],
    max_token_distance: int | None = None,
) -> tuple[PairCandidate, ...]:
    if max_token_distance is not None and max_token_distance < 0:
        raise ValueError("max_token_distance must be >= 0 or None")

    pairs: list[PairCandidate] = []
    for source_index, source in enumerate(spans):
        for target_index, target in enumerate(spans):
            if source_index == target_index:
                continue
            if source.document_id != target.document_id:
                continue
            distance = pair_distance(source, target)
            if max_token_distance is not None and distance > max_token_distance:
                continue
            pairs.append(PairCandidate(source, target, distance))
    return tuple(pairs)
