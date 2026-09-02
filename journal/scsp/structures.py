"""Small immutable records shared by Gate A span logic."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_span(start: int, end: int) -> None:
    if start < 0:
        raise ValueError("span start must be >= 0")
    if end < start:
        raise ValueError("span end must be >= start")


@dataclass(frozen=True, slots=True)
class GoldSpan:
    document_id: str
    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        _validate_span(self.start, self.end)
        if not self.document_id:
            raise ValueError("document_id must be non-empty")
        if not self.label:
            raise ValueError("gold span label must be non-empty")

    @property
    def width(self) -> int:
        return self.end - self.start + 1

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (self.document_id, self.start, self.end, self.label)


@dataclass(frozen=True, slots=True)
class SpanCandidate:
    document_id: str
    start: int
    end: int
    label: str | None = None
    entity_score: float = 0.0
    proposal_source: str = "enumerated"

    def __post_init__(self) -> None:
        _validate_span(self.start, self.end)
        if not self.document_id:
            raise ValueError("document_id must be non-empty")
        if not 0.0 <= self.entity_score <= 1.0:
            raise ValueError("entity_score must be in [0, 1]")
        if not self.proposal_source:
            raise ValueError("proposal_source must be non-empty")

    @property
    def width(self) -> int:
        return self.end - self.start + 1

    @property
    def typed_key(self) -> tuple[str, int, int, str | None]:
        return (self.document_id, self.start, self.end, self.label)


@dataclass(frozen=True, slots=True)
class SpanDiagnostics:
    gold_count: int
    matched_count: int
    recall: float
    recall_by_type: tuple[tuple[str, float], ...]
    recall_by_width_bucket: tuple[tuple[str, float], ...]
