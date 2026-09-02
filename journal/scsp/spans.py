"""Span proposal, pruning, and endpoint-recall diagnostics for Gate A."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from .structures import GoldSpan, SpanCandidate, SpanDiagnostics


def derive_width_cap(gold_spans: Sequence[GoldSpan], coverage: float = 0.995) -> int:
    if not gold_spans:
        raise ValueError("gold_spans must be non-empty")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")

    counts = Counter(span.width for span in gold_spans)
    target = len(gold_spans) * coverage
    cumulative = 0
    for width in sorted(counts):
        cumulative += counts[width]
        if cumulative >= target:
            return width
    return max(counts)


def enumerate_spans(
    token_count: int,
    max_width: int,
    document_id: str,
) -> tuple[SpanCandidate, ...]:
    if token_count < 0:
        raise ValueError("token_count must be >= 0")
    if max_width < 1:
        raise ValueError("max_width must be >= 1")
    spans: list[SpanCandidate] = []
    for start in range(token_count):
        for width in range(1, min(max_width, token_count - start) + 1):
            spans.append(SpanCandidate(document_id, start, start + width - 1))
    return tuple(spans)


def prune_span_candidates(
    candidates: Iterable[SpanCandidate],
    max_candidates: int,
    min_entity_score: float,
) -> tuple[SpanCandidate, ...]:
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if not 0.0 <= min_entity_score <= 1.0:
        raise ValueError("min_entity_score must be in [0, 1]")
    eligible = [c for c in candidates if c.entity_score >= min_entity_score]
    eligible.sort(
        key=lambda c: (
            -c.entity_score,
            c.document_id,
            c.start,
            c.end,
            c.label or "",
            c.proposal_source,
        )
    )
    return tuple(eligible[:max_candidates])


def span_recall(
    gold_spans: Sequence[GoldSpan],
    candidates: Iterable[SpanCandidate],
) -> float:
    if not gold_spans:
        return 1.0
    predicted = {candidate.typed_key for candidate in candidates if candidate.label is not None}
    matched = sum(gold.key in predicted for gold in gold_spans)
    return matched / len(gold_spans)


def span_diagnostics(
    gold_spans: Sequence[GoldSpan],
    candidates: Iterable[SpanCandidate],
) -> SpanDiagnostics:
    candidate_list = tuple(candidates)
    predicted = {c.typed_key for c in candidate_list if c.label is not None}
    matched = sum(gold.key in predicted for gold in gold_spans)

    totals_by_type: Counter[str] = Counter()
    matches_by_type: Counter[str] = Counter()
    totals_by_width: Counter[str] = Counter()
    matches_by_width: Counter[str] = Counter()

    for gold in gold_spans:
        totals_by_type[gold.label] += 1
        bucket = _width_bucket(gold.width)
        totals_by_width[bucket] += 1
        if gold.key in predicted:
            matches_by_type[gold.label] += 1
            matches_by_width[bucket] += 1

    by_type = tuple(
        (label, matches_by_type[label] / total)
        for label, total in sorted(totals_by_type.items())
    )
    bucket_order = ("1", "2-3", "4-7", "8+")
    by_width = tuple(
        (bucket, matches_by_width[bucket] / totals_by_width[bucket])
        for bucket in bucket_order
        if totals_by_width[bucket]
    )
    gold_count = len(gold_spans)
    return SpanDiagnostics(
        gold_count=gold_count,
        matched_count=matched,
        recall=(matched / gold_count) if gold_count else 1.0,
        recall_by_type=by_type,
        recall_by_width_bucket=by_width,
    )


def _width_bucket(width: int) -> str:
    if width == 1:
        return "1"
    if width <= 3:
        return "2-3"
    if width <= 7:
        return "4-7"
    return "8+"
