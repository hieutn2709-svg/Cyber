"""Training/inference candidate construction for Gate A.

All span coordinates inside the model remain window-local. Conversion to
controlled-corpus document-global coordinates happens only when artifacts are
serialized, preventing collisions between multiple windows from one document.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .data import WindowExample
from .pairs import GoldRelation, PairCandidate, pair_distance
from .structures import GoldSpan, SpanCandidate


@dataclass(frozen=True, slots=True)
class TrainingPairBatch:
    pairs: tuple[PairCandidate, ...]
    positive_mask: tuple[bool, ...]
    relation_labels: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.pairs)
            == len(self.positive_mask)
            == len(self.relation_labels)
        ):
            raise ValueError("pair batch fields must have identical length")


def enumerate_content_spans(
    window: WindowExample,
    max_width: int,
) -> tuple[SpanCandidate, ...]:
    """Enumerate spans only inside labelable content, excluding special tokens."""
    if max_width < 1:
        raise ValueError("max_width must be >= 1")
    spans: list[SpanCandidate] = []
    for start in range(window.content_start, window.content_end + 1):
        max_end = min(window.content_end, start + max_width - 1)
        for end in range(start, max_end + 1):
            spans.append(SpanCandidate(window.doc_id, start, end))
    return tuple(spans)


def local_to_global_candidate(
    window: WindowExample,
    span: SpanCandidate,
) -> SpanCandidate:
    if span.document_id != window.doc_id:
        raise ValueError("span/window document mismatch")
    if span.start < window.content_start or span.end > window.content_end:
        raise ValueError("local span falls outside window content bounds")
    offset = window.token_start_global - window.content_start
    global_start = span.start + offset
    global_end = span.end + offset
    if global_start < window.token_start_global or global_end >= window.token_end_global:
        raise ValueError("converted global span falls outside global window bounds")
    return SpanCandidate(
        document_id=span.document_id,
        start=global_start,
        end=global_end,
        label=span.label,
        entity_score=span.entity_score,
        proposal_source=span.proposal_source,
    )


def local_to_global_gold_span(
    window: WindowExample,
    span: GoldSpan,
) -> GoldSpan:
    candidate = local_to_global_candidate(
        window,
        SpanCandidate(span.document_id, span.start, span.end, span.label),
    )
    return GoldSpan(candidate.document_id, candidate.start, candidate.end, span.label)


def local_to_global_gold_relation(
    window: WindowExample,
    relation: GoldRelation,
) -> GoldRelation:
    return GoldRelation(
        local_to_global_gold_span(window, relation.source),
        local_to_global_gold_span(window, relation.target),
        relation.label,
    )


def inject_gold_spans(
    pruned: Sequence[SpanCandidate],
    gold_spans: Sequence[GoldSpan],
) -> tuple[SpanCandidate, ...]:
    """Training-only teacher forcing: ensure every typed gold span is available."""
    result = list(pruned)
    present = {candidate.typed_key for candidate in result}
    for gold in gold_spans:
        if gold.key in present:
            continue
        result.append(
            SpanCandidate(
                document_id=gold.document_id,
                start=gold.start,
                end=gold.end,
                label=gold.label,
                entity_score=1.0,
                proposal_source="gold_injected",
            )
        )
        present.add(gold.key)
    return tuple(result)


def build_training_pairs(
    spans: Sequence[SpanCandidate],
    gold_relations: Sequence[GoldRelation],
    max_token_distance: int,
    negative_ratio: int,
    *,
    seed: int,
    diagnostics: dict | None = None,
    negative_policy: str = "typed",
) -> TrainingPairBatch:
    """Preserve all positives; sample deterministic nearby negatives.

    Gold positives are retained even when their distance exceeds the inference
    cutoff. This is training-only supervision and never changes validation/test
    candidate generation.
    """
    if max_token_distance < 0:
        raise ValueError("max_token_distance must be >= 0")
    if negative_ratio < 0:
        raise ValueError("negative_ratio must be >= 0")
    if negative_policy not in {"typed", "boundary_replace"}:
        raise ValueError(f"unsupported negative policy: {negative_policy}")

    gold_by_endpoint = {
        relation.endpoint_key: relation.label for relation in gold_relations
    }
    positives: list[PairCandidate] = []
    negatives: list[PairCandidate] = []

    for source_index, source in enumerate(spans):
        for target_index, target in enumerate(spans):
            if source_index == target_index:
                continue
            if source.document_id != target.document_id:
                continue
            if source.start == target.start and source.end == target.end:
                continue
            distance = pair_distance(source, target)
            pair = PairCandidate(source, target, distance)
            if pair.ordered_key in gold_by_endpoint:
                positives.append(pair)
            elif distance <= max_token_distance:
                negatives.append(pair)

    eligible_negatives = negatives
    rng = random.Random(seed)
    negative_limit = (
        len(positives) * negative_ratio if positives else negative_ratio
    )
    if len(negatives) > negative_limit:
        chosen_indices = sorted(rng.sample(range(len(negatives)), negative_limit))
        negatives = [negatives[index] for index in chosen_indices]

    policy_record = None
    baseline_negatives = negatives
    if negative_policy == "boundary_replace":
        # Repair only the original draw: unaffected windows retain exact batches.
        # Direction and document matter; endpoint types intentionally do not.
        def boundary(pair):
            return (pair.document_id, pair.source.start, pair.source.end,
                    pair.target.start, pair.target.end)

        positive_boundaries = {boundary(pair) for pair in positives}
        drawn = (chosen_indices if len(eligible_negatives) > negative_limit
                 else list(range(len(eligible_negatives))))
        drawn_set = set(drawn)
        safe = [i for i, pair in enumerate(eligible_negatives)
                if boundary(pair) not in positive_boundaries]
        safe_set = set(safe)
        retained = [i for i in drawn if i in safe_set]
        removed = len(drawn) - len(retained)
        remaining = [i for i in safe if i not in drawn_set]
        replacement = rng.sample(remaining, min(removed, len(remaining))) if removed else []
        negatives = [eligible_negatives[i] for i in sorted(retained + replacement)]
        policy_record = {
            "name": negative_policy,
            "baseline_sampled_negatives": len(drawn),
            "removed_sampled_negatives": removed,
            "replacement_negatives": len(replacement),
            "negative_budget_shortfall": removed - len(replacement),
            "safe_eligible_negatives": len(safe),
        }

    pairs = positives + negatives
    positive_mask = tuple([True] * len(positives) + [False] * len(negatives))
    relation_labels = tuple(
        [gold_by_endpoint[pair.ordered_key] for pair in positives]
        + [None] * len(negatives)
    )
    batch = TrainingPairBatch(tuple(pairs), positive_mask, relation_labels)
    if diagnostics is not None:
        from .training_pair_diagnostics import summarize_training_pairs
        diagnostics.update(summarize_training_pairs(
            positives, eligible_negatives, batch, gold_relations,
            max_token_distance=max_token_distance,
        ))
        if policy_record is not None:
            baseline = TrainingPairBatch(tuple(positives + baseline_negatives),
                tuple([True] * len(positives) + [False] * len(baseline_negatives)),
                tuple([gold_by_endpoint[p.ordered_key] for p in positives]
                      + [None] * len(baseline_negatives)))
            original = summarize_training_pairs(positives, eligible_negatives,
                baseline, gold_relations, max_token_distance=max_token_distance)
            policy_record["baseline_sampled_batch_sha256"] = original["sampled_batch_sha256"]
            diagnostics["negative_policy"] = policy_record
    return batch
