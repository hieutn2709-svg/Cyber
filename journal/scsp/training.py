"""Core training/inference operations for Gate A Plain SpanPair.

This module contains no split discovery and no test-set model selection. The CLI
is responsible for supplying document-disjoint windows. Threshold selection is
explicitly validation-only.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .candidates import (
    build_training_pairs,
    enumerate_content_spans,
    inject_gold_spans,
)
from .data import LabelInventory, WindowExample
from .losses import focal_binary_cross_entropy, positive_relation_type_loss
from .pairs import PairCandidate, generate_ordered_pairs
from .spans import prune_span_candidates
from .structures import SpanCandidate
from .training_utils import (
    build_entity_targets,
    sample_entity_training_indices,
    scored_entity_candidates,
)


@dataclass(frozen=True, slots=True)
class TrainingWindowResult:
    total_loss: torch.Tensor
    entity_loss: torch.Tensor
    relation_existence_loss: torch.Tensor
    relation_type_loss: torch.Tensor
    relation_positive_count: int
    relation_pair_count: int
    proposal_gold_count: int
    proposal_matched_count: int
    pruned_span_count: int


@dataclass(frozen=True, slots=True)
class ScoredRelationPair:
    pair: PairCandidate
    existence_logit: float
    type_logits: tuple[float, ...]

    @property
    def existence_probability(self) -> float:
        return float(
            torch.sigmoid(torch.tensor(self.existence_logit)).item()
        )


@dataclass(frozen=True, slots=True)
class WindowInference:
    window: WindowExample
    predicted_spans: tuple[SpanCandidate, ...]
    scored_pairs: tuple[ScoredRelationPair, ...]
    proposal_gold_count: int
    proposal_matched_count: int
    post_pruning_typed_matched_count: int


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    relation_f1: float
    scores: tuple[tuple[float, float], ...]


def _span_tensor(
    spans: Sequence[SpanCandidate],
    device: torch.device,
) -> torch.Tensor:
    if not spans:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.tensor(
        [[span.start, span.end] for span in spans],
        dtype=torch.long,
        device=device,
    )


def _window_tensors(
    window: WindowExample,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [window.input_ids], dtype=torch.long, device=device
    )
    attention_mask = torch.tensor(
        [window.attention_mask], dtype=torch.long, device=device
    )
    return input_ids, attention_mask


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def compute_training_window_loss(
    model,
    window: WindowExample,
    inventory: LabelInventory,
    *,
    width_cap: int,
    base_config,
    train_config,
    seed: int,
    device: torch.device,
) -> TrainingWindowResult:
    """Compute one teacher-forced training-window loss.

    Entity negatives and relation negatives are sampled deterministically. Gold
    spans are injected only into the relation-training candidate set, never into
    inference.
    """
    input_ids, attention_mask = _window_tensors(window, device)
    token_states = model.encode(input_ids, attention_mask)[0]

    proposals = enumerate_content_spans(window, width_cap)
    proposal_coords = {
        (candidate.document_id, candidate.start, candidate.end)
        for candidate in proposals
    }
    proposal_matched = sum(
        (gold.document_id, gold.start, gold.end) in proposal_coords
        for gold in window.gold_spans
    )

    proposal_tensor = _span_tensor(proposals, device)
    proposal_reps = model.span_pooler(token_states, proposal_tensor)
    entity_logits = model.heads.entity_head(proposal_reps)
    entity_types = inventory.trainable_entity_types
    targets = build_entity_targets(proposals, window.gold_spans, entity_types)
    sampled_indices = sample_entity_training_indices(
        proposals,
        window.gold_spans,
        negative_ratio=train_config.entity_negative_ratio,
        seed=seed,
    )
    if sampled_indices:
        index_tensor = torch.tensor(
            sampled_indices, dtype=torch.long, device=device
        )
        target_tensor = torch.tensor(
            [targets[index] for index in sampled_indices],
            dtype=torch.long,
            device=device,
        )
        entity_loss = F.cross_entropy(
            entity_logits[index_tensor], target_tensor
        )
    else:
        entity_loss = _zero_loss(entity_logits)

    scored = scored_entity_candidates(proposals, entity_logits, entity_types)
    pruned = prune_span_candidates(
        scored,
        max_candidates=base_config.max_span_candidates,
        min_entity_score=base_config.min_entity_score,
    )

    relation_spans = inject_gold_spans(pruned, window.gold_spans)
    pair_batch = build_training_pairs(
        relation_spans,
        window.gold_relations,
        max_token_distance=base_config.max_relation_token_distance,
        negative_ratio=base_config.relation_negative_ratio,
        seed=seed,
    )

    if pair_batch.pairs:
        relation_span_tensor = _span_tensor(relation_spans, device)
        relation_span_reps = model.span_pooler(
            token_states, relation_span_tensor
        )
        index_by_key = {
            span.typed_key: index
            for index, span in enumerate(relation_spans)
        }
        source_indices = torch.tensor(
            [
                index_by_key[pair.source.typed_key]
                for pair in pair_batch.pairs
            ],
            dtype=torch.long,
            device=device,
        )
        target_indices = torch.tensor(
            [
                index_by_key[pair.target.typed_key]
                for pair in pair_batch.pairs
            ],
            dtype=torch.long,
            device=device,
        )
        endpoint_tensor = torch.tensor(
            [
                [
                    pair.source.start,
                    pair.source.end,
                    pair.target.start,
                    pair.target.end,
                ]
                for pair in pair_batch.pairs
            ],
            dtype=torch.long,
            device=device,
        )
        context = model.context_pooler(token_states, endpoint_tensor)
        distances = torch.tensor(
            [pair.token_distance for pair in pair_batch.pairs],
            dtype=torch.long,
            device=device,
        )
        pair_reps = model.heads.pair_representation(
            relation_span_reps[source_indices],
            relation_span_reps[target_indices],
            context,
            distances,
        )
        existence_logits = model.heads.existence_head(pair_reps)
        type_logits = model.heads.type_head(pair_reps)
        existence_targets = torch.tensor(
            pair_batch.positive_mask,
            dtype=existence_logits.dtype,
            device=device,
        )
        existence_loss = focal_binary_cross_entropy(
            existence_logits,
            existence_targets,
            pos_weight=train_config.relation_existence_pos_weight,
            gamma=train_config.focal_gamma,
        )
        relation_to_id = {
            label: index
            for index, label in enumerate(inventory.relation_types)
        }
        type_targets = torch.tensor(
            [
                relation_to_id[label] if label is not None else 0
                for label in pair_batch.relation_labels
            ],
            dtype=torch.long,
            device=device,
        )
        positive_mask = torch.tensor(
            pair_batch.positive_mask,
            dtype=torch.bool,
            device=device,
        )
        type_loss = positive_relation_type_loss(
            type_logits, type_targets, positive_mask
        )
    else:
        existence_loss = _zero_loss(entity_logits)
        type_loss = _zero_loss(entity_logits)

    total = (
        train_config.entity_loss_weight * entity_loss
        + train_config.relation_existence_loss_weight * existence_loss
        + train_config.relation_type_loss_weight * type_loss
    )
    return TrainingWindowResult(
        total_loss=total,
        entity_loss=entity_loss,
        relation_existence_loss=existence_loss,
        relation_type_loss=type_loss,
        relation_positive_count=sum(pair_batch.positive_mask),
        relation_pair_count=len(pair_batch.pairs),
        proposal_gold_count=len(window.gold_spans),
        proposal_matched_count=proposal_matched,
        pruned_span_count=len(pruned),
    )


def infer_window(
    model,
    window: WindowExample,
    inventory: LabelInventory,
    *,
    width_cap: int,
    base_config,
    relation_chunk_size: int,
    device: torch.device,
) -> WindowInference:
    """Run no-gold-injection inference for one controlled-corpus window."""
    if relation_chunk_size < 1:
        raise ValueError("relation_chunk_size must be >= 1")
    input_ids, attention_mask = _window_tensors(window, device)
    with torch.no_grad():
        token_states = model.encode(input_ids, attention_mask)[0]
        proposals = enumerate_content_spans(window, width_cap)
        proposal_coords = {
            (candidate.document_id, candidate.start, candidate.end)
            for candidate in proposals
        }
        proposal_matched = sum(
            (gold.document_id, gold.start, gold.end) in proposal_coords
            for gold in window.gold_spans
        )
        proposal_reps = model.span_pooler(
            token_states, _span_tensor(proposals, device)
        )
        entity_logits = model.heads.entity_head(proposal_reps)
        scored = scored_entity_candidates(
            proposals,
            entity_logits,
            inventory.trainable_entity_types,
        )
        pruned = prune_span_candidates(
            scored,
            max_candidates=base_config.max_span_candidates,
            min_entity_score=base_config.min_entity_score,
        )
        typed_gold = {gold.key for gold in window.gold_spans}
        post_pruning_matched = sum(
            candidate.typed_key in typed_gold for candidate in pruned
        )

        pairs = generate_ordered_pairs(
            pruned,
            max_token_distance=base_config.max_relation_token_distance,
        )
        if not pairs:
            return WindowInference(
                window=window,
                predicted_spans=pruned,
                scored_pairs=(),
                proposal_gold_count=len(window.gold_spans),
                proposal_matched_count=proposal_matched,
                post_pruning_typed_matched_count=post_pruning_matched,
            )

        span_reps = model.span_pooler(
            token_states, _span_tensor(pruned, device)
        )
        index_by_key = {
            span.typed_key: index for index, span in enumerate(pruned)
        }
        scored_pairs: list[ScoredRelationPair] = []
        for start in range(0, len(pairs), relation_chunk_size):
            chunk = pairs[start : start + relation_chunk_size]
            source_indices = torch.tensor(
                [index_by_key[pair.source.typed_key] for pair in chunk],
                dtype=torch.long,
                device=device,
            )
            target_indices = torch.tensor(
                [index_by_key[pair.target.typed_key] for pair in chunk],
                dtype=torch.long,
                device=device,
            )
            endpoint_tensor = torch.tensor(
                [
                    [
                        pair.source.start,
                        pair.source.end,
                        pair.target.start,
                        pair.target.end,
                    ]
                    for pair in chunk
                ],
                dtype=torch.long,
                device=device,
            )
            context = model.context_pooler(token_states, endpoint_tensor)
            distances = torch.tensor(
                [pair.token_distance for pair in chunk],
                dtype=torch.long,
                device=device,
            )
            pair_reps = model.heads.pair_representation(
                span_reps[source_indices],
                span_reps[target_indices],
                context,
                distances,
            )
            existence_logits = model.heads.existence_head(pair_reps)
            type_logits = model.heads.type_head(pair_reps)
            for pair, existence_logit, relation_logits in zip(
                chunk,
                existence_logits.detach().cpu().tolist(),
                type_logits.detach().cpu().tolist(),
            ):
                scored_pairs.append(
                    ScoredRelationPair(
                        pair=pair,
                        existence_logit=float(existence_logit),
                        type_logits=tuple(
                            float(x) for x in relation_logits
                        ),
                    )
                )

    return WindowInference(
        window=window,
        predicted_spans=pruned,
        scored_pairs=tuple(scored_pairs),
        proposal_gold_count=len(window.gold_spans),
        proposal_matched_count=proposal_matched,
        post_pruning_typed_matched_count=post_pruning_matched,
    )


def _gold_relation_strict_key(relation) -> tuple[object, ...]:
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


def _predicted_relation_key(
    scored_pair: ScoredRelationPair,
    relation_types: Sequence[str],
) -> tuple[object, ...]:
    if len(scored_pair.type_logits) != len(relation_types):
        raise ValueError("relation logit count does not match inventory")
    type_index = max(
        range(len(scored_pair.type_logits)),
        key=lambda index: scored_pair.type_logits[index],
    )
    pair = scored_pair.pair
    return (
        pair.source.document_id,
        pair.source.start,
        pair.source.end,
        pair.source.label,
        relation_types[type_index],
        pair.target.start,
        pair.target.end,
        pair.target.label,
    )


def select_relation_threshold(
    inferences: Sequence[WindowInference],
    thresholds: Sequence[float],
    *,
    split: str,
    relation_types: Sequence[str] | None = None,
) -> ThresholdSelection:
    if split != "validation":
        raise ValueError("relation threshold selection is validation-only")
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    if relation_types is None:
        if inferences:
            raise ValueError(
                "relation_types are required for threshold selection"
            )
        scores = tuple((float(t), 0.0) for t in thresholds)
        best = min(scores, key=lambda item: (-item[1], item[0]))
        return ThresholdSelection(best[0], best[1], scores)

    scored: list[tuple[float, float]] = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for inference in inferences:
            gold = {
                _gold_relation_strict_key(relation)
                for relation in inference.window.gold_relations
            }
            predicted = {
                _predicted_relation_key(pair, relation_types)
                for pair in inference.scored_pairs
                if pair.existence_probability >= threshold
            }
            tp += len(gold & predicted)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        scored.append((float(threshold), float(f1)))
    best = min(scored, key=lambda item: (-item[1], item[0]))
    return ThresholdSelection(best[0], best[1], tuple(scored))
