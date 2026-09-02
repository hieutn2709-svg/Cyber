"""Neural heads for the no-schema Gate A SpanPair model.

The encoder is intentionally kept outside this module so unit tests can operate
on contextual token states without network access. Runtime code may obtain those
states from a pinned RoBERTa revision.
"""

from __future__ import annotations

import torch
from torch import nn


class SpanPooler(nn.Module):
    """Build [start; end; attention-pool; width-embedding] span vectors."""

    def __init__(self, hidden_size: int, width_embedding_dim: int, max_width: int) -> None:
        super().__init__()
        if hidden_size < 1 or width_embedding_dim < 1 or max_width < 1:
            raise ValueError("hidden_size, width_embedding_dim, and max_width must be positive")
        self.hidden_size = hidden_size
        self.max_width = max_width
        self.width_embedding = nn.Embedding(max_width + 1, width_embedding_dim)
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        self.output_dim = hidden_size * 3 + width_embedding_dim

    def forward(self, token_states: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        if token_states.ndim != 2:
            raise ValueError("token_states must have shape [tokens, hidden]")
        if spans.ndim != 2 or spans.shape[-1] != 2:
            raise ValueError("spans must have shape [num_spans, 2]")
        if token_states.shape[-1] != self.hidden_size:
            raise ValueError("token_states hidden dimension does not match hidden_size")
        if spans.numel() == 0:
            return token_states.new_empty((0, self.output_dim))

        reps: list[torch.Tensor] = []
        token_count = token_states.shape[0]
        for start_tensor, end_tensor in spans.long():
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            if start < 0 or end < start or end >= token_count:
                raise ValueError(f"invalid span [{start}, {end}] for {token_count} tokens")
            segment = token_states[start : end + 1]
            weights = torch.softmax(self.attention(segment).squeeze(-1), dim=0)
            pooled = torch.sum(segment * weights.unsqueeze(-1), dim=0)
            width = min(end - start + 1, self.max_width)
            width_index = torch.tensor(width, device=token_states.device)
            width_rep = self.width_embedding(width_index)
            reps.append(
                torch.cat(
                    (token_states[start], token_states[end], pooled, width_rep), dim=-1
                )
            )
        return torch.stack(reps, dim=0)


class SpanEntityHead(nn.Module):
    def __init__(self, span_dim: int, num_entity_types: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(span_dim, num_entity_types)

    def forward(self, span_representations: torch.Tensor) -> torch.Tensor:
        return self.classifier(span_representations)


class PairRepresentation(nn.Module):
    """Directional pair vector [s; t; s*t; |s-t|; context; distance]."""

    def __init__(
        self,
        span_dim: int,
        context_dim: int,
        distance_embedding_dim: int,
        max_distance: int,
    ) -> None:
        super().__init__()
        if min(span_dim, context_dim, distance_embedding_dim) < 1 or max_distance < 0:
            raise ValueError("pair representation dimensions must be positive")
        self.span_dim = span_dim
        self.context_dim = context_dim
        self.max_distance = max_distance
        self.distance_embedding = nn.Embedding(max_distance + 1, distance_embedding_dim)
        self.output_dim = span_dim * 4 + context_dim + distance_embedding_dim

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        context: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        if source.shape != target.shape:
            raise ValueError("source and target representations must have identical shapes")
        if source.ndim != 2 or source.shape[-1] != self.span_dim:
            raise ValueError("source/target must have shape [pairs, span_dim]")
        if context.ndim != 2 or context.shape[0] != source.shape[0] or context.shape[-1] != self.context_dim:
            raise ValueError("context must have shape [pairs, context_dim]")
        if distance.ndim != 1 or distance.shape[0] != source.shape[0]:
            raise ValueError("distance must have shape [pairs]")
        distance_index = distance.long().clamp(min=0, max=self.max_distance)
        distance_rep = self.distance_embedding(distance_index)
        return torch.cat(
            (source, target, source * target, torch.abs(source - target), context, distance_rep),
            dim=-1,
        )


class RelationExistenceHead(nn.Module):
    def __init__(self, pair_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(pair_dim, 1)

    def forward(self, pair_representations: torch.Tensor) -> torch.Tensor:
        return self.classifier(pair_representations).squeeze(-1)


class RelationTypeHead(nn.Module):
    def __init__(self, pair_dim: int, num_relation_types: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(pair_dim, num_relation_types)

    def forward(self, pair_representations: torch.Tensor) -> torch.Tensor:
        return self.classifier(pair_representations)


class PlainSpanPairHeads(nn.Module):
    """Gate A heads only; deliberately has no schema/compatibility argument."""

    def __init__(
        self,
        span_dim: int,
        num_entity_types: int,
        context_dim: int,
        distance_embedding_dim: int,
        max_distance: int,
        num_relation_types: int,
    ) -> None:
        super().__init__()
        self.entity_head = SpanEntityHead(span_dim, num_entity_types)
        self.pair_representation = PairRepresentation(
            span_dim,
            context_dim,
            distance_embedding_dim,
            max_distance,
        )
        self.existence_head = RelationExistenceHead(self.pair_representation.output_dim)
        self.type_head = RelationTypeHead(
            self.pair_representation.output_dim, num_relation_types
        )

    def forward(
        self,
        span_representations: torch.Tensor,
        source_representations: torch.Tensor,
        target_representations: torch.Tensor,
        context_representations: torch.Tensor,
        distances: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pair_representations = self.pair_representation(
            source_representations,
            target_representations,
            context_representations,
            distances,
        )
        return {
            "entity_logits": self.entity_head(span_representations),
            "relation_existence_logits": self.existence_head(pair_representations),
            "relation_type_logits": self.type_head(pair_representations),
            "pair_representations": pair_representations,
        }
