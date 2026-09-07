"""Learned pooling for the token context strictly between relation endpoints."""
from __future__ import annotations

import torch
from torch import nn


class BetweenSpanContextPooler(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        if hidden_size < 1 or output_dim < 1:
            raise ValueError("hidden_size and output_dim must be positive")
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        self.projection = nn.Linear(hidden_size, output_dim)
        self.empty_context = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        token_states: torch.Tensor,
        endpoint_spans: torch.Tensor,
    ) -> torch.Tensor:
        if token_states.ndim != 2 or token_states.shape[-1] != self.hidden_size:
            raise ValueError("token_states must have shape [tokens, hidden_size]")
        if endpoint_spans.ndim != 2 or endpoint_spans.shape[-1] != 4:
            raise ValueError(
                "endpoint_spans must have shape [pairs, 4] as "
                "[source_start, source_end, target_start, target_end]"
            )
        if endpoint_spans.numel() == 0:
            return token_states.new_empty((0, self.output_dim))

        endpoints = endpoint_spans.long()
        token_count = token_states.shape[0]
        source_start = endpoints[:, 0]
        source_end = endpoints[:, 1]
        target_start = endpoints[:, 2]
        target_end = endpoints[:, 3]

        invalid = (
            (source_start < 0)
            | (source_end < source_start)
            | (source_end >= token_count)
            | (target_start < 0)
            | (target_end < target_start)
            | (target_end >= token_count)
        )
        if bool(invalid.any()):
            index = int(torch.nonzero(invalid, as_tuple=False)[0, 0].item())
            values = endpoints[index].detach().cpu().tolist()
            raise ValueError(
                "invalid endpoint spans "
                f"[{values[0]}, {values[1]}] and "
                f"[{values[2]}, {values[3]}] for {token_count} tokens"
            )

        source_before_target = source_end < target_start
        target_before_source = target_end < source_start
        separated = source_before_target | target_before_source

        between_start = torch.where(
            source_before_target,
            source_end + 1,
            target_end + 1,
        )
        between_end = torch.where(
            source_before_target,
            target_start - 1,
            source_start - 1,
        )
        nonempty = separated & (between_start <= between_end)

        positions = torch.arange(token_count, device=token_states.device)
        between_mask = (
            nonempty.unsqueeze(1)
            & (positions.unsqueeze(0) >= between_start.unsqueeze(1))
            & (positions.unsqueeze(0) <= between_end.unsqueeze(1))
        )

        token_scores = self.attention(token_states).squeeze(-1)
        scores = token_scores.unsqueeze(0).expand(endpoints.shape[0], -1)
        masked_scores = scores.masked_fill(
            ~between_mask,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(masked_scores, dim=1)
        pooled = weights @ token_states
        projected = self.projection(pooled)

        empty = self.empty_context.unsqueeze(0).expand(endpoints.shape[0], -1)
        return torch.where(nonempty.unsqueeze(1), projected, empty)
