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

        token_count = token_states.shape[0]
        outputs: list[torch.Tensor] = []
        for values in endpoint_spans.long():
            source_start, source_end, target_start, target_end = [
                int(value.item()) for value in values
            ]
            for start, end in (
                (source_start, source_end),
                (target_start, target_end),
            ):
                if start < 0 or end < start or end >= token_count:
                    raise ValueError(
                        f"invalid endpoint span [{start}, {end}] for "
                        f"{token_count} tokens"
                    )

            if source_end < target_start:
                between_start = source_end + 1
                between_end = target_start - 1
            elif target_end < source_start:
                between_start = target_end + 1
                between_end = source_start - 1
            else:
                between_start = 1
                between_end = 0

            if between_start > between_end:
                outputs.append(self.empty_context)
                continue

            segment = token_states[between_start : between_end + 1]
            weights = torch.softmax(
                self.attention(segment).squeeze(-1),
                dim=0,
            )
            pooled = torch.sum(segment * weights.unsqueeze(-1), dim=0)
            outputs.append(self.projection(pooled))

        return torch.stack(outputs, dim=0)
