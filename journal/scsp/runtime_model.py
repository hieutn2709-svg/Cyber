"""Runtime composition of the pinned encoder and Gate A SpanPair heads."""
from __future__ import annotations

import re
from collections.abc import Iterator

import torch
from torch import nn

from .context_pooler import BetweenSpanContextPooler
from .model import PlainSpanPairHeads, SpanPooler

_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")


class GateASpanPairModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        hidden_size: int,
        num_entity_classes: int,
        num_relation_types: int,
        max_width: int,
        width_embedding_dim: int,
        context_dim: int,
        distance_embedding_dim: int,
        max_distance: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.span_pooler = SpanPooler(
            hidden_size=hidden_size,
            width_embedding_dim=width_embedding_dim,
            max_width=max_width,
        )
        self.context_pooler = BetweenSpanContextPooler(
            hidden_size=hidden_size,
            output_dim=context_dim,
        )
        self.heads = PlainSpanPairHeads(
            span_dim=self.span_pooler.output_dim,
            num_entity_types=num_entity_classes,
            context_dim=context_dim,
            distance_embedding_dim=distance_embedding_dim,
            max_distance=max_distance,
            num_relation_types=num_relation_types,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_name: str,
        revision: str,
        num_entity_classes: int,
        num_relation_types: int,
        max_width: int,
        width_embedding_dim: int,
        context_dim: int,
        distance_embedding_dim: int,
        max_distance: int,
    ) -> "GateASpanPairModel":
        if not _HEX40.fullmatch(revision):
            raise ValueError(
                "encoder revision must be an immutable 40-character commit hash"
            )
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "transformers is required for runtime encoder loading"
            ) from exc
        encoder = AutoModel.from_pretrained(model_name, revision=revision)
        hidden_size = int(encoder.config.hidden_size)
        return cls(
            encoder=encoder,
            hidden_size=hidden_size,
            num_entity_classes=num_entity_classes,
            num_relation_types=num_relation_types,
            max_width=max_width,
            width_embedding_dim=width_embedding_dim,
            context_dim=context_dim,
            distance_embedding_dim=distance_embedding_dim,
            max_distance=max_distance,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        states = outputs.last_hidden_state
        if states.ndim != 3 or states.shape[-1] != self.hidden_size:
            raise ValueError(
                "encoder last_hidden_state must have shape [batch, tokens, hidden]"
            )
        return states

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.encoder.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        encoder_ids = {id(parameter) for parameter in self.encoder.parameters()}
        for parameter in self.parameters():
            if id(parameter) not in encoder_ids:
                yield parameter
