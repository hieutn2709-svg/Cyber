"""RoBERTa/BiGRU model with separate sequence-tag and role heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import RobertaModel


class MultiHeadSelfAttention(nn.Module):
    """Self-attention that supports BiGRU widths not divisible by head count."""

    def __init__(self, input_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if num_heads < 1 or input_dim < num_heads:
            raise ValueError("Attention requires at least one dimension per head")
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.query = nn.Linear(input_dim, self.inner_dim)
        self.key = nn.Linear(input_dim, self.inner_dim)
        self.value = nn.Linear(input_dim, self.inner_dim)
        self.output = nn.Linear(self.inner_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = values.shape
        return values.view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, values: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._split_heads(self.query(values))
        key = self._split_heads(self.key(values))
        value = self._split_heads(self.value(values))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            key_mask = ~attention_mask.bool()[:, None, None, :]
            scores = scores.masked_fill(key_mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous()
        batch, sequence, _, _ = attended.shape
        attended = attended.view(batch, sequence, self.inner_dim)
        return self.output(attended), weights


class CyberEntRelModel(nn.Module):
    def __init__(self, num_labels: int, num_roles: int, model_config: dict):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_config["encoder"])
        roberta_hidden_size = self.roberta.config.hidden_size
        bigru_config = model_config["bigru"]

        self.bigru = nn.GRU(
            input_size=roberta_hidden_size,
            hidden_size=bigru_config["dimension"],
            num_layers=bigru_config["num_layers"],
            bidirectional=True,
            batch_first=True,
            dropout=bigru_config["dropout"] if bigru_config["num_layers"] > 1 else 0.0,
        )
        gru_hidden_dim = bigru_config["dimension"] * 2
        self.attention = MultiHeadSelfAttention(
            gru_hidden_dim,
            bigru_config["attention_heads"],
            model_config["dropout"],
        )
        projection_size = model_config["hidden_layer_neurons"]
        self.projection = nn.Linear(gru_hidden_dim, projection_size)
        self.dropout = nn.Dropout(model_config["dropout"])
        self.layer_norm = nn.LayerNorm(projection_size)

        self.tag_classifier = nn.Linear(projection_size, num_labels)
        self.role_classifier = nn.Linear(projection_size, num_roles)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        role_labels: torch.Tensor | None = None,
    ):
        encoded = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        contextual, _ = self.bigru(encoded)
        contextual, _ = self.attention(contextual, attention_mask)
        features = self.layer_norm(
            self.dropout(torch.relu(self.projection(contextual)))
        )
        tag_emissions = self.tag_classifier(features)
        role_logits = self.role_classifier(features)

        if labels is not None:
            if role_labels is None:
                raise ValueError("role_labels are required when sequence labels are provided")
            tag_loss = -self.crf(
                tag_emissions,
                labels,
                mask=attention_mask.bool(),
                reduction="token_mean",
            )
            role_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                role_logits.reshape(-1, role_logits.shape[-1]),
                role_labels.reshape(-1),
            )
            return (0.8 * tag_loss) + (0.2 * role_loss)

        tag_predictions = self.crf.decode(
            tag_emissions, mask=attention_mask.bool()
        )
        role_predictions = torch.argmax(role_logits, dim=-1)
        return tag_predictions, role_predictions
