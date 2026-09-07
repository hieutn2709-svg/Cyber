"""Immutable engineering hyperparameters for Gate A training."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GateATrainingConfig:
    encoder_lr: float
    head_lr: float
    max_epochs: int
    smoke_epochs: int
    early_stopping_patience: int
    gradient_accumulation_steps: int
    weight_decay: float
    max_grad_norm: float
    width_embedding_dim: int
    context_dim: int
    distance_embedding_dim: int
    entity_negative_ratio: int
    relation_existence_pos_weight: float
    focal_gamma: float
    entity_loss_weight: float
    relation_existence_loss_weight: float
    relation_type_loss_weight: float
    relation_inference_chunk_size: int
    fp16: bool
    threshold_grid: tuple[float, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "GateATrainingConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = set(cls.__dataclass_fields__)
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        if missing:
            raise ValueError(f"missing Gate A training fields: {', '.join(missing)}")
        if unexpected:
            raise ValueError(
                f"unexpected Gate A training fields: {', '.join(unexpected)}"
            )
        values: dict[str, Any] = dict(payload)
        values["threshold_grid"] = tuple(
            float(x) for x in payload["threshold_grid"]
        )
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.encoder_lr <= 0 or self.head_lr <= 0:
            raise ValueError("learning rates must be > 0")
        if self.max_epochs < 1 or self.smoke_epochs < 1:
            raise ValueError("epoch counts must be >= 1")
        if self.smoke_epochs > self.max_epochs:
            raise ValueError("smoke_epochs cannot exceed max_epochs")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0")
        if min(
            self.width_embedding_dim,
            self.context_dim,
            self.distance_embedding_dim,
            self.relation_inference_chunk_size,
        ) < 1:
            raise ValueError("embedding/chunk dimensions must be positive")
        if self.entity_negative_ratio < 0:
            raise ValueError("entity_negative_ratio must be >= 0")
        if self.relation_existence_pos_weight <= 0:
            raise ValueError("relation_existence_pos_weight must be > 0")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be >= 0")
        if min(
            self.entity_loss_weight,
            self.relation_existence_loss_weight,
            self.relation_type_loss_weight,
        ) < 0:
            raise ValueError("loss weights must be >= 0")
        if not self.threshold_grid:
            raise ValueError("threshold_grid must be non-empty")
        if any(not 0.0 < value < 1.0 for value in self.threshold_grid):
            raise ValueError("threshold_grid values must lie in (0, 1)")
        if tuple(sorted(set(self.threshold_grid))) != self.threshold_grid:
            raise ValueError("threshold_grid must be unique and sorted")
