"""Immutable configuration for the Gate A plain SpanPair experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GateAConfig:
    experiment_name: str
    schema_mode: str
    encoder_model: str
    encoder_revision: str
    split_seed: int
    seed: int
    fold: int
    span_width_coverage: float
    max_span_candidates: int
    min_entity_score: float
    max_relation_token_distance: int
    relation_negative_ratio: int
    output_dir: str

    @classmethod
    def from_json(cls, path: str | Path) -> "GateAConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**_required_fields(payload))
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_mode != "none":
            raise ValueError(
                "Gate A requires schema_mode='none'; schema constraints belong to Gate B"
            )
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if not self.encoder_model.strip():
            raise ValueError("encoder_model must be non-empty")
        if not self.encoder_revision.strip():
            raise ValueError("encoder_revision must be explicitly recorded")
        if self.fold < 1:
            raise ValueError("fold must be >= 1")
        if not 0.0 < self.span_width_coverage <= 1.0:
            raise ValueError("span_width_coverage must be in (0, 1]")
        if self.max_span_candidates < 1:
            raise ValueError("max_span_candidates must be >= 1")
        if not 0.0 <= self.min_entity_score <= 1.0:
            raise ValueError("min_entity_score must be in [0, 1]")
        if self.max_relation_token_distance < 0:
            raise ValueError("max_relation_token_distance must be >= 0")
        if self.relation_negative_ratio < 0:
            raise ValueError("relation_negative_ratio must be >= 0")


_REQUIRED = tuple(GateAConfig.__dataclass_fields__.keys())


def _required_fields(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in _REQUIRED if name not in payload]
    if missing:
        raise ValueError(f"missing Gate A config fields: {', '.join(missing)}")
    unexpected = sorted(set(payload) - set(_REQUIRED))
    if unexpected:
        raise ValueError(f"unexpected Gate A config fields: {', '.join(unexpected)}")
    return {name: payload[name] for name in _REQUIRED}
