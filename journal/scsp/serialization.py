"""Prediction artifact serialization for independently rescorable Gate A runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .pairs import GoldRelation
from .structures import GoldSpan, SpanCandidate


@dataclass(frozen=True, slots=True)
class PredictedRelation:
    source: SpanCandidate
    target: SpanCandidate
    label: str
    relation_score: float

    def __post_init__(self) -> None:
        if self.source.document_id != self.target.document_id:
            raise ValueError("predicted relation endpoints must belong to the same document")
        if not self.label:
            raise ValueError("predicted relation label must be non-empty")
        if not 0.0 <= self.relation_score <= 1.0:
            raise ValueError("relation_score must be in [0, 1]")

    @property
    def strict_key(self) -> tuple[object, ...]:
        return (
            self.source.document_id,
            self.source.start,
            self.source.end,
            self.source.label,
            self.label,
            self.target.start,
            self.target.end,
            self.target.label,
        )


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    run_id: str
    git_commit: str
    dataset_sha256: str
    config_sha256: str
    fold: int
    seed: int
    split: str
    document_id: str
    gold_spans: tuple[GoldSpan, ...]
    predicted_spans: tuple[SpanCandidate, ...]
    gold_relations: tuple[GoldRelation, ...]
    predicted_relations: tuple[PredictedRelation, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "git_commit",
            "dataset_sha256",
            "config_sha256",
            "document_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.fold < 1:
            raise ValueError("fold must be >= 1")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")

        if any(span.document_id != self.document_id for span in self.gold_spans):
            raise ValueError("gold span document mismatch")
        if any(span.document_id != self.document_id for span in self.predicted_spans):
            raise ValueError("predicted span document mismatch")
        if any(rel.document_id != self.document_id for rel in self.gold_relations):
            raise ValueError("gold relation document mismatch")
        if any(
            rel.source.document_id != self.document_id
            for rel in self.predicted_relations
        ):
            raise ValueError("predicted relation document mismatch")

        gold_keys = {span.key for span in self.gold_spans}
        for relation in self.gold_relations:
            if relation.source.key not in gold_keys or relation.target.key not in gold_keys:
                raise ValueError("gold relation endpoint is not present in gold_spans")

        predicted_keys = {span.typed_key for span in self.predicted_spans}
        for relation in self.predicted_relations:
            if (
                relation.source.typed_key not in predicted_keys
                or relation.target.typed_key not in predicted_keys
            ):
                raise ValueError(
                    "predicted relation endpoint is not present in predicted_spans"
                )


def write_prediction_jsonl(
    records: Iterable[PredictionRecord], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(_record_to_dict(record), sort_keys=True, ensure_ascii=False)
            )
            handle.write("\n")


def read_prediction_jsonl(path: str | Path) -> tuple[PredictionRecord, ...]:
    records: list[PredictionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(_record_from_dict(payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid prediction JSONL record at line {line_number}: {exc}"
                ) from exc
    return tuple(records)


def _record_to_dict(record: PredictionRecord) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "git_commit": record.git_commit,
        "dataset_sha256": record.dataset_sha256,
        "config_sha256": record.config_sha256,
        "fold": record.fold,
        "seed": record.seed,
        "split": record.split,
        "document_id": record.document_id,
        "gold_spans": [asdict(span) for span in record.gold_spans],
        "predicted_spans": [asdict(span) for span in record.predicted_spans],
        "gold_relations": [
            {
                "source": asdict(rel.source),
                "target": asdict(rel.target),
                "label": rel.label,
            }
            for rel in record.gold_relations
        ],
        "predicted_relations": [
            {
                "source": asdict(rel.source),
                "target": asdict(rel.target),
                "label": rel.label,
                "relation_score": rel.relation_score,
            }
            for rel in record.predicted_relations
        ],
    }


def _record_from_dict(payload: dict[str, Any]) -> PredictionRecord:
    gold_spans = tuple(GoldSpan(**item) for item in payload["gold_spans"])
    predicted_spans = tuple(
        SpanCandidate(**item) for item in payload["predicted_spans"]
    )
    gold_relations = tuple(
        GoldRelation(
            source=GoldSpan(**item["source"]),
            target=GoldSpan(**item["target"]),
            label=item["label"],
        )
        for item in payload["gold_relations"]
    )
    predicted_relations = tuple(
        PredictedRelation(
            source=SpanCandidate(**item["source"]),
            target=SpanCandidate(**item["target"]),
            label=item["label"],
            relation_score=float(item["relation_score"]),
        )
        for item in payload["predicted_relations"]
    )
    return PredictionRecord(
        run_id=str(payload["run_id"]),
        git_commit=str(payload["git_commit"]),
        dataset_sha256=str(payload["dataset_sha256"]),
        config_sha256=str(payload["config_sha256"]),
        fold=int(payload["fold"]),
        seed=int(payload["seed"]),
        split=str(payload["split"]),
        document_id=str(payload["document_id"]),
        gold_spans=gold_spans,
        predicted_spans=predicted_spans,
        gold_relations=gold_relations,
        predicted_relations=predicted_relations,
    )
