"""Verified clean-window adapter for the SCSP Gate A controlled corpus."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .pairs import GoldRelation
from .structures import GoldSpan


@dataclass(frozen=True, slots=True)
class LabelInventory:
    primary_entity_types: tuple[str, ...]
    auxiliary_entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "LabelInventory":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelInventory":
        expected = {
            "primary_entity_types",
            "auxiliary_entity_types",
            "relation_types",
        }
        missing = expected - set(payload)
        extra = set(payload) - expected
        if missing:
            raise ValueError(f"missing inventory fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"unexpected inventory fields: {sorted(extra)}")
        primary = _clean_labels(
            payload["primary_entity_types"], "primary_entity_types"
        )
        auxiliary = _clean_labels(
            payload["auxiliary_entity_types"], "auxiliary_entity_types"
        )
        relations = _clean_labels(payload["relation_types"], "relation_types")
        overlap = set(primary) & set(auxiliary)
        if overlap:
            raise ValueError(
                f"primary/auxiliary entity type overlap: {sorted(overlap)}"
            )
        return cls(primary, auxiliary, relations)

    @property
    def trainable_entity_types(self) -> tuple[str, ...]:
        return self.primary_entity_types + self.auxiliary_entity_types

    def is_primary(self, label: str) -> bool:
        return label in self.primary_entity_types

    def is_auxiliary(self, label: str) -> bool:
        return label in self.auxiliary_entity_types


@dataclass(frozen=True, slots=True)
class WindowExample:
    doc_seq_index: int
    doc_id: str
    window_index: int
    token_start_global: int
    token_end_global: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    label_mask: tuple[bool, ...]
    content_start: int
    content_end: int
    gold_spans: tuple[GoldSpan, ...]
    gold_relations: tuple[GoldRelation, ...]
    primary_entity_types: tuple[str, ...]
    auxiliary_entity_types: tuple[str, ...]

    @property
    def primary_gold_spans(self) -> tuple[GoldSpan, ...]:
        labels = set(self.primary_entity_types)
        return tuple(
            span for span in self.gold_spans if span.label in labels
        )

    @property
    def auxiliary_gold_spans(self) -> tuple[GoldSpan, ...]:
        labels = set(self.auxiliary_entity_types)
        return tuple(
            span for span in self.gold_spans if span.label in labels
        )


def load_clean_windows(
    dataset_path: str | Path,
    inventory: LabelInventory,
) -> tuple[WindowExample, ...]:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(
            "controlled clean-window dataset must be a JSON list"
        )

    windows: list[WindowExample] = []
    seen_window_keys: set[tuple[int, int]] = set()
    doc_identity: dict[int, str] = {}
    trainable = set(inventory.trainable_entity_types)
    relation_types = set(inventory.relation_types)

    for row_index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"dataset row {row_index} must be an object")
        required = {
            "doc_seq_index",
            "doc_id",
            "window_index",
            "token_start_global",
            "token_end_global",
            "input_ids",
            "attention_mask",
            "label_mask",
            "entity_spans",
            "relations",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(
                f"dataset row {row_index} missing fields: {sorted(missing)}"
            )

        doc_seq_index = int(raw["doc_seq_index"])
        doc_id = str(raw["doc_id"])
        window_index = int(raw["window_index"])
        key = (doc_seq_index, window_index)
        if key in seen_window_keys:
            raise ValueError(f"duplicate document/window key: {key}")
        seen_window_keys.add(key)
        previous_doc_id = doc_identity.setdefault(doc_seq_index, doc_id)
        if previous_doc_id != doc_id:
            raise ValueError(
                f"doc_seq_index {doc_seq_index} maps to multiple doc_ids: "
                f"{previous_doc_id!r} and {doc_id!r}"
            )

        input_ids = tuple(int(value) for value in raw["input_ids"])
        attention_mask = tuple(int(value) for value in raw["attention_mask"])
        label_mask = tuple(bool(value) for value in raw["label_mask"])
        if not (
            len(input_ids) == len(attention_mask) == len(label_mask)
        ):
            raise ValueError(
                f"vector length mismatch in document {doc_id} "
                f"window {window_index}: input_ids={len(input_ids)} "
                f"attention_mask={len(attention_mask)} "
                f"label_mask={len(label_mask)}"
            )
        if not input_ids:
            raise ValueError(
                f"empty input_ids in document {doc_id} window {window_index}"
            )
        true_positions = [
            index for index, flag in enumerate(label_mask) if flag
        ]
        if not true_positions:
            raise ValueError(
                f"label_mask has no content tokens in document {doc_id} "
                f"window {window_index}"
            )
        content_start = true_positions[0]
        content_end = true_positions[-1]
        if true_positions != list(range(content_start, content_end + 1)):
            raise ValueError(
                "label_mask content tokens must be contiguous in document "
                f"{doc_id} window {window_index}"
            )

        entity_by_id: dict[str, GoldSpan] = {}
        gold_spans: list[GoldSpan] = []
        for raw_entity in raw["entity_spans"]:
            entity_id = str(raw_entity["entity_id"])
            label = str(raw_entity["type"])
            if entity_id in entity_by_id:
                raise ValueError(
                    f"duplicate entity_id {entity_id!r} in document "
                    f"{doc_id} window {window_index}"
                )
            if label not in trainable:
                raise ValueError(
                    f"entity type {label!r} is absent from the Gate A "
                    "trainable inventory"
                )
            span = GoldSpan(
                document_id=doc_id,
                start=int(raw_entity["token_start"]),
                end=int(raw_entity["token_end"]),
                label=label,
            )
            if span.start < content_start or span.end > content_end:
                raise ValueError(
                    f"entity {entity_id!r} falls outside content-token bounds "
                    f"[{content_start}, {content_end}]"
                )
            entity_by_id[entity_id] = span
            gold_spans.append(span)

        gold_relations: list[GoldRelation] = []
        for raw_relation in raw["relations"]:
            source_id = str(raw_relation["source_id"])
            target_id = str(raw_relation["target_id"])
            label = str(raw_relation["type"])
            if label not in relation_types:
                raise ValueError(
                    f"relation type {label!r} is absent from Gate A inventory"
                )
            if (
                source_id not in entity_by_id
                or target_id not in entity_by_id
            ):
                raise ValueError(
                    "relation endpoint missing from entity_spans in document "
                    f"{doc_id} window {window_index}: "
                    f"{source_id!r}->{target_id!r}"
                )
            gold_relations.append(
                GoldRelation(
                    entity_by_id[source_id],
                    entity_by_id[target_id],
                    label,
                )
            )

        windows.append(
            WindowExample(
                doc_seq_index=doc_seq_index,
                doc_id=doc_id,
                window_index=window_index,
                token_start_global=int(raw["token_start_global"]),
                token_end_global=int(raw["token_end_global"]),
                input_ids=input_ids,
                attention_mask=attention_mask,
                label_mask=label_mask,
                content_start=content_start,
                content_end=content_end,
                gold_spans=tuple(gold_spans),
                gold_relations=tuple(gold_relations),
                primary_entity_types=inventory.primary_entity_types,
                auxiliary_entity_types=inventory.auxiliary_entity_types,
            )
        )

    return tuple(windows)


def audit_windows(
    windows: Iterable[WindowExample],
    inventory: LabelInventory,
) -> dict[str, Any]:
    window_list = tuple(windows)
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    relations_with_auxiliary_endpoint = 0
    auxiliary = set(inventory.auxiliary_entity_types)

    for window in window_list:
        entity_counts.update(span.label for span in window.gold_spans)
        relation_counts.update(
            relation.label for relation in window.gold_relations
        )
        relations_with_auxiliary_endpoint += sum(
            relation.source.label in auxiliary
            or relation.target.label in auxiliary
            for relation in window.gold_relations
        )

    primary_count = sum(
        entity_counts[label] for label in inventory.primary_entity_types
    )
    auxiliary_count = sum(
        entity_counts[label] for label in inventory.auxiliary_entity_types
    )
    relation_count = sum(relation_counts.values())
    return {
        "window_count": len(window_list),
        "document_count": len(
            {window.doc_seq_index for window in window_list}
        ),
        "primary_entity_count": primary_count,
        "auxiliary_entity_count": auxiliary_count,
        "trainable_entity_count": primary_count + auxiliary_count,
        "entity_counts": {
            label: entity_counts[label]
            for label in inventory.trainable_entity_types
        },
        "relation_count": relation_count,
        "relation_counts": {
            label: relation_counts[label]
            for label in inventory.relation_types
        },
        "relations_with_auxiliary_endpoint": (
            relations_with_auxiliary_endpoint
        ),
        "core_to_core_relation_count": (
            relation_count - relations_with_auxiliary_endpoint
        ),
    }


def _clean_labels(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    labels = tuple(str(item).strip() for item in value)
    if any(not label for label in labels):
        raise ValueError(f"{field} contains an empty label")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{field} contains duplicate labels")
    return labels
