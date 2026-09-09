"""Pure schema/profile utilities for SCSP Gate B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

RelationStatus = Literal["direct", "alias", "inverse", "unresolved"]
_VALID_RELATION_STATUSES = {"direct", "alias", "inverse", "unresolved"}
_REQUIRED_CANONICALIZATION_FIELDS = (
    "project_label",
    "canonical_label",
    "swap_endpoints",
    "status",
)
_REQUIRED_PROFILE_FIELDS = (
    "version",
    "resolved_entity_types",
    "unresolved_entity_types",
    "allowed_triples",
)
_REQUIRED_PROFILE_TRIPLE_FIELDS = (
    "source_type",
    "relation_type",
    "target_type",
)


@dataclass(frozen=True, slots=True)
class CanonicalRelation:
    label: str | None
    swap_endpoints: bool
    status: RelationStatus


@dataclass(frozen=True, slots=True)
class CanonicalizationTable:
    by_project_label: Mapping[str, CanonicalRelation]


@dataclass(frozen=True, slots=True)
class TaskRelationshipProfile:
    allowed_triples: frozenset[tuple[str, str, str]]
    resolved_entity_types: frozenset[str]
    unresolved_entity_types: frozenset[str]


@dataclass(frozen=True, slots=True)
class HardProfileMaskResult:
    masked_logits: tuple[float, ...]
    compatibility: tuple[bool | None, ...]
    selected_index: int | None


def load_relation_canonicalization(
    path: str | Path,
) -> CanonicalizationTable:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError(f"unsupported canonicalization version: {payload.get('version')}")
    if not isinstance(payload.get("rules"), list):
        raise ValueError("canonicalization rules must be a list")

    rules: dict[str, CanonicalRelation] = {}
    for item in payload["rules"]:
        if not isinstance(item, dict):
            raise ValueError("canonicalization rule must be an object")
        for field in _REQUIRED_CANONICALIZATION_FIELDS:
            if field not in item:
                raise ValueError(f"missing required field: {field}")

        project_label = item["project_label"]
        if not isinstance(project_label, str) or not project_label.strip():
            raise ValueError("project_label must be a non-empty string")
        if project_label in rules:
            raise ValueError(f"duplicate project_label: {project_label}")
        status = item["status"]
        if status not in _VALID_RELATION_STATUSES:
            raise ValueError(f"invalid status: {status}")
        canonical_label = item["canonical_label"]
        if status == "unresolved" and canonical_label is not None:
            raise ValueError("canonical_label must be null for unresolved status")
        if status != "unresolved" and (
            not isinstance(canonical_label, str) or not canonical_label.strip()
        ):
            raise ValueError("canonical_label must be defined for resolved status")
        swap_endpoints = item["swap_endpoints"]
        if not isinstance(swap_endpoints, bool):
            raise ValueError("swap_endpoints must be boolean")
        if status == "unresolved" and swap_endpoints:
            raise ValueError("unresolved relations cannot swap endpoints")
        rules[project_label] = CanonicalRelation(
            label=canonical_label,
            swap_endpoints=swap_endpoints,
            status=status,
        )
    return CanonicalizationTable(by_project_label=rules)


def load_task_relationship_profile(
    path: str | Path,
) -> TaskRelationshipProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in _REQUIRED_PROFILE_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    if payload["version"] != 1:
        raise ValueError(f"unsupported profile version: {payload['version']}")

    resolved_raw = payload["resolved_entity_types"]
    unresolved_raw = payload["unresolved_entity_types"]
    if not isinstance(resolved_raw, list) or not isinstance(unresolved_raw, list):
        raise ValueError("entity type status sets must be lists")
    if not resolved_raw:
        raise ValueError("resolved_entity_types must be non-empty")
    if any(not isinstance(value, str) or not value.strip() for value in resolved_raw):
        raise ValueError("resolved_entity_types must contain non-empty strings")
    if any(not isinstance(value, str) or not value.strip() for value in unresolved_raw):
        raise ValueError("unresolved_entity_types must contain non-empty strings")
    if len(set(resolved_raw)) != len(resolved_raw):
        raise ValueError("resolved_entity_types must be unique")
    if len(set(unresolved_raw)) != len(unresolved_raw):
        raise ValueError("unresolved_entity_types must be unique")

    resolved_entity_types = frozenset(resolved_raw)
    unresolved_entity_types = frozenset(unresolved_raw)
    overlap = resolved_entity_types & unresolved_entity_types
    if overlap:
        raise ValueError(f"endpoint status overlap: {sorted(overlap)}")

    allowed_raw = payload["allowed_triples"]
    if not isinstance(allowed_raw, list):
        raise ValueError("allowed_triples must be a list")
    triple_list: list[tuple[str, str, str]] = []
    for item in allowed_raw:
        if not isinstance(item, dict):
            raise ValueError("task-profile triple must be an object")
        for field in _REQUIRED_PROFILE_TRIPLE_FIELDS:
            if field not in item:
                raise ValueError(f"missing required field: {field}")
        triple = (
            item["source_type"],
            item["relation_type"],
            item["target_type"],
        )
        if any(not isinstance(value, str) or not value.strip() for value in triple):
            raise ValueError("task-profile triple fields must be non-empty strings")
        triple_list.append(triple)

    triples = frozenset(triple_list)
    if len(triples) != len(triple_list):
        raise ValueError("duplicate task-profile triple")
    declared_entity_types = resolved_entity_types | unresolved_entity_types
    for source_type, _, target_type in triple_list:
        for entity_type in (source_type, target_type):
            if entity_type not in declared_entity_types:
                raise ValueError(f"undeclared entity type: {entity_type}")
    return TaskRelationshipProfile(
        allowed_triples=triples,
        resolved_entity_types=resolved_entity_types,
        unresolved_entity_types=unresolved_entity_types,
    )


def canonicalize_relation(
    label: str,
    canonicalization: CanonicalizationTable,
) -> CanonicalRelation:
    try:
        return canonicalization.by_project_label[label]
    except KeyError as exc:
        raise ValueError(f"unknown project relation label: {label}") from exc


def is_profile_compatible(
    source_type: str,
    relation_label: str,
    target_type: str,
    *,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
) -> bool | None:
    known_entity_types = (
        profile.resolved_entity_types | profile.unresolved_entity_types
    )
    for entity_type in (source_type, target_type):
        if entity_type not in known_entity_types:
            raise ValueError(f"unknown entity type: {entity_type}")

    if (
        source_type in profile.unresolved_entity_types
        or target_type in profile.unresolved_entity_types
    ):
        return None

    relation = canonicalize_relation(relation_label, canonicalization)
    if relation.status == "unresolved":
        return None
    lookup_source, lookup_target = (
        (target_type, source_type)
        if relation.swap_endpoints
        else (source_type, target_type)
    )
    return (lookup_source, relation.label, lookup_target) in profile.allowed_triples


def hard_profile_mask(
    relation_logits: list[float] | tuple[float, ...],
    *,
    source_type: str,
    target_type: str,
    relation_types: list[str] | tuple[str, ...],
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
) -> HardProfileMaskResult:
    logits = tuple(float(value) for value in relation_logits)
    labels = tuple(relation_types)
    if len(logits) != len(labels):
        raise ValueError("relation_logits and relation_types must have the same length")
    if not labels:
        raise ValueError("relation_logits and relation_types must be non-empty")
    compatibility = tuple(
        is_profile_compatible(
            source_type,
            label,
            target_type,
            canonicalization=canonicalization,
            profile=profile,
        )
        for label in labels
    )
    masked_logits = tuple(
        float("-inf") if state is False else logit
        for logit, state in zip(logits, compatibility)
    )
    survivors = [
        index for index, state in enumerate(compatibility) if state is not False
    ]
    selected_index = (
        max(survivors, key=lambda index: (masked_logits[index], -index))
        if survivors
        else None
    )
    return HardProfileMaskResult(
        masked_logits=masked_logits,
        compatibility=compatibility,
        selected_index=selected_index,
    )
