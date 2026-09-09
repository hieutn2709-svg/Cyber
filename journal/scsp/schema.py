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


def load_relation_canonicalization(
    path: str | Path,
) -> CanonicalizationTable:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rules: dict[str, CanonicalRelation] = {}
    for item in payload["rules"]:
        for field in _REQUIRED_CANONICALIZATION_FIELDS:
            if field not in item:
                raise ValueError(f"missing required field: {field}")

        project_label = item["project_label"]
        if project_label in rules:
            raise ValueError(f"duplicate project_label: {project_label}")
        status = item["status"]
        if status not in _VALID_RELATION_STATUSES:
            raise ValueError(f"invalid status: {status}")
        canonical_label = item["canonical_label"]
        if status == "unresolved" and canonical_label is not None:
            raise ValueError("canonical_label must be null for unresolved status")
        if status != "unresolved" and canonical_label is None:
            raise ValueError("canonical_label must be defined for resolved status")
        swap_endpoints = item["swap_endpoints"]
        if not isinstance(swap_endpoints, bool):
            raise ValueError("swap_endpoints must be boolean")
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

    resolved_entity_types = frozenset(payload["resolved_entity_types"])
    unresolved_entity_types = frozenset(payload["unresolved_entity_types"])
    overlap = resolved_entity_types & unresolved_entity_types
    if overlap:
        raise ValueError(f"endpoint status overlap: {sorted(overlap)}")

    triple_list: list[tuple[str, str, str]] = []
    for item in payload["allowed_triples"]:
        for field in _REQUIRED_PROFILE_TRIPLE_FIELDS:
            if field not in item:
                raise ValueError(f"missing required field: {field}")
        triple_list.append(
            (
                item["source_type"],
                item["relation_type"],
                item["target_type"],
            )
        )

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
    return canonicalization.by_project_label[label]


def is_profile_compatible(
    source_type: str,
    relation_label: str,
    target_type: str,
    *,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
) -> bool | None:
    relation = canonicalize_relation(relation_label, canonicalization)
    if relation.status == "unresolved":
        return None
    if (
        source_type in profile.unresolved_entity_types
        or target_type in profile.unresolved_entity_types
    ):
        return None
    known_entity_types = (
        profile.resolved_entity_types | profile.unresolved_entity_types
    )
    for entity_type in (source_type, target_type):
        if entity_type not in known_entity_types:
            raise ValueError(f"unknown entity type: {entity_type}")
    lookup_source, lookup_target = (
        (target_type, source_type)
        if relation.swap_endpoints
        else (source_type, target_type)
    )
    return (lookup_source, relation.label, lookup_target) in profile.allowed_triples
