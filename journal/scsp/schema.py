"""Pure schema/profile utilities for SCSP Gate B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

RelationStatus = Literal["direct", "alias", "inverse", "unresolved"]


@dataclass(frozen=True, slots=True)
class CanonicalRelation:
    label: str | None
    swap_endpoints: bool
    status: RelationStatus


@dataclass(frozen=True, slots=True)
class CanonicalizationTable:
    by_project_label: Mapping[str, CanonicalRelation]


def load_relation_canonicalization(
    path: str | Path,
) -> CanonicalizationTable:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rules: dict[str, CanonicalRelation] = {}
    for item in payload["rules"]:
        project_label = item["project_label"]
        if project_label in rules:
            raise ValueError(f"duplicate project_label: {project_label}")
        rules[project_label] = CanonicalRelation(
            label=item["canonical_label"],
            swap_endpoints=bool(item["swap_endpoints"]),
            status=item["status"],
        )
    return CanonicalizationTable(by_project_label=rules)


def canonicalize_relation(
    label: str,
    canonicalization: CanonicalizationTable,
) -> CanonicalRelation:
    return canonicalization.by_project_label[label]
