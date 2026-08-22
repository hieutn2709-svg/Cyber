"""BIEOS sequence tags and relation-role labels used by the reviewed model.

Entity boundary/type prediction and relation-role prediction are deliberately
separate tasks. The sequence tag space contains ``O`` plus four BIEOS tags for
each of 15 model entity types (61 tags total). A separate token-level head uses
four role classes: ``O``, ``ROLE_1``, ``ROLE_2``, and ``ROLE_BOTH``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_ENTITY_TYPES = (
    "attack-pattern",
    "campaign",
    "domain-name",
    "file_paths",
    "identity",
    "indicator",
    "intrusion-set",
    "location",
    "malware",
    "sha256s",
    "tactic",
    "threat-actor",
    "tool",
    "url",
    "vulnerability",
)
ROLE_LABELS = ("O", "ROLE_1", "ROLE_2", "ROLE_BOTH")


class BIEOSScheme:
    def __init__(self, mapping_config_path: str | Path = "Configs/stix_mapping.json"):
        mapping_path = Path(mapping_config_path)
        if mapping_path.exists():
            config = json.loads(mapping_path.read_text(encoding="utf-8"))
            configured_types = config.get("model_entity_types")
            self.entity_types = tuple(configured_types or DEFAULT_MODEL_ENTITY_TYPES)
        else:
            self.entity_types = DEFAULT_MODEL_ENTITY_TYPES

        if len(self.entity_types) != 15 or len(set(self.entity_types)) != 15:
            raise ValueError("The reviewed sequence tag inventory requires 15 unique entity types")

        self.tag_to_id: dict[str, int] = {}
        self.id_to_tag: dict[int, str] = {}
        self.role_to_id = {role: index for index, role in enumerate(ROLE_LABELS)}
        self.id_to_role = {index: role for role, index in self.role_to_id.items()}
        self._build_labels()

    def _build_labels(self) -> None:
        labels = ["O"]
        for entity_type in self.entity_types:
            for prefix in ("B", "I", "E", "S"):
                labels.append(f"{prefix}-{entity_type}")
        self.tag_to_id = {tag: index for index, tag in enumerate(labels)}
        self.id_to_tag = {index: tag for tag, index in self.tag_to_id.items()}

    def get_num_labels(self) -> int:
        return len(self.tag_to_id)

    def get_num_role_labels(self) -> int:
        return len(self.role_to_id)

    def decode_entities(
        self,
        tag_sequence: Iterable[int],
        role_sequence: Iterable[int] | None = None,
    ) -> list[dict[str, int | str]]:
        tag_ids = list(tag_sequence)
        role_ids = list(role_sequence) if role_sequence is not None else [0] * len(tag_ids)
        if len(tag_ids) != len(role_ids):
            raise ValueError("Tag and role sequences must have the same length")

        tags = [self.id_to_tag.get(tag_id, "O") for tag_id in tag_ids]
        roles = [self.id_to_role.get(role_id, "O") for role_id in role_ids]
        entities: list[dict[str, int | str]] = []
        current: dict[str, int | str] | None = None
        current_roles: list[str] = []

        def close_entity(end: int) -> None:
            nonlocal current, current_roles
            if current is not None:
                current["end"] = end
                current["role"] = self._merge_roles(current_roles)
                entities.append(current)
            current = None
            current_roles = []

        for index, (tag, role) in enumerate(zip(tags, roles)):
            if tag == "O" or "-" not in tag:
                close_entity(index - 1)
                continue
            prefix, entity_type = tag.split("-", 1)
            if prefix == "S":
                close_entity(index - 1)
                entities.append(
                    {"type": entity_type, "role": self._merge_roles([role]),
                     "start": index, "end": index}
                )
            elif prefix == "B":
                close_entity(index - 1)
                current = {"type": entity_type, "start": index}
                current_roles = [role]
            elif prefix in {"I", "E"} and current is not None:
                if current["type"] != entity_type:
                    close_entity(index - 1)
                    continue
                current_roles.append(role)
                if prefix == "E":
                    close_entity(index)
            else:
                close_entity(index - 1)
        close_entity(len(tags) - 1)
        return entities

    @staticmethod
    def _merge_roles(roles: Iterable[str]) -> str:
        observed = set(roles)
        if "ROLE_BOTH" in observed or {"ROLE_1", "ROLE_2"} <= observed:
            return "ROLE_BOTH"
        if "ROLE_1" in observed:
            return "ROLE_1"
        if "ROLE_2" in observed:
            return "ROLE_2"
        return "O"

    @staticmethod
    def get_relation_triples(entities: Iterable[dict]) -> list[dict[str, dict]]:
        entity_list = list(entities)
        subjects = [entity for entity in entity_list if entity.get("role") in {"ROLE_1", "ROLE_BOTH"}]
        objects = [entity for entity in entity_list if entity.get("role") in {"ROLE_2", "ROLE_BOTH"}]
        return [
            {"subject": subject, "object": obj}
            for subject in subjects
            for obj in objects
            if subject is not obj
        ]


if __name__ == "__main__":
    scheme = BIEOSScheme()
    print(
        f"Initialized {scheme.get_num_labels()} sequence tags and "
        f"{scheme.get_num_role_labels()} role labels."
    )
