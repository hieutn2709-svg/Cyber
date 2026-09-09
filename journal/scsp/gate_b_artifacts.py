"""Reproducible scored-pair artifacts and diagnostics for Gate B Hard Profile."""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schema import (
    CanonicalizationTable,
    TaskRelationshipProfile,
    canonicalize_relation,
    hard_profile_mask,
)
from .training import WindowInference


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _json_logit(value: float) -> float | str:
    if math.isfinite(value):
        return float(value)
    if value < 0:
        return "-inf"
    if value > 0:
        return "inf"
    return "nan"


def build_hard_profile_scored_pair_rows(
    inferences: Sequence[WindowInference],
    relation_types: Sequence[str],
    *,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
) -> tuple[dict[str, Any], ...]:
    """Build one deterministic, prediction-side row per scored relation pair."""
    labels = tuple(relation_types)
    if not labels:
        raise ValueError("relation_types must be non-empty")
    rows: list[dict[str, Any]] = []
    for inference in inferences:
        window = inference.window
        for scored in inference.scored_pairs:
            if len(scored.type_logits) != len(labels):
                raise ValueError("relation logit count does not match inventory")
            source_type = scored.pair.source.label
            target_type = scored.pair.target.label
            if source_type is None or target_type is None:
                raise ValueError("Gate B relation endpoints require predicted entity labels")
            mask = hard_profile_mask(
                scored.type_logits,
                source_type=source_type,
                target_type=target_type,
                relation_types=labels,
                canonicalization=canonicalization,
                profile=profile,
            )
            selected_project_label = (
                labels[mask.selected_index]
                if mask.selected_index is not None
                else None
            )
            rows.append(
                {
                    "document_id": window.doc_id,
                    "window_index": int(window.window_index),
                    "source": {
                        "start": int(scored.pair.source.start),
                        "end": int(scored.pair.source.end),
                        "label": source_type,
                    },
                    "target": {
                        "start": int(scored.pair.target.start),
                        "end": int(scored.pair.target.end),
                        "label": target_type,
                    },
                    "token_distance": int(scored.pair.token_distance),
                    "existence_logit": float(scored.existence_logit),
                    "relation_type_logits": [float(value) for value in scored.type_logits],
                    "relation_types": list(labels),
                    "compatibility": list(mask.compatibility),
                    "masked_relation_type_logits": [
                        _json_logit(value) for value in mask.masked_logits
                    ],
                    "selected_index": mask.selected_index,
                    "selected_project_label": selected_project_label,
                }
            )
    return tuple(rows)


def write_hard_profile_scored_pair_jsonl(
    path: str | Path,
    inferences: Sequence[WindowInference],
    relation_types: Sequence[str],
    *,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
) -> tuple[dict[str, Any], ...]:
    """Write deterministic strict-JSON scored-pair decisions and return rows."""
    rows = build_hard_profile_scored_pair_rows(
        inferences,
        relation_types,
        canonicalization=canonicalization,
        profile=profile,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    return rows


def build_hard_profile_diagnostics(
    inferences: Sequence[WindowInference],
    relation_types: Sequence[str],
    threshold: float,
    *,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
) -> dict[str, Any]:
    """Aggregate prediction-side profile diagnostics without reading gold labels."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("relation threshold must lie in [0, 1]")
    labels = tuple(relation_types)
    if not labels:
        raise ValueError("relation_types must be non-empty")

    candidate_relation_count = 0
    endpoint_unresolved_pair_count = 0
    resolved_compatible_class_opportunities = 0
    resolved_blocked_class_opportunities = 0
    unresolved_relation_label_opportunities = 0
    emitted_relation_count = 0
    emitted_task_profile_compatible_count = 0
    emitted_unresolved_count = 0
    emitted_blocked_count = 0
    no_compatible_type_count = 0
    by_project: Counter[str] = Counter()
    by_canonical: Counter[str] = Counter()
    by_endpoint_types: Counter[str] = Counter()

    for inference in inferences:
        for scored in inference.scored_pairs:
            candidate_relation_count += 1
            if len(scored.type_logits) != len(labels):
                raise ValueError("relation logit count does not match inventory")
            source_type = scored.pair.source.label
            target_type = scored.pair.target.label
            if source_type is None or target_type is None:
                raise ValueError("Gate B relation endpoints require predicted entity labels")
            endpoint_unresolved = (
                source_type in profile.unresolved_entity_types
                or target_type in profile.unresolved_entity_types
            )
            if endpoint_unresolved:
                endpoint_unresolved_pair_count += 1
            by_endpoint_types[f"{source_type}->{target_type}"] += 1

            mask = hard_profile_mask(
                scored.type_logits,
                source_type=source_type,
                target_type=target_type,
                relation_types=labels,
                canonicalization=canonicalization,
                profile=profile,
            )
            for label, state in zip(labels, mask.compatibility):
                rule = canonicalize_relation(label, canonicalization)
                if rule.status == "unresolved":
                    unresolved_relation_label_opportunities += 1
                elif not endpoint_unresolved:
                    if state is True:
                        resolved_compatible_class_opportunities += 1
                    elif state is False:
                        resolved_blocked_class_opportunities += 1

            if _sigmoid(scored.existence_logit) < threshold:
                continue
            if mask.selected_index is None:
                no_compatible_type_count += 1
                continue

            emitted_relation_count += 1
            selected_label = labels[mask.selected_index]
            selected_state = mask.compatibility[mask.selected_index]
            if selected_state is True:
                emitted_task_profile_compatible_count += 1
            elif selected_state is None:
                emitted_unresolved_count += 1
            else:
                emitted_blocked_count += 1
            by_project[selected_label] += 1
            rule = canonicalize_relation(selected_label, canonicalization)
            canonical_key = rule.label if rule.label is not None else "<unresolved>"
            by_canonical[canonical_key] += 1

    resolved_emitted = (
        emitted_task_profile_compatible_count + emitted_blocked_count
    )
    compatibility_rate = (
        emitted_task_profile_compatible_count / resolved_emitted
        if resolved_emitted
        else 1.0
    )
    return {
        "candidate_relation_count": candidate_relation_count,
        "endpoint_unresolved_pair_count": endpoint_unresolved_pair_count,
        "resolved_compatible_class_opportunities": resolved_compatible_class_opportunities,
        "resolved_blocked_class_opportunities": resolved_blocked_class_opportunities,
        "unresolved_relation_label_opportunities": unresolved_relation_label_opportunities,
        "emitted_relation_count": emitted_relation_count,
        "emitted_task_profile_compatible_count": emitted_task_profile_compatible_count,
        "emitted_unresolved_count": emitted_unresolved_count,
        "emitted_blocked_count": emitted_blocked_count,
        "no_compatible_type_count": no_compatible_type_count,
        "compatibility_rate_resolved_emitted": float(compatibility_rate),
        "counts_by_project_relation_label": dict(sorted(by_project.items())),
        "counts_by_canonical_relation_label": dict(sorted(by_canonical.items())),
        "counts_by_source_target_entity_type": dict(sorted(by_endpoint_types.items())),
    }
