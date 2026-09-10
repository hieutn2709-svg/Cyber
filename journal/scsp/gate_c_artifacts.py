"""Deterministic posterior and scored-pair artifacts for Gate C."""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .gate_c import (
    adjust_relation_type_logits,
    probabilistic_compatibility,
    select_relation_type,
)
from .schema import CanonicalizationTable, TaskRelationshipProfile


def _write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def build_gate_c_span_rows(
    inferences: Sequence[object],
) -> tuple[dict[str, Any], ...]:
    """Build one deterministic prediction-side row per retained span."""
    rows: list[dict[str, Any]] = []
    for inference in inferences:
        base = inference.base
        window = base.window
        for span in base.predicted_spans:
            key = span.typed_key
            if key not in inference.posterior_by_typed_key:
                raise ValueError(f"missing posterior for retained span {key!r}")
            posterior = inference.posterior_by_typed_key[key]
            rows.append(
                {
                    "document_id": window.doc_id,
                    "window_index": int(window.window_index),
                    "start": int(span.start),
                    "end": int(span.end),
                    "top1_entity_type": posterior.top1_entity_type,
                    "entity_probability": float(posterior.entity_probability),
                    "none_probability": float(posterior.none_probability),
                    "entity_types": list(posterior.entity_types),
                    "conditional_non_none_posterior": [
                        float(value)
                        for value in posterior.conditional_probabilities
                    ],
                }
            )
    return tuple(rows)


def write_gate_c_span_jsonl(
    path: str | Path,
    inferences: Sequence[object],
) -> tuple[dict[str, Any], ...]:
    """Write deterministic strict-JSON retained-span posterior rows."""
    rows = build_gate_c_span_rows(inferences)
    _write_jsonl(path, rows)
    return rows


def build_gate_c_scored_pair_rows(
    inferences: Sequence[object],
    relation_types: Sequence[str],
    *,
    beta: float,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
    epsilon: float = 1e-8,
) -> tuple[dict[str, Any], ...]:
    """Build one deterministic prediction-side row per Gate A scored pair."""
    labels = tuple(str(label) for label in relation_types)
    if not labels:
        raise ValueError("relation_types must be non-empty")

    rows: list[dict[str, Any]] = []
    for inference in inferences:
        base = inference.base
        window = base.window
        for scored in base.scored_pairs:
            if len(scored.type_logits) != len(labels):
                raise ValueError("relation logit count does not match inventory")
            if not math.isfinite(float(scored.existence_logit)):
                raise ValueError("existence_logit must be finite")
            if any(not math.isfinite(float(value)) for value in scored.type_logits):
                raise ValueError("relation type logits must be finite")

            source_key = scored.pair.source.typed_key
            target_key = scored.pair.target.typed_key
            if source_key not in inference.posterior_by_typed_key:
                raise ValueError(f"missing posterior for relation source {source_key!r}")
            if target_key not in inference.posterior_by_typed_key:
                raise ValueError(f"missing posterior for relation target {target_key!r}")

            source_posterior = inference.posterior_by_typed_key[source_key]
            target_posterior = inference.posterior_by_typed_key[target_key]
            compatibility = probabilistic_compatibility(
                source_posterior,
                target_posterior,
                labels,
                canonicalization=canonicalization,
                profile=profile,
            )
            adjusted = adjust_relation_type_logits(
                scored.type_logits,
                compatibility,
                beta=beta,
                epsilon=epsilon,
            )
            if any(not math.isfinite(float(value)) for value in adjusted):
                raise ValueError("adjusted relation type logits must be finite")
            selected_index = select_relation_type(adjusted)

            rows.append(
                {
                    "document_id": window.doc_id,
                    "window_index": int(window.window_index),
                    "source": {
                        "start": int(scored.pair.source.start),
                        "end": int(scored.pair.source.end),
                        "label": scored.pair.source.label,
                    },
                    "target": {
                        "start": int(scored.pair.target.start),
                        "end": int(scored.pair.target.end),
                        "label": scored.pair.target.label,
                    },
                    "token_distance": int(scored.pair.token_distance),
                    "existence_logit": float(scored.existence_logit),
                    "relation_types": list(labels),
                    "raw_relation_type_logits": [
                        float(value) for value in scored.type_logits
                    ],
                    "compatibility_scores": [
                        float(value) for value in compatibility
                    ],
                    "beta": float(beta),
                    "epsilon": float(epsilon),
                    "adjusted_relation_type_logits": [
                        float(value) for value in adjusted
                    ],
                    "selected_index": int(selected_index),
                    "selected_project_label": labels[selected_index],
                }
            )
    return tuple(rows)


def write_gate_c_scored_pair_jsonl(
    path: str | Path,
    inferences: Sequence[object],
    relation_types: Sequence[str],
    *,
    beta: float,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
    epsilon: float = 1e-8,
) -> tuple[dict[str, Any], ...]:
    """Write deterministic strict-JSON Gate C scored-pair rows."""
    rows = build_gate_c_scored_pair_rows(
        inferences,
        relation_types,
        beta=beta,
        canonicalization=canonicalization,
        profile=profile,
        epsilon=epsilon,
    )
    _write_jsonl(path, rows)
    return rows


def build_gate_c_diagnostics(
    inferences: Sequence[object],
    relation_types: Sequence[str],
    *,
    beta: float,
    threshold: float,
    canonicalization: CanonicalizationTable,
    profile: TaskRelationshipProfile,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Build deterministic Gate C diagnostics from prediction state only."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("relation threshold must lie in [0, 1]")
    labels = tuple(str(label) for label in relation_types)
    if not labels:
        raise ValueError("relation_types must be non-empty")

    retained_span_count = 0
    top1_mismatch_count = 0
    posterior_max_normalization_deviation = 0.0
    entity_score_parity_max_abs_difference = 0.0

    for inference in inferences:
        for span in inference.base.predicted_spans:
            retained_span_count += 1
            key = span.typed_key
            if key not in inference.posterior_by_typed_key:
                raise ValueError(f"missing posterior for retained span {key!r}")
            posterior = inference.posterior_by_typed_key[key]
            if posterior.top1_entity_type != span.label:
                top1_mismatch_count += 1
            posterior_max_normalization_deviation = max(
                posterior_max_normalization_deviation,
                abs(sum(posterior.conditional_probabilities) - 1.0),
            )
            entity_score_parity_max_abs_difference = max(
                entity_score_parity_max_abs_difference,
                abs(float(posterior.entity_probability) - float(span.entity_score)),
            )

    pair_rows = build_gate_c_scored_pair_rows(
        inferences,
        labels,
        beta=beta,
        canonicalization=canonicalization,
        profile=profile,
        epsilon=epsilon,
    )

    compatibility_values = {label: [] for label in labels}
    transition_counts: Counter[str] = Counter()
    candidate_relation_count = len(pair_rows)
    argmax_change_count = 0
    above_threshold_pair_count = 0
    above_threshold_change_count = 0
    emitted_relation_count = 0
    unresolved_source_masses: list[float] = []
    unresolved_target_masses: list[float] = []

    row_index = 0
    for inference in inferences:
        for scored in inference.base.scored_pairs:
            row = pair_rows[row_index]
            row_index += 1

            for label, score in zip(labels, row["compatibility_scores"]):
                compatibility_values[label].append(float(score))

            raw_index = select_relation_type(row["raw_relation_type_logits"])
            selected_index = int(row["selected_index"])
            changed = raw_index != selected_index
            if changed:
                argmax_change_count += 1
                transition_counts[
                    f"{labels[raw_index]}->{labels[selected_index]}"
                ] += 1

            probability = _sigmoid(float(row["existence_logit"]))
            if probability >= threshold:
                above_threshold_pair_count += 1
                emitted_relation_count += 1
                if changed:
                    above_threshold_change_count += 1

            source_key = scored.pair.source.typed_key
            target_key = scored.pair.target.typed_key
            source_posterior = inference.posterior_by_typed_key[source_key]
            target_posterior = inference.posterior_by_typed_key[target_key]
            unresolved_source_masses.append(
                sum(
                    float(probability)
                    for entity_type, probability in zip(
                        source_posterior.entity_types,
                        source_posterior.conditional_probabilities,
                    )
                    if entity_type in profile.unresolved_entity_types
                )
            )
            unresolved_target_masses.append(
                sum(
                    float(probability)
                    for entity_type, probability in zip(
                        target_posterior.entity_types,
                        target_posterior.conditional_probabilities,
                    )
                    if entity_type in profile.unresolved_entity_types
                )
            )

    compatibility_summary = {}
    for label in labels:
        values = compatibility_values[label]
        compatibility_summary[label] = {
            "count": len(values),
            "mean": float(sum(values) / len(values)) if values else 0.0,
            "min": float(min(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
        }

    return {
        "retained_span_count": retained_span_count,
        "posterior_top1_parity_mismatch_count": top1_mismatch_count,
        "posterior_max_normalization_deviation": float(
            posterior_max_normalization_deviation
        ),
        "entity_score_parity_max_abs_difference": float(
            entity_score_parity_max_abs_difference
        ),
        "compatibility_summary_by_project_relation": compatibility_summary,
        "mean_unresolved_source_posterior_mass": float(
            sum(unresolved_source_masses) / len(unresolved_source_masses)
        ) if unresolved_source_masses else 0.0,
        "mean_unresolved_target_posterior_mass": float(
            sum(unresolved_target_masses) / len(unresolved_target_masses)
        ) if unresolved_target_masses else 0.0,
        "candidate_relation_count": candidate_relation_count,
        "argmax_change_count_vs_gate_a": argmax_change_count,
        "argmax_change_rate_vs_gate_a": float(
            argmax_change_count / candidate_relation_count
        ) if candidate_relation_count else 0.0,
        "above_threshold_pair_count": above_threshold_pair_count,
        "above_threshold_argmax_change_count_vs_gate_a": (
            above_threshold_change_count
        ),
        "transition_counts": dict(sorted(transition_counts.items())),
        "emitted_relation_count": emitted_relation_count,
    }
