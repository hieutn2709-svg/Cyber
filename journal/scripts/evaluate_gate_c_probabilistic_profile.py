#!/usr/bin/env python3
"""Validation-only selection primitives for Gate C probabilistic decoding.

This Task 5 surface intentionally contains only frozen mode/grid constants and
pure beta x threshold selection. Frozen-checkpoint provenance and full CLI
runtime behavior are added separately in Task 6.
"""
from __future__ import annotations

from typing import Any

from journal.scsp.gate_c import build_probabilistic_prediction_records
from journal.scripts.train_gate_a import _score_records

_VALID_MODES = ("dev", "full")
_FROZEN_GATE_A_COMMIT = "b4033edbaf2150605c286a36e4b0564d75b0ac91"
_EXPECTED_THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(85, 100))
_EXPECTED_BETA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
_EPSILON = 1e-8


def mode_evaluates_test(mode: str) -> bool:
    """Return whether a Gate C mode is authorized to evaluate test."""
    if mode not in _VALID_MODES:
        raise ValueError(f"unsupported Gate C mode: {mode}")
    return mode == "full"


def requested_evaluation_splits(mode: str) -> tuple[str, ...]:
    """Return only splits that the evaluator is authorized to infer/score."""
    return ("validation", "test") if mode_evaluates_test(mode) else ("validation",)


def _select_probabilistic_decoder(
    inferences,
    betas,
    thresholds,
    inventory,
    *,
    canonicalization,
    profile,
    run_id: str,
    git_commit: str,
    dataset_sha256: str,
    config_sha256: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    """Select beta/threshold on validation using the frozen lexicographic rule."""
    beta_values = tuple(float(beta) for beta in betas)
    threshold_values = tuple(float(threshold) for threshold in thresholds)
    if not beta_values:
        raise ValueError("beta grid must be non-empty")
    if not threshold_values:
        raise ValueError("threshold grid must be non-empty")

    scored: list[dict[str, float]] = []
    best: dict[str, float] | None = None

    for beta in beta_values:
        for threshold in threshold_values:
            records = build_probabilistic_prediction_records(
                inferences,
                inventory.relation_types,
                beta=beta,
                threshold=threshold,
                canonicalization=canonicalization,
                profile=profile,
                run_id=run_id,
                git_commit=git_commit,
                dataset_sha256=dataset_sha256,
                config_sha256=config_sha256,
                fold=fold,
                seed=seed,
                split="validation",
                epsilon=_EPSILON,
            )
            metrics = _score_records(records, inventory)
            item = {
                "beta": beta,
                "threshold": threshold,
                "relation_f1": float(metrics["all_relation"]["f1"]),
                "primary_entity_f1": float(metrics["primary_entity"]["f1"]),
            }
            scored.append(item)

            if best is None or (
                item["relation_f1"],
                item["primary_entity_f1"],
                -item["beta"],
                -item["threshold"],
            ) > (
                best["relation_f1"],
                best["primary_entity_f1"],
                -best["beta"],
                -best["threshold"],
            ):
                best = item

    if best is None:  # Defensive; non-empty grids above make this unreachable.
        raise ValueError("decoder grid must be non-empty")
    return {"best": best, "grid": scored}
