#!/usr/bin/env python3
"""Validation-only selection primitives for Gate C probabilistic decoding.

This surface contains frozen mode/grid constants, validation decoder selection,
the frozen Gate A provenance guard, the frozen Gate C config/profile contract,
split-firewall orchestration, and validation runtime evaluation. Concrete
runtime preparation is added separately in later Task 6 steps.
"""
from __future__ import annotations

from typing import Any

from journal.scsp.gate_c import build_probabilistic_prediction_records
from journal.scsp.gate_c_inference import infer_gate_c_split
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


def validate_gate_a_provenance(
    checkpoint: dict[str, Any],
    run_metadata: dict[str, Any],
    *,
    dataset_sha256: str,
    config_sha256: str,
    fold: int,
    seed: int,
    width_cap: int,
) -> None:
    """Reject any mismatch from the frozen Gate A checkpoint lineage."""
    required_checkpoint = (
        "model_state_dict",
        "git_commit",
        "dataset_sha256",
        "config_sha256",
        "width_cap",
    )
    for field in required_checkpoint:
        if field not in checkpoint:
            raise ValueError(f"Gate A checkpoint missing metadata: {field}")

    if checkpoint["git_commit"] != _FROZEN_GATE_A_COMMIT:
        raise ValueError(
            "Gate A checkpoint lineage mismatch: "
            f"{checkpoint['git_commit']} != {_FROZEN_GATE_A_COMMIT}"
        )
    if checkpoint["dataset_sha256"] != dataset_sha256:
        raise ValueError("Gate A checkpoint dataset hash mismatch")
    if checkpoint["config_sha256"] != config_sha256:
        raise ValueError("Gate A checkpoint config hash mismatch")
    if int(checkpoint["width_cap"]) != int(width_cap):
        raise ValueError("Gate A checkpoint width_cap mismatch")

    required_run = (
        "fold",
        "seed",
        "dataset_sha256",
        "combined_config_sha256",
    )
    for field in required_run:
        if field not in run_metadata:
            raise ValueError(f"Gate A run metadata missing field: {field}")

    if int(run_metadata["fold"]) != int(fold):
        raise ValueError("Gate A checkpoint fold mismatch")
    if int(run_metadata["seed"]) != int(seed):
        raise ValueError("Gate A checkpoint seed mismatch")
    if run_metadata["dataset_sha256"] != dataset_sha256:
        raise ValueError("Gate A run metadata dataset hash mismatch")
    if run_metadata["combined_config_sha256"] != config_sha256:
        raise ValueError("Gate A run metadata config hash mismatch")


def validate_gate_c_frozen_contract(
    base_config,
    training_config,
    inventory,
    canonicalization,
    profile,
    *,
    canonicalization_sha256: str,
    task_profile_sha256: str,
) -> dict[str, str]:
    """Validate the predeclared Gate C model/training/profile contract."""
    if int(base_config.max_span_candidates) != 128:
        raise ValueError("Gate C requires max_span_candidates=128")
    if int(base_config.max_relation_token_distance) != 96:
        raise ValueError("Gate C requires max_relation_token_distance=96")
    if int(training_config.max_epochs) != 12:
        raise ValueError("Gate C requires max_epochs=12")

    threshold_grid = tuple(float(value) for value in training_config.threshold_grid)
    if threshold_grid != _EXPECTED_THRESHOLD_GRID:
        raise ValueError("Gate C threshold grid drift")
    if _EXPECTED_BETA_GRID != (0.0, 0.25, 0.5, 1.0, 2.0):
        raise ValueError("Gate C beta grid drift")
    if _EPSILON != 1e-8:
        raise ValueError("Gate C epsilon drift")

    relation_types = set(inventory.relation_types)
    canonicalized_labels = set(canonicalization.by_project_label)
    missing_relations = sorted(relation_types - canonicalized_labels)
    if missing_relations:
        raise ValueError(
            "Gate C canonicalization missing relation labels: "
            f"{missing_relations}"
        )

    trainable_entity_types = set(inventory.trainable_entity_types)
    resolved_entity_types = set(profile.resolved_entity_types)
    unresolved_entity_types = set(profile.unresolved_entity_types)
    overlap = resolved_entity_types & unresolved_entity_types
    if overlap:
        raise ValueError(
            "Gate C profile entity status overlap: "
            f"{sorted(overlap)}"
        )
    declared_entity_types = resolved_entity_types | unresolved_entity_types
    if declared_entity_types != trainable_entity_types:
        raise ValueError(
            "Gate C profile entity inventory mismatch: "
            f"declared={sorted(declared_entity_types)} "
            f"trainable={sorted(trainable_entity_types)}"
        )

    return {
        "canonicalization_sha256": str(canonicalization_sha256),
        "task_profile_sha256": str(task_profile_sha256),
    }


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


def _evaluate_validation(prepared, validation_windows) -> dict[str, Any]:
    """Infer Gate C on validation and apply the predeclared decoder search."""
    inferences = infer_gate_c_split(
        prepared.model,
        validation_windows,
        prepared.inventory,
        width_cap=prepared.width_cap,
        base_config=prepared.base_config,
        training_config=prepared.training_config,
        device=prepared.device,
    )
    selection = _select_probabilistic_decoder(
        inferences,
        _EXPECTED_BETA_GRID,
        _EXPECTED_THRESHOLD_GRID,
        prepared.inventory,
        canonicalization=prepared.canonicalization,
        profile=prepared.profile,
        run_id=prepared.run_id,
        git_commit=prepared.git_commit,
        dataset_sha256=prepared.dataset_sha256,
        config_sha256=prepared.config_sha256,
        fold=prepared.fold,
        seed=prepared.seed,
    )
    best = selection["best"]
    beta = float(best["beta"])
    threshold = float(best["threshold"])
    records = build_probabilistic_prediction_records(
        inferences,
        prepared.inventory.relation_types,
        beta=beta,
        threshold=threshold,
        canonicalization=prepared.canonicalization,
        profile=prepared.profile,
        run_id=prepared.run_id,
        git_commit=prepared.git_commit,
        dataset_sha256=prepared.dataset_sha256,
        config_sha256=prepared.config_sha256,
        fold=prepared.fold,
        seed=prepared.seed,
        split="validation",
        epsilon=_EPSILON,
    )
    metrics = _score_records(records, prepared.inventory)
    return {
        "status": "validation_complete",
        "mode": "dev",
        "beta": beta,
        "relation_threshold": threshold,
        "validation": best,
        "validation_metrics": metrics,
        "test_evaluated": False,
        "artifacts": {
            "validation_decoder_selection.json": selection,
            "validation_metrics.json": metrics,
        },
    }


def run(args) -> dict[str, Any]:
    """Orchestrate validation first and keep test strictly behind full mode."""
    mode = str(args.mode)
    if mode not in _VALID_MODES:
        raise ValueError(f"unsupported Gate C mode: {mode}")

    prepared = _prepare_evaluation_context(args)
    validation_windows = _windows_for_ids(
        prepared.windows,
        prepared.partition.validation_document_ids,
    )
    validation_result = _evaluate_validation(prepared, validation_windows)
    if not isinstance(validation_result, dict):
        raise ValueError("validation evaluator must return a mapping")

    result = dict(validation_result)
    artifacts = dict(result.get("artifacts", {}))

    if mode_evaluates_test(mode):
        test_windows = _windows_for_ids(
            prepared.windows,
            prepared.partition.test_document_ids,
        )
        test_result = _evaluate_test(prepared, test_windows, result)
        if not isinstance(test_result, dict):
            raise ValueError("test evaluator must return a mapping")
        artifacts.update(dict(test_result.get("artifacts", {})))
        for key, value in test_result.items():
            if key != "artifacts":
                result[key] = value

    result["artifacts"] = artifacts
    _write_artifacts(args.output_dir, artifacts)
    return result
