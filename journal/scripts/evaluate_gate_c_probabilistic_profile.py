#!/usr/bin/env python3
"""Validation-only evaluator for Gate C probabilistic decoding.

This surface contains frozen mode/grid constants, validation decoder selection,
the frozen Gate A provenance guard, the frozen Gate C config/profile contract,
split-firewall orchestration, validation runtime evaluation, and frozen runtime
preparation. Test evaluation remains gated behind explicit full mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from journal.scsp.artifacts import build_prediction_records as build_gate_a_prediction_records
from journal.scsp.config import GateAConfig
from journal.scsp.data import LabelInventory, load_clean_windows
from journal.scsp.gate_c import build_probabilistic_prediction_records
from journal.scsp.gate_c_artifacts import (
    build_gate_c_diagnostics,
    build_gate_c_scored_pair_rows,
    build_gate_c_span_rows,
)
from journal.scsp.gate_c_inference import infer_gate_c_split
from journal.scsp.schema import (
    load_relation_canonicalization,
    load_task_relationship_profile,
)
from journal.scsp.serialization import _record_to_dict
from journal.scsp.splits import load_fold_partition
from journal.scsp.training_config import GateATrainingConfig
from journal.scripts.run_gate_a import build_preflight
from journal.scripts.train_gate_a import (
    _combined_config_hash,
    _derive_width_cap,
    _environment,
    _git_commit,
    _make_model,
    _resolve_device,
    _score_records,
    _set_seed,
    _sha256_file,
)

_VALID_MODES = ("dev", "full")
_FROZEN_GATE_A_COMMIT = "b4033edbaf2150605c286a36e4b0564d75b0ac91"
_FROZEN_GATE_B_PARENT_COMMIT = "2e74e98231c3c5bb1b0db4d826602b61b71ba07d"
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=_VALID_MODES, default="dev")
    parser.add_argument("--gate-a-checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="journal/configs/gate_a_plain_spanpair.json",
    )
    parser.add_argument(
        "--training-config",
        default="journal/configs/gate_a_training_threshold_refine.json",
    )
    parser.add_argument(
        "--inventory",
        default="journal/configs/gate_a_label_inventory.json",
    )
    parser.add_argument(
        "--canonicalization",
        default="journal/configs/stix/relation_canonicalization_v1.json",
    )
    parser.add_argument(
        "--profile",
        default="journal/configs/stix/task_relationship_profile_v1.json",
    )
    parser.add_argument(
        "--manifest",
        default="experiments/cv_manifest/run_partitions_seed_42.json",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _windows_for_ids(windows, document_ids) -> tuple[Any, ...]:
    """Return only windows whose document id is explicitly requested."""
    wanted = set(document_ids)
    return tuple(window for window in windows if window.doc_id in wanted)


def _write_artifacts(output_dir, artifacts) -> None:
    """Write deterministic JSON/JSONL artifacts without renaming them."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        path = destination / str(name)
        if path.suffix == ".jsonl":
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in payload:
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
            continue
        path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


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

    if best is None:
        raise ValueError("decoder grid must be non-empty")
    return {"best": best, "grid": scored}


def _check_beta_zero_parity(prepared, inferences) -> dict[str, Any]:
    """Check exact Gate C beta=0 prediction parity against frozen Gate A."""
    base_inferences = tuple(inference.base for inference in inferences)
    checked_thresholds: list[float] = []
    mismatch_count = 0

    for threshold_value in _EXPECTED_THRESHOLD_GRID:
        threshold = float(threshold_value)
        checked_thresholds.append(threshold)
        gate_a_records = build_gate_a_prediction_records(
            base_inferences,
            prepared.inventory.relation_types,
            threshold,
            run_id=prepared.run_id,
            git_commit=prepared.git_commit,
            dataset_sha256=prepared.dataset_sha256,
            config_sha256=prepared.config_sha256,
            fold=prepared.fold,
            seed=prepared.seed,
            split="validation",
        )
        gate_c_records = build_probabilistic_prediction_records(
            inferences,
            prepared.inventory.relation_types,
            beta=0.0,
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
        if gate_c_records != gate_a_records:
            mismatch_count += 1

    return {
        "checked_thresholds": checked_thresholds,
        "prediction_parity": mismatch_count == 0,
        "mismatch_count": mismatch_count,
    }


def _prepare_evaluation_context(args):
    """Load and validate the frozen Gate A state without touching val/test splits."""
    mode = str(args.mode)
    if mode not in _VALID_MODES:
        raise ValueError(f"unsupported Gate C mode: {mode}")

    repo_root = Path(__file__).resolve().parents[2]
    config_path = _resolve_repo_path(repo_root, args.config)
    training_config_path = _resolve_repo_path(repo_root, args.training_config)
    inventory_path = _resolve_repo_path(repo_root, args.inventory)
    canonicalization_path = _resolve_repo_path(repo_root, args.canonicalization)
    profile_path = _resolve_repo_path(repo_root, args.profile)
    manifest_path = _resolve_repo_path(repo_root, args.manifest)
    dataset_path = Path(args.dataset).resolve()
    checkpoint_path = Path(args.gate_a_checkpoint).resolve()

    base_config = GateAConfig.from_json(config_path)
    training_config = GateATrainingConfig.from_json(training_config_path)
    inventory = LabelInventory.from_json(inventory_path)
    canonicalization = load_relation_canonicalization(canonicalization_path)
    profile = load_task_relationship_profile(profile_path)

    if int(base_config.seed) != int(args.seed):
        raise ValueError(
            f"Gate A config fixes seed={base_config.seed}; received --seed {args.seed}"
        )

    preflight = build_preflight(
        config_path,
        manifest_path,
        fold=args.fold,
        dataset_path=dataset_path,
        dry_run=False,
    )
    partition = load_fold_partition(manifest_path, args.fold)
    windows = load_clean_windows(dataset_path, inventory)

    train_windows = _windows_for_ids(windows, partition.train_document_ids)
    width_cap = _derive_width_cap(train_windows, base_config.span_width_coverage)

    dataset_sha256 = _sha256_file(dataset_path)
    config_sha256 = _combined_config_hash(
        [config_path, training_config_path, inventory_path]
    )
    canonicalization_sha256 = _sha256_file(canonicalization_path)
    task_profile_sha256 = _sha256_file(profile_path)
    gate_a_checkpoint_sha256 = _sha256_file(checkpoint_path)

    validate_gate_c_frozen_contract(
        base_config,
        training_config,
        inventory,
        canonicalization,
        profile,
        canonicalization_sha256=canonicalization_sha256,
        task_profile_sha256=task_profile_sha256,
    )

    _set_seed(args.seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    companion_path = checkpoint_path.parent / "training_run_config.json"
    if not companion_path.is_file():
        raise ValueError(
            "Gate A checkpoint requires companion training_run_config.json "
            "for fold/seed provenance"
        )
    run_metadata = json.loads(companion_path.read_text(encoding="utf-8"))
    validate_gate_a_provenance(
        checkpoint,
        run_metadata,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        fold=args.fold,
        seed=args.seed,
        width_cap=width_cap,
    )

    device = _resolve_device(args.device)
    model = _make_model(base_config, training_config, inventory, width_cap)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    commit = _git_commit(repo_root)
    run_id = f"gate-c-prob-profile-f{args.fold}-s{args.seed}-{mode}-{commit[:8]}"

    return SimpleNamespace(
        base_config=base_config,
        training_config=training_config,
        inventory=inventory,
        canonicalization=canonicalization,
        profile=profile,
        partition=partition,
        windows=windows,
        model=model,
        width_cap=width_cap,
        device=device,
        fold=int(args.fold),
        seed=int(args.seed),
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        canonicalization_sha256=canonicalization_sha256,
        task_profile_sha256=task_profile_sha256,
        gate_a_checkpoint_sha256=gate_a_checkpoint_sha256,
        git_commit=commit,
        preflight=preflight,
        checkpoint=checkpoint,
        run_metadata=run_metadata,
        run_id=run_id,
    )


def _build_validation_run_artifacts(
    prepared,
    inferences,
    records,
    selection,
    metrics,
    *,
    beta: float,
    threshold: float,
    mode: str,
    parity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frozen validation-only journal artifact bundle in memory."""
    prediction_rows = tuple(_record_to_dict(record) for record in records)
    posterior_rows = build_gate_c_span_rows(inferences)
    pair_rows = build_gate_c_scored_pair_rows(
        inferences,
        prepared.inventory.relation_types,
        beta=beta,
        canonicalization=prepared.canonicalization,
        profile=prepared.profile,
        epsilon=_EPSILON,
    )
    diagnostics = build_gate_c_diagnostics(
        inferences,
        prepared.inventory.relation_types,
        beta=beta,
        threshold=threshold,
        canonicalization=prepared.canonicalization,
        profile=prepared.profile,
        epsilon=_EPSILON,
    )
    environment = _environment(prepared.device)
    checkpoint = prepared.checkpoint
    checkpoint_metadata = {
        "git_commit": checkpoint.get("git_commit"),
        "dataset_sha256": checkpoint.get("dataset_sha256"),
        "config_sha256": checkpoint.get("config_sha256"),
        "epoch": checkpoint.get("epoch"),
        "threshold": checkpoint.get("threshold"),
        "validation": checkpoint.get("validation"),
        "width_cap": checkpoint.get("width_cap"),
    }
    summary = {
        "status": "dev_complete" if mode == "dev" else "validation_complete",
        "mode": mode,
        "fold": prepared.fold,
        "seed": prepared.seed,
        "selection_scope": "validation-only",
        "beta": float(beta),
        "relation_threshold": float(threshold),
        "validation": selection["best"],
        "validation_metrics": metrics,
        "test_evaluated": False,
        "gate_a_parent_commit": _FROZEN_GATE_A_COMMIT,
        "gate_b_parent_commit": _FROZEN_GATE_B_PARENT_COMMIT,
        "git_commit": prepared.git_commit,
        "dataset_sha256": prepared.dataset_sha256,
        "config_sha256": prepared.config_sha256,
        "canonicalization_sha256": prepared.canonicalization_sha256,
        "task_profile_sha256": prepared.task_profile_sha256,
        "epsilon": _EPSILON,
        "beta_grid": list(_EXPECTED_BETA_GRID),
    }
    artifacts = {
        "validation_predictions.jsonl": prediction_rows,
        "validation_entity_posteriors.jsonl": posterior_rows,
        "validation_scored_pairs.jsonl": pair_rows,
        "validation_decoder_selection.json": selection,
        "validation_metrics.json": metrics,
        "validation_profile_diagnostics.json": diagnostics,
        "environment.json": environment,
        "hashes.json": {
            "dataset_sha256": prepared.dataset_sha256,
            "gate_a_combined_config_sha256": prepared.config_sha256,
            "canonicalization_sha256": prepared.canonicalization_sha256,
            "task_profile_sha256": prepared.task_profile_sha256,
            "gate_a_checkpoint_sha256": prepared.gate_a_checkpoint_sha256,
        },
        "checkpoint_metadata.json": {
            "checkpoint": checkpoint_metadata,
            "companion_training_run_config": prepared.run_metadata,
        },
        "run_summary.json": summary,
    }
    if parity is not None:
        artifacts["validation_beta_zero_parity.json"] = parity
        summary["beta_zero_parity"] = parity
    return artifacts


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
    parity = _check_beta_zero_parity(prepared, inferences)
    if not bool(parity.get("prediction_parity")) or int(parity.get("mismatch_count", 0)) != 0:
        raise ValueError("Gate C beta=0 prediction parity check failed")

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
    artifacts = _build_validation_run_artifacts(
        prepared,
        inferences,
        records,
        selection,
        metrics,
        beta=beta,
        threshold=threshold,
        mode="dev",
        parity=parity,
    )
    return {
        "status": "validation_complete",
        "mode": "dev",
        "beta": beta,
        "relation_threshold": threshold,
        "validation": best,
        "validation_metrics": metrics,
        "test_evaluated": False,
        "artifacts": artifacts,
    }


def _evaluate_test(prepared, test_windows, validation_result) -> dict[str, Any]:
    """Evaluate test once using the decoder frozen on validation."""
    beta = float(validation_result["beta"])
    threshold = float(validation_result["relation_threshold"])
    inferences = infer_gate_c_split(
        prepared.model,
        test_windows,
        prepared.inventory,
        width_cap=prepared.width_cap,
        base_config=prepared.base_config,
        training_config=prepared.training_config,
        device=prepared.device,
    )
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
        split="test",
        epsilon=_EPSILON,
    )
    metrics = _score_records(records, prepared.inventory)
    return {
        "mode": "full",
        "test_evaluated": True,
        "test_metrics": metrics,
        "artifacts": {"test_metrics.json": metrics},
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