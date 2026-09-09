#!/usr/bin/env python3
"""Evaluate Gate B Hard Profile from a frozen Gate A checkpoint.

`dev` performs validation-only inference/threshold selection and never evaluates
the test partition. `full` repeats the frozen validation selection and then
performs exactly one test inference/evaluation pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_VALID_MODES = ("dev", "full")
_FROZEN_GATE_A_COMMIT = "b4033edbaf2150605c286a36e4b0564d75b0ac91"
_EXPECTED_THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(85, 100))


def mode_evaluates_test(mode: str) -> bool:
    if mode not in _VALID_MODES:
        raise ValueError(f"unsupported Gate B mode: {mode}")
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
) -> None:
    """Reject a checkpoint/run-metadata mismatch before evaluation proceeds."""
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


def _windows_for_ids(windows: Iterable[Any], document_ids: Iterable[str]) -> tuple[Any, ...]:
    wanted = set(document_ids)
    selected = tuple(window for window in windows if window.doc_id in wanted)
    found = {window.doc_id for window in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"manifest documents missing from controlled dataset: {missing}")
    return selected


def _select_hard_profile_threshold(
    inferences,
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
    from journal.scsp.gate_b import build_hard_profile_prediction_records
    from journal.scripts.train_gate_a import _score_records

    scored: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    for threshold in thresholds:
        records = build_hard_profile_prediction_records(
            inferences,
            inventory.relation_types,
            float(threshold),
            canonicalization=canonicalization,
            profile=profile,
            run_id=run_id,
            git_commit=git_commit,
            dataset_sha256=dataset_sha256,
            config_sha256=config_sha256,
            fold=fold,
            seed=seed,
            split="validation",
        )
        metrics = _score_records(records, inventory)
        item = {
            "threshold": float(threshold),
            "relation_f1": float(metrics["all_relation"]["f1"]),
            "primary_entity_f1": float(metrics["primary_entity"]["f1"]),
        }
        scored.append(item)
        if best is None or (
            item["relation_f1"],
            item["primary_entity_f1"],
            -item["threshold"],
        ) > (
            best["relation_f1"],
            best["primary_entity_f1"],
            -best["threshold"],
        ):
            best = item
    if best is None:
        raise ValueError("threshold grid must be non-empty")
    return {"best": best, "grid": scored}


def _safe_checkpoint_metadata(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    return {
        "source_path": str(checkpoint_path),
        "git_commit": checkpoint.get("git_commit"),
        "dataset_sha256": checkpoint.get("dataset_sha256"),
        "config_sha256": checkpoint.get("config_sha256"),
        "epoch": checkpoint.get("epoch"),
        "threshold": checkpoint.get("threshold"),
        "validation": checkpoint.get("validation"),
        "width_cap": checkpoint.get("width_cap"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from journal.scripts.run_gate_a import build_preflight, write_preflight_artifacts
    from journal.scripts.train_gate_a import (
        _combined_config_hash,
        _derive_width_cap,
        _environment,
        _git_commit,
        _infer_split,
        _make_model,
        _resolve_device,
        _score_records,
        _set_seed,
        _sha256_file,
        _write_json,
    )
    from journal.scsp.config import GateAConfig
    from journal.scsp.data import LabelInventory, load_clean_windows
    from journal.scsp.gate_b import build_hard_profile_prediction_records
    from journal.scsp.gate_b_artifacts import (
        build_hard_profile_diagnostics,
        write_hard_profile_scored_pair_jsonl,
    )
    from journal.scsp.schema import (
        load_relation_canonicalization,
        load_task_relationship_profile,
    )
    from journal.scsp.serialization import write_prediction_jsonl
    from journal.scsp.splits import load_fold_partition
    from journal.scsp.training_config import GateATrainingConfig

    if args.mode not in _VALID_MODES:
        raise ValueError(f"unsupported Gate B mode: {args.mode}")
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        raise RuntimeError(
            "Gate B requires launch-time PYTHONHASHSEED matching --seed; "
            f"run with PYTHONHASHSEED={args.seed}"
        )

    repo_root = Path(__file__).resolve().parents[2]
    config_path = _resolve_repo_path(repo_root, args.config)
    training_config_path = _resolve_repo_path(repo_root, args.training_config)
    inventory_path = _resolve_repo_path(repo_root, args.inventory)
    canonicalization_path = _resolve_repo_path(repo_root, args.canonicalization)
    profile_path = _resolve_repo_path(repo_root, args.profile)
    manifest_path = _resolve_repo_path(repo_root, args.manifest)
    dataset_path = Path(args.dataset).resolve()
    checkpoint_path = Path(args.gate_a_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = GateAConfig.from_json(config_path)
    training_config = GateATrainingConfig.from_json(training_config_path)
    inventory = LabelInventory.from_json(inventory_path)
    canonicalization = load_relation_canonicalization(canonicalization_path)
    profile = load_task_relationship_profile(profile_path)

    if args.seed != base_config.seed:
        raise ValueError(
            f"Gate A config fixes seed={base_config.seed}; received --seed {args.seed}"
        )
    if base_config.max_span_candidates != 128:
        raise ValueError("Gate B requires frozen max_span_candidates=128")
    if base_config.max_relation_token_distance != 96:
        raise ValueError("Gate B requires frozen max_relation_token_distance=96")
    if training_config.max_epochs != 12:
        raise ValueError("Gate B requires frozen max_epochs=12")
    if tuple(training_config.threshold_grid) != _EXPECTED_THRESHOLD_GRID:
        raise ValueError("Gate B requires the frozen 0.85-0.99 threshold grid")
    if set(canonicalization.by_project_label) != set(inventory.relation_types):
        raise ValueError("Gate B canonicalization does not cover relation inventory")
    if (
        profile.resolved_entity_types | profile.unresolved_entity_types
        != set(inventory.trainable_entity_types)
    ):
        raise ValueError("Gate B profile endpoint types do not match entity inventory")

    preflight = build_preflight(
        config_path,
        manifest_path,
        fold=args.fold,
        dataset_path=dataset_path,
        dry_run=False,
    )
    write_preflight_artifacts(preflight, output_dir)

    partition = load_fold_partition(manifest_path, args.fold)
    windows = load_clean_windows(dataset_path, inventory)
    train_windows = _windows_for_ids(windows, partition.train_document_ids)
    validation_windows = _windows_for_ids(windows, partition.validation_document_ids)
    width_cap = _derive_width_cap(train_windows, base_config.span_width_coverage)

    dataset_sha256 = _sha256_file(dataset_path)
    config_sha256 = _combined_config_hash(
        [config_path, training_config_path, inventory_path]
    )
    canonicalization_sha256 = _sha256_file(canonicalization_path)
    profile_sha256 = _sha256_file(profile_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)

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
    )
    if int(checkpoint["width_cap"]) != int(width_cap):
        raise ValueError("Gate A checkpoint width_cap mismatch")

    device = _resolve_device(args.device)
    model = _make_model(base_config, training_config, inventory, width_cap)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    commit = _git_commit(repo_root)
    run_id = f"gate-b-hard-profile-f{args.fold}-s{args.seed}-{args.mode}-{commit[:8]}"
    validation_inferences = _infer_split(
        model,
        validation_windows,
        inventory,
        width_cap=width_cap,
        base_config=base_config,
        training_config=training_config,
        device=device,
    )
    threshold_selection = _select_hard_profile_threshold(
        validation_inferences,
        training_config.threshold_grid,
        inventory,
        canonicalization=canonicalization,
        profile=profile,
        run_id=run_id,
        git_commit=commit,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        fold=args.fold,
        seed=args.seed,
    )
    threshold = float(threshold_selection["best"]["threshold"])
    validation_records = build_hard_profile_prediction_records(
        validation_inferences,
        inventory.relation_types,
        threshold,
        canonicalization=canonicalization,
        profile=profile,
        run_id=run_id,
        git_commit=commit,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        fold=args.fold,
        seed=args.seed,
        split="validation",
    )
    validation_metrics = _score_records(validation_records, inventory)
    validation_diagnostics = build_hard_profile_diagnostics(
        validation_inferences,
        inventory.relation_types,
        threshold,
        canonicalization=canonicalization,
        profile=profile,
    )

    write_prediction_jsonl(
        validation_records, output_dir / "validation_predictions.jsonl"
    )
    write_hard_profile_scored_pair_jsonl(
        output_dir / "validation_scored_pairs.jsonl",
        validation_inferences,
        inventory.relation_types,
        canonicalization=canonicalization,
        profile=profile,
    )
    _write_json(output_dir / "validation_threshold_selection.json", threshold_selection)
    _write_json(output_dir / "validation_metrics.json", validation_metrics)
    _write_json(output_dir / "validation_profile_diagnostics.json", validation_diagnostics)
    _write_json(output_dir / "environment.json", _environment(device))
    _write_json(
        output_dir / "hashes.json",
        {
            "dataset_sha256": dataset_sha256,
            "gate_a_combined_config_sha256": config_sha256,
            "canonicalization_sha256": canonicalization_sha256,
            "task_profile_sha256": profile_sha256,
            "gate_a_checkpoint_sha256": checkpoint_sha256,
        },
    )
    _write_json(
        output_dir / "checkpoint_metadata.json",
        {
            "checkpoint": _safe_checkpoint_metadata(checkpoint, checkpoint_path),
            "companion_training_run_config": run_metadata,
        },
    )

    summary: dict[str, Any] = {
        "status": "dev_complete" if args.mode == "dev" else "validation_complete",
        "mode": args.mode,
        "fold": args.fold,
        "seed": args.seed,
        "selection_scope": "validation-only",
        "relation_threshold": threshold,
        "validation": threshold_selection["best"],
        "validation_metrics": validation_metrics,
        "test_evaluated": False,
        "gate_a_parent_commit": _FROZEN_GATE_A_COMMIT,
        "git_commit": commit,
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
        "canonicalization_sha256": canonicalization_sha256,
        "task_profile_sha256": profile_sha256,
    }

    if mode_evaluates_test(args.mode):
        # The only branch that obtains and evaluates test windows.
        test_windows = _windows_for_ids(windows, partition.test_document_ids)
        test_inferences = _infer_split(
            model,
            test_windows,
            inventory,
            width_cap=width_cap,
            base_config=base_config,
            training_config=training_config,
            device=device,
        )
        test_records = build_hard_profile_prediction_records(
            test_inferences,
            inventory.relation_types,
            threshold,
            canonicalization=canonicalization,
            profile=profile,
            run_id=run_id,
            git_commit=commit,
            dataset_sha256=dataset_sha256,
            config_sha256=config_sha256,
            fold=args.fold,
            seed=args.seed,
            split="test",
        )
        test_metrics = _score_records(test_records, inventory)
        test_diagnostics = build_hard_profile_diagnostics(
            test_inferences,
            inventory.relation_types,
            threshold,
            canonicalization=canonicalization,
            profile=profile,
        )
        write_prediction_jsonl(test_records, output_dir / "test_predictions.jsonl")
        write_hard_profile_scored_pair_jsonl(
            output_dir / "test_scored_pairs.jsonl",
            test_inferences,
            inventory.relation_types,
            canonicalization=canonicalization,
            profile=profile,
        )
        _write_json(output_dir / "test_metrics.json", test_metrics)
        _write_json(output_dir / "test_profile_diagnostics.json", test_diagnostics)
        summary.update(
            {
                "status": "full_complete",
                "test_evaluated": True,
                "test": test_metrics,
            }
        )

    _write_json(output_dir / "run_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
