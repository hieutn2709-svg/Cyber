#!/usr/bin/env python3
"""Train/evaluate SCSP-CTI Gate A Plain SpanPair on one fixed manifest fold.

Modes deliberately separate diagnostic execution from the publishable run:
- overfit: one positive training window only; no validation/test evaluation.
- smoke: configured smoke epochs on train + validation; test is never evaluated.
- dev: full max-epoch train + validation development; test is never evaluated.
- full: validation-only checkpoint/threshold selection followed by one frozen test pass.

Transformers is imported lazily by ``GateASpanPairModel.from_pretrained`` so
``--help`` and unit tests do not require a model download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


_VALID_MODES = ("overfit", "smoke", "dev", "full")


def mode_evaluates_test(mode: str) -> bool:
    if mode not in _VALID_MODES:
        raise ValueError(f"unsupported Gate A mode: {mode}")
    return mode == "full"


def mode_epoch_budget(
    mode: str,
    training_config: Any,
    *,
    overfit_epochs: int = 20,
) -> int:
    if mode == "overfit":
        if overfit_epochs < 1:
            raise ValueError("overfit_epochs must be >= 1")
        return int(overfit_epochs)
    if mode == "smoke":
        return int(training_config.smoke_epochs)
    if mode in {"dev", "full"}:
        return int(training_config.max_epochs)
    raise ValueError(f"unsupported Gate A mode: {mode}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=_VALID_MODES,
        default="smoke",
        help=(
            "overfit diagnostic, smoke train/validation, validation-only dev, "
            "or frozen full test run"
        ),
    )
    parser.add_argument(
        "--config",
        default="journal/configs/gate_a_plain_spanpair.json",
        help="immutable Gate A base config",
    )
    parser.add_argument(
        "--training-config",
        default="journal/configs/gate_a_training.json",
        help="Gate A engineering/training hyperparameters",
    )
    parser.add_argument(
        "--inventory",
        default="journal/configs/gate_a_label_inventory.json",
        help="versioned entity/relation label inventory",
    )
    parser.add_argument(
        "--manifest",
        default="experiments/cv_manifest/run_partitions_seed_42.json",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=20,
        help="diagnostic-only epoch count for --mode overfit",
    )
    return parser


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_config_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise RuntimeError("Gate A training requires an exact Git commit")
    return commit


def _resolve_device(requested: str):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _environment(device) -> dict[str, Any]:
    import torch

    try:
        import importlib.metadata as metadata

        transformers_version = metadata.version("transformers")
    except Exception:
        transformers_version = None
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
    }


def _split_windows(windows, partition):
    train_ids = set(partition.train_document_ids)
    validation_ids = set(partition.validation_document_ids)
    test_ids = set(partition.test_document_ids)
    train = tuple(w for w in windows if w.doc_id in train_ids)
    validation = tuple(w for w in windows if w.doc_id in validation_ids)
    test = tuple(w for w in windows if w.doc_id in test_ids)
    found = {w.doc_id for w in windows}
    expected = train_ids | validation_ids | test_ids
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"manifest documents missing from controlled dataset: {missing}")
    return train, validation, test


def _derive_width_cap(train_windows, coverage: float) -> int:
    from journal.scsp.spans import derive_width_cap

    spans = [span for window in train_windows for span in window.gold_spans]
    if not spans:
        raise ValueError("training partition contains no gold spans")
    return derive_width_cap(spans, coverage=coverage)


def _make_model(base_config, training_config, inventory, width_cap: int):
    from journal.scsp.runtime_model import GateASpanPairModel

    return GateASpanPairModel.from_pretrained(
        model_name=base_config.encoder_model,
        revision=base_config.encoder_revision,
        num_entity_classes=len(inventory.trainable_entity_types) + 1,
        num_relation_types=len(inventory.relation_types),
        max_width=width_cap,
        width_embedding_dim=training_config.width_embedding_dim,
        context_dim=training_config.context_dim,
        distance_embedding_dim=training_config.distance_embedding_dim,
        max_distance=base_config.max_relation_token_distance,
    )


def _make_optimizer(model, training_config):
    import torch

    encoder_parameters = list(model.encoder_parameters())
    head_parameters = list(model.head_parameters())
    if not encoder_parameters or not head_parameters:
        raise ValueError("Gate A optimizer requires encoder and head parameters")
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": training_config.encoder_lr},
            {"params": head_parameters, "lr": training_config.head_lr},
        ],
        weight_decay=training_config.weight_decay,
    )


def _train_epoch(
    model,
    optimizer,
    scaler,
    windows,
    inventory,
    *,
    width_cap: int,
    base_config,
    training_config,
    seed: int,
    epoch: int,
    device,
) -> dict[str, float | int]:
    import torch
    from journal.scsp.training import compute_training_window_loss

    model.train()
    order = list(windows)
    random.Random(seed + epoch * 1009).shuffle(order)
    optimizer.zero_grad(set_to_none=True)

    totals = {
        "total_loss": 0.0,
        "entity_loss": 0.0,
        "relation_existence_loss": 0.0,
        "relation_type_loss": 0.0,
        "relation_positive_count": 0,
        "relation_pair_count": 0,
        "proposal_gold_count": 0,
        "proposal_matched_count": 0,
    }
    amp_enabled = bool(training_config.fp16 and device.type == "cuda")
    accumulation = int(training_config.gradient_accumulation_steps)

    for step, window in enumerate(order, start=1):
        sample_seed = seed + epoch * 100_000 + step
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            result = compute_training_window_loss(
                model,
                window,
                inventory,
                width_cap=width_cap,
                base_config=base_config,
                train_config=training_config,
                seed=sample_seed,
                device=device,
            )
            scaled_loss = result.total_loss / accumulation
        scaler.scale(scaled_loss).backward()

        if step % accumulation == 0 or step == len(order):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        totals["total_loss"] += float(result.total_loss.detach().cpu())
        totals["entity_loss"] += float(result.entity_loss.detach().cpu())
        totals["relation_existence_loss"] += float(
            result.relation_existence_loss.detach().cpu()
        )
        totals["relation_type_loss"] += float(
            result.relation_type_loss.detach().cpu()
        )
        totals["relation_positive_count"] += int(result.relation_positive_count)
        totals["relation_pair_count"] += int(result.relation_pair_count)
        totals["proposal_gold_count"] += int(result.proposal_gold_count)
        totals["proposal_matched_count"] += int(result.proposal_matched_count)

    count = max(len(order), 1)
    for key in (
        "total_loss",
        "entity_loss",
        "relation_existence_loss",
        "relation_type_loss",
    ):
        totals[key] /= count
    totals["window_count"] = len(order)
    return totals


def _infer_split(
    model,
    windows,
    inventory,
    *,
    width_cap: int,
    base_config,
    training_config,
    device,
):
    from journal.scsp.training import infer_window

    model.eval()
    return tuple(
        infer_window(
            model,
            window,
            inventory,
            width_cap=width_cap,
            base_config=base_config,
            relation_chunk_size=training_config.relation_inference_chunk_size,
            device=device,
        )
        for window in windows
    )


def _score_records(records, inventory) -> dict[str, Any]:
    from journal.evaluation.rescore_gate_a import strict_micro_scores

    primary = set(inventory.primary_entity_types)
    all_scores = strict_micro_scores(records)
    primary_scores = strict_micro_scores(records, entity_labels=primary)
    core_relation_scores = strict_micro_scores(
        records,
        entity_labels=primary,
        relation_endpoint_labels=primary,
    )
    return {
        "all": all_scores,
        "primary_entity": primary_scores["entity"],
        "all_relation": all_scores["relation"],
        "core_to_core_relation": core_relation_scores["relation"],
    }


def _build_records(
    inferences,
    inventory,
    threshold: float,
    *,
    run_id: str,
    git_commit: str,
    dataset_sha256: str,
    config_sha256: str,
    fold: int,
    seed: int,
    split: str,
):
    from journal.scsp.artifacts import build_prediction_records

    return build_prediction_records(
        inferences,
        inventory.relation_types,
        threshold,
        run_id=run_id,
        git_commit=git_commit,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        fold=fold,
        seed=seed,
        split=split,
    )


def _select_validation_threshold(
    inferences,
    thresholds,
    inventory,
    *,
    run_id: str,
    git_commit: str,
    dataset_sha256: str,
    config_sha256: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    scored: list[dict[str, float]] = []
    best = None
    for threshold in thresholds:
        records = _build_records(
            inferences,
            inventory,
            float(threshold),
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
    assert best is not None
    return {"best": best, "grid": scored}


def _candidate_diagnostics(inferences, base_config) -> dict[str, Any]:
    from journal.scsp.diagnostics import pair_diagnostics
    from journal.scsp.pairs import generate_ordered_pairs

    gold_spans = sum(i.proposal_gold_count for i in inferences)
    proposal_matches = sum(i.proposal_matched_count for i in inferences)
    pruned_matches = sum(i.post_pruning_typed_matched_count for i in inferences)

    gold_relations = pre_matches = post_matches = 0
    pre_candidates = post_candidates = 0
    post_positive = post_negative = 0
    for inference in inferences:
        pre_pairs = generate_ordered_pairs(inference.predicted_spans, None)
        post_pairs = generate_ordered_pairs(
            inference.predicted_spans,
            base_config.max_relation_token_distance,
        )
        pre = pair_diagnostics(inference.window.gold_relations, pre_pairs)
        post = pair_diagnostics(inference.window.gold_relations, post_pairs)
        gold_relations += pre.gold_relation_count
        pre_matches += pre.matched_relation_count
        post_matches += post.matched_relation_count
        pre_candidates += pre.candidate_count
        post_candidates += post.candidate_count
        post_positive += post.positive_candidate_count
        post_negative += post.negative_candidate_count

    return {
        "span_proposal": {
            "gold_count": gold_spans,
            "matched_count": proposal_matches,
            "recall": proposal_matches / gold_spans if gold_spans else 1.0,
        },
        "span_post_pruning_typed": {
            "gold_count": gold_spans,
            "matched_count": pruned_matches,
            "recall": pruned_matches / gold_spans if gold_spans else 1.0,
        },
        "pair_pre_distance": {
            "gold_relation_count": gold_relations,
            "matched_relation_count": pre_matches,
            "recall": pre_matches / gold_relations if gold_relations else 1.0,
            "candidate_count": pre_candidates,
        },
        "pair_post_distance": {
            "gold_relation_count": gold_relations,
            "matched_relation_count": post_matches,
            "recall": post_matches / gold_relations if gold_relations else 1.0,
            "candidate_count": post_candidates,
            "positive_candidate_count": post_positive,
            "negative_candidate_count": post_negative,
        },
    }


def build_history_row(
    *,
    epoch: int,
    seconds: float,
    train_metrics: dict[str, Any],
    selected: dict[str, float],
    candidate_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build one immutable-by-convention development history record."""
    recall_keys = (
        "span_proposal",
        "span_post_pruning_typed",
        "pair_pre_distance",
        "pair_post_distance",
    )
    return {
        "epoch": int(epoch),
        "seconds": float(seconds),
        "train": dict(train_metrics),
        "validation": dict(selected),
        "candidate_recall": {
            key: float(candidate_diagnostics[key]["recall"])
            for key in recall_keys
        },
    }


def _write_validation_logits(path: Path, inferences) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for inference in inferences:
            window = inference.window
            for scored in inference.scored_pairs:
                row = {
                    "document_id": window.doc_id,
                    "window_index": window.window_index,
                    "source": {
                        "start": scored.pair.source.start,
                        "end": scored.pair.source.end,
                        "label": scored.pair.source.label,
                    },
                    "target": {
                        "start": scored.pair.target.start,
                        "end": scored.pair.target.end,
                        "label": scored.pair.target.label,
                    },
                    "token_distance": scored.pair.token_distance,
                    "existence_logit": scored.existence_logit,
                    "relation_type_logits": list(scored.type_logits),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_per_document_metrics(path: Path, records, inventory) -> None:
    payload = {
        record.document_id: _score_records((record,), inventory)
        for record in records
    }
    _write_json(path, payload)


def _is_better(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    return (
        candidate["relation_f1"],
        candidate["primary_entity_f1"],
    ) > (
        incumbent["relation_f1"],
        incumbent["primary_entity_f1"],
    )


def _run_overfit(
    model,
    optimizer,
    scaler,
    train_windows,
    inventory,
    *,
    epochs: int,
    width_cap: int,
    base_config,
    training_config,
    seed: int,
    device,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = [window for window in train_windows if window.gold_relations]
    if not candidates:
        raise ValueError("overfit mode requires a training window with a relation")
    window = min(
        candidates,
        key=lambda item: (len(item.input_ids), item.doc_seq_index, item.window_index),
    )
    history = []
    singleton = (window,)
    for epoch in range(1, epochs + 1):
        metrics = _train_epoch(
            model,
            optimizer,
            scaler,
            singleton,
            inventory,
            width_cap=width_cap,
            base_config=base_config,
            training_config=training_config,
            seed=seed,
            epoch=epoch,
            device=device,
        )
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"overfit epoch={epoch:02d} loss={metrics['total_loss']:.6f} "
            f"entity={metrics['entity_loss']:.6f} "
            f"rel_exist={metrics['relation_existence_loss']:.6f} "
            f"rel_type={metrics['relation_type_loss']:.6f}",
            flush=True,
        )
    _write_json(output_dir / "overfit_history.json", history)
    return {
        "status": "overfit_complete",
        "test_evaluated": False,
        "document_id": window.doc_id,
        "window_index": window.window_index,
        "epochs": epochs,
        "initial_loss": history[0]["total_loss"],
        "final_loss": history[-1]["total_loss"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from journal.scripts.run_gate_a import build_preflight, write_preflight_artifacts
    from journal.scsp.config import GateAConfig
    from journal.scsp.data import LabelInventory, load_clean_windows
    from journal.scsp.serialization import write_prediction_jsonl
    from journal.scsp.splits import load_fold_partition
    from journal.scsp.training_config import GateATrainingConfig

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        (repo_root / args.config).resolve()
        if not Path(args.config).is_absolute()
        else Path(args.config)
    )
    training_config_path = (
        (repo_root / args.training_config).resolve()
        if not Path(args.training_config).is_absolute()
        else Path(args.training_config)
    )
    inventory_path = (
        (repo_root / args.inventory).resolve()
        if not Path(args.inventory).is_absolute()
        else Path(args.inventory)
    )
    manifest_path = (
        (repo_root / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    )
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = GateAConfig.from_json(config_path)
    training_config = GateATrainingConfig.from_json(training_config_path)
    inventory = LabelInventory.from_json(inventory_path)
    if args.seed != base_config.seed:
        raise ValueError(
            f"current Gate A development config fixes seed={base_config.seed}; "
            f"received --seed {args.seed}"
        )

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
    train_windows, validation_windows, test_windows = _split_windows(windows, partition)
    width_cap = _derive_width_cap(train_windows, base_config.span_width_coverage)

    commit = _git_commit(repo_root)
    dataset_sha256 = _sha256_file(dataset_path)
    config_sha256 = _combined_config_hash(
        [config_path, training_config_path, inventory_path]
    )
    run_id = (
        f"{base_config.experiment_name}-f{args.fold}-s{args.seed}-"
        f"{args.mode}-{commit[:8]}"
    )
    device = _resolve_device(args.device)
    _set_seed(args.seed)
    _write_json(output_dir / "environment.json", _environment(device))
    _write_json(
        output_dir / "training_run_config.json",
        {
            "mode": args.mode,
            "fold": args.fold,
            "seed": args.seed,
            "epoch_budget": mode_epoch_budget(
                args.mode,
                training_config,
                overfit_epochs=args.overfit_epochs,
            ),
            "width_cap": width_cap,
            "selection_scope": "validation-only",
            "test_evaluated": mode_evaluates_test(args.mode),
            "dataset_sha256": dataset_sha256,
            "combined_config_sha256": config_sha256,
            "git_commit": commit,
            "split_window_counts": {
                "train": len(train_windows),
                "validation": len(validation_windows),
                "test": len(test_windows),
            },
        },
    )

    model = _make_model(base_config, training_config, inventory, width_cap)
    model.to(device)
    optimizer = _make_optimizer(model, training_config)
    amp_enabled = bool(training_config.fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    epoch_budget = mode_epoch_budget(
        args.mode,
        training_config,
        overfit_epochs=args.overfit_epochs,
    )

    if args.mode == "overfit":
        summary = _run_overfit(
            model,
            optimizer,
            scaler,
            train_windows,
            inventory,
            epochs=epoch_budget,
            width_cap=width_cap,
            base_config=base_config,
            training_config=training_config,
            seed=args.seed,
            device=device,
            output_dir=output_dir,
        )
        _write_json(output_dir / "run_summary.json", summary)
        return summary

    history: list[dict[str, Any]] = []
    best_selection: dict[str, float] | None = None
    best_epoch = None
    best_threshold = None
    stale_epochs = 0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, epoch_budget + 1):
        start_time = time.time()
        train_metrics = _train_epoch(
            model,
            optimizer,
            scaler,
            train_windows,
            inventory,
            width_cap=width_cap,
            base_config=base_config,
            training_config=training_config,
            seed=args.seed,
            epoch=epoch,
            device=device,
        )
        validation_inferences = _infer_split(
            model,
            validation_windows,
            inventory,
            width_cap=width_cap,
            base_config=base_config,
            training_config=training_config,
            device=device,
        )
        threshold_selection = _select_validation_threshold(
            validation_inferences,
            training_config.threshold_grid,
            inventory,
            run_id=run_id,
            git_commit=commit,
            dataset_sha256=dataset_sha256,
            config_sha256=config_sha256,
            fold=args.fold,
            seed=args.seed,
        )
        selected = threshold_selection["best"]
        candidate_diagnostics = _candidate_diagnostics(
            validation_inferences, base_config
        )
        row = build_history_row(
            epoch=epoch,
            seconds=time.time() - start_time,
            train_metrics=train_metrics,
            selected=selected,
            candidate_diagnostics=candidate_diagnostics,
        )
        history.append(row)
        _write_json(output_dir / "training_history.json", history)
        recall = row["candidate_recall"]
        print(
            f"epoch={epoch:02d} train_loss={train_metrics['total_loss']:.6f} "
            f"val_RF1={selected['relation_f1']:.6f} "
            f"val_EF1_primary={selected['primary_entity_f1']:.6f} "
            f"spanR={recall['span_post_pruning_typed']:.4f} "
            f"pairR={recall['pair_post_distance']:.4f} "
            f"threshold={selected['threshold']:.2f} "
            f"time={row['seconds'] / 60:.1f}m",
            flush=True,
        )

        if _is_better(selected, best_selection):
            best_selection = dict(selected)
            best_epoch = epoch
            best_threshold = float(selected["threshold"])
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "threshold": best_threshold,
                    "validation": best_selection,
                    "git_commit": commit,
                    "dataset_sha256": dataset_sha256,
                    "config_sha256": config_sha256,
                    "width_cap": width_cap,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        if (
            args.mode == "full"
            and stale_epochs >= training_config.early_stopping_patience
        ):
            print(
                f"early stopping after epoch {epoch}; best_epoch={best_epoch}",
                flush=True,
            )
            break

    if best_epoch is None or best_threshold is None:
        raise RuntimeError("training produced no validation checkpoint")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    validation_inferences = _infer_split(
        model,
        validation_windows,
        inventory,
        width_cap=width_cap,
        base_config=base_config,
        training_config=training_config,
        device=device,
    )
    frozen_threshold_selection = _select_validation_threshold(
        validation_inferences,
        training_config.threshold_grid,
        inventory,
        run_id=run_id,
        git_commit=commit,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        fold=args.fold,
        seed=args.seed,
    )
    frozen_threshold = float(frozen_threshold_selection["best"]["threshold"])
    if abs(frozen_threshold - best_threshold) > 1e-12:
        raise RuntimeError(
            "reloaded best checkpoint changed validation-selected threshold"
        )

    validation_records = _build_records(
        validation_inferences,
        inventory,
        frozen_threshold,
        run_id=run_id,
        git_commit=commit,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        fold=args.fold,
        seed=args.seed,
        split="validation",
    )
    write_prediction_jsonl(
        validation_records, output_dir / "validation_predictions.jsonl"
    )
    _write_validation_logits(
        output_dir / "validation_logits.jsonl", validation_inferences
    )
    _write_json(
        output_dir / "validation_threshold_selection.json",
        frozen_threshold_selection,
    )
    _write_json(
        output_dir / "validation_metrics.json",
        _score_records(validation_records, inventory),
    )
    _write_json(
        output_dir / "validation_candidate_diagnostics.json",
        _candidate_diagnostics(validation_inferences, base_config),
    )
    _write_per_document_metrics(
        output_dir / "validation_per_document_metrics.json",
        validation_records,
        inventory,
    )

    status = "smoke_complete" if args.mode == "smoke" else "training_complete"
    if args.mode == "dev":
        status = "dev_complete"
    summary: dict[str, Any] = {
        "status": status,
        "mode": args.mode,
        "fold": args.fold,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "relation_threshold": frozen_threshold,
        "validation": frozen_threshold_selection["best"],
        "test_evaluated": False,
        "git_commit": commit,
        "dataset_sha256": dataset_sha256,
        "config_sha256": config_sha256,
    }

    if args.mode == "full":
        # This is the only code path that evaluates the fixed test partition.
        # Checkpoint and threshold are already frozen from validation above.
        test_inferences = _infer_split(
            model,
            test_windows,
            inventory,
            width_cap=width_cap,
            base_config=base_config,
            training_config=training_config,
            device=device,
        )
        test_records = _build_records(
            test_inferences,
            inventory,
            frozen_threshold,
            run_id=run_id,
            git_commit=commit,
            dataset_sha256=dataset_sha256,
            config_sha256=config_sha256,
            fold=args.fold,
            seed=args.seed,
            split="test",
        )
        write_prediction_jsonl(test_records, output_dir / "test_predictions.jsonl")
        test_metrics = _score_records(test_records, inventory)
        _write_json(output_dir / "test_metrics.json", test_metrics)
        _write_json(
            output_dir / "test_candidate_diagnostics.json",
            _candidate_diagnostics(test_inferences, base_config),
        )
        _write_per_document_metrics(
            output_dir / "test_per_document_metrics.json",
            test_records,
            inventory,
        )
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
