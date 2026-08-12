#!/usr/bin/env python3
"""Build reproducible, leakage-safe five-fold document splits.

The input is the Label Studio JSON export used by the project.  Every task is
treated as one CTI report.  Fold assignment is performed once at document
level; entity and relation label counts are used only to improve balance.

For outer fold k, fold k is the test partition.  A small, deterministic,
label-aware validation subset is selected only from the remaining four folds;
all other non-test reports are training data.

This rotating 3/1/1 protocol ensures that each report is used as test data
exactly once and that no report contributes content to more than one partition
within a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


UUID_PREFIX = re.compile(r"^[0-9a-fA-F]{8}-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("Data/Annotations.json"),
        help="Label Studio JSON export (one task per CTI report).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/cv_manifest"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def source_name(task: dict[str, Any]) -> str:
    uploaded = str(task.get("file_upload") or "").strip()
    if uploaded:
        return UUID_PREFIX.sub("", uploaded)
    return f"task-{task['id']}"


def extract_document(task: dict[str, Any]) -> dict[str, Any]:
    annotations = task.get("annotations") or []
    if not annotations:
        raise ValueError(f"Task {task.get('id')} has no completed annotation")

    results = annotations[0].get("result") or []
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()

    for item in results:
        item_type = item.get("type")
        if item_type == "labels":
            labels = item.get("value", {}).get("labels") or []
            if labels:
                entity_counts[str(labels[0])] += 1
        elif item_type == "relation":
            labels = item.get("labels") or []
            relation_counts[str(labels[0]) if labels else "<untyped>"] += 1

    text = str(task.get("data", {}).get("text") or "")
    return {
        "document_id": str(task["id"]),
        "source_file": source_name(task),
        "characters": len(text),
        "entity_mentions": sum(entity_counts.values()),
        "relation_instances": sum(relation_counts.values()),
        "entity_counts": dict(sorted(entity_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


def feature_counts(document: dict[str, Any]) -> Counter[str]:
    values: Counter[str] = Counter()
    values.update({f"entity:{k}": v for k, v in document["entity_counts"].items()})
    values.update({f"relation:{k}": v for k, v in document["relation_counts"].items()})
    return values


def assign_folds(
    documents: list[dict[str, Any]], n_folds: int, seed: int
) -> list[list[dict[str, Any]]]:
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    if len(documents) < n_folds:
        raise ValueError("Number of documents must be at least the number of folds")

    rng = random.Random(seed)
    totals: Counter[str] = Counter()
    doc_features: dict[str, Counter[str]] = {}
    for document in documents:
        features = feature_counts(document)
        doc_features[document["document_id"]] = features
        totals.update(features)

    # Documents containing rare labels are placed first.  A seeded jitter gives
    # deterministic tie-breaking without depending on input order.
    jitter = {d["document_id"]: rng.random() for d in documents}

    def rarity_key(document: dict[str, Any]) -> tuple[float, int, float]:
        features = doc_features[document["document_id"]]
        rarity = sum(value / totals[label] for label, value in features.items())
        return (-rarity, -sum(features.values()), jitter[document["document_id"]])

    ordered = sorted(documents, key=rarity_key)
    fold_documents: list[list[dict[str, Any]]] = [[] for _ in range(n_folds)]
    fold_features: list[Counter[str]] = [Counter() for _ in range(n_folds)]
    base_size, remainder = divmod(len(documents), n_folds)
    capacities = [base_size + (1 if index < remainder else 0) for index in range(n_folds)]

    for document in ordered:
        features = doc_features[document["document_id"]]
        scores: list[tuple[float, int]] = []
        for fold_index in range(n_folds):
            if len(fold_documents[fold_index]) >= capacities[fold_index]:
                continue
            current = fold_features[fold_index]
            prospective = fold_features[fold_index] + features
            feature_delta = 0.0
            for label, total in totals.items():
                target = total / n_folds
                weight = 1.0 / math.sqrt(total)
                before = ((current[label] - target) ** 2) / (target + 1.0)
                after = ((prospective[label] - target) ** 2) / (target + 1.0)
                feature_delta += weight * (after - before)
            # Prefer the least-filled fold when label-balance changes are close.
            occupancy = len(fold_documents[fold_index]) / capacities[fold_index]
            scores.append((feature_delta + 0.10 * occupancy, fold_index))

        best_score = min(score for score, _ in scores)
        candidates = [index for score, index in scores if abs(score - best_score) < 1e-12]
        selected = rng.choice(candidates)
        fold_documents[selected].append(document)
        fold_features[selected].update(features)

    for fold in fold_documents:
        fold.sort(key=lambda item: (item["source_file"].lower(), item["document_id"]))
    return fold_documents


def select_validation_documents(
    candidates: list[dict[str, Any]], validation_size: int, seed: int
) -> list[dict[str, Any]]:
    """Select an approximately label-balanced validation subset greedily."""
    if not 0 < validation_size < len(candidates):
        raise ValueError("Validation size must be between zero and candidate size")
    rng = random.Random(seed)
    totals: Counter[str] = Counter()
    features_by_id: dict[str, Counter[str]] = {}
    for document in candidates:
        features = feature_counts(document)
        features_by_id[document["document_id"]] = features
        totals.update(features)
    target_fraction = validation_size / len(candidates)
    targets = {label: count * target_fraction for label, count in totals.items()}
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    remaining = list(candidates)

    while len(selected) < validation_size:
        scored: list[tuple[float, float, dict[str, Any]]] = []
        for document in remaining:
            prospective = selected_counts + features_by_id[document["document_id"]]
            error = sum(
                ((prospective[label] - target) ** 2) / (target + 1.0)
                for label, target in targets.items()
            )
            scored.append((error, rng.random(), document))
        _, _, chosen = min(scored, key=lambda item: (item[0], item[1]))
        selected.append(chosen)
        selected_counts.update(features_by_id[chosen["document_id"]])
        remaining.remove(chosen)
    return selected


def write_outputs(
    documents: list[dict[str, Any]],
    folds: list[list[dict[str, Any]]],
    output_dir: Path,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment = {
        document["document_id"]: fold_index
        for fold_index, fold in enumerate(folds)
        for document in fold
    }
    if set(assignment) != {document["document_id"] for document in documents}:
        raise RuntimeError("Fold assignment is incomplete")

    manifest = {
        "protocol": "five-fold document-level cross-validation with rotating validation fold",
        "assignment_seed": seed,
        "number_of_folds": len(folds),
        "number_of_documents": len(documents),
        "folds": [
            {
                "fold": fold_index,
                "documents": [document["document_id"] for document in fold],
                "source_files": [document["source_file"] for document in fold],
            }
            for fold_index, fold in enumerate(folds)
        ],
        "documents": [dict(document, fold=assignment[document["document_id"]]) for document in documents],
    }
    (output_dir / "document_folds.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "document_folds.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "document_id",
            "source_file",
            "fold",
            "characters",
            "entity_mentions",
            "relation_instances",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for document in sorted(documents, key=lambda item: int(item["document_id"])):
            row = {key: document[key] for key in fieldnames if key != "fold"}
            row["fold"] = assignment[document["document_id"]]
            writer.writerow(row)

    split_dir = output_dir / "outer_splits"
    split_dir.mkdir(exist_ok=True)
    n_folds = len(folds)
    validation_size = max(1, round(len(documents) * 0.10))
    for test_fold in range(n_folds):
        test_documents = list(folds[test_fold])
        outer_training_candidates = [
            document
            for fold_index in range(n_folds)
            if fold_index != test_fold
            for document in folds[fold_index]
        ]
        validation_documents = select_validation_documents(
            outer_training_candidates, validation_size, seed + 1000 + test_fold
        )
        validation_ids = {document["document_id"] for document in validation_documents}
        training_documents = [
            document
            for document in outer_training_candidates
            if document["document_id"] not in validation_ids
        ]
        split = {
            "outer_fold": test_fold,
            "test_fold": test_fold,
            "validation_selection_seed": seed + 1000 + test_fold,
            "train_document_ids": [document["document_id"] for document in training_documents],
            "validation_document_ids": [document["document_id"] for document in validation_documents],
            "test_document_ids": [document["document_id"] for document in test_documents],
        }
        (split_dir / f"outer_fold_{test_fold}.json").write_text(
            json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    with (output_dir / "fold_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["fold", "documents", "entity_mentions", "relation_instances", "characters"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for fold_index, fold in enumerate(folds):
            writer.writerow(
                {
                    "fold": fold_index,
                    "documents": len(fold),
                    "entity_mentions": sum(d["entity_mentions"] for d in fold),
                    "relation_instances": sum(d["relation_instances"] for d in fold),
                    "characters": sum(d["characters"] for d in fold),
                }
            )


def main() -> None:
    args = parse_args()
    tasks = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
    documents = [extract_document(task) for task in tasks]
    ids = [document["document_id"] for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document/task identifiers detected")
    folds = assign_folds(documents, args.folds, args.seed)
    write_outputs(documents, folds, args.output_dir, args.seed)
    print(f"Wrote {args.folds} document folds for {len(documents)} reports to {args.output_dir}")


if __name__ == "__main__":
    main()
