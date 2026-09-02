"""Fixed-fold manifest loading and leakage guards for SCSP Gate A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FoldPartition:
    fold: int
    split_seed: int
    run_seed: int
    dataset_sha256: str
    train_document_ids: tuple[str, ...]
    validation_document_ids: tuple[str, ...]
    test_document_ids: tuple[str, ...]


def assert_disjoint_partition(partition: FoldPartition) -> None:
    train = set(partition.train_document_ids)
    validation = set(partition.validation_document_ids)
    test = set(partition.test_document_ids)
    overlaps = {
        "train/validation": sorted(train & validation),
        "train/test": sorted(train & test),
        "validation/test": sorted(validation & test),
    }
    present = {name: values for name, values in overlaps.items() if values}
    if present:
        detail = "; ".join(f"{name}={values}" for name, values in present.items())
        raise ValueError(f"document split overlap detected: {detail}")


def load_fold_partition(manifest_path: str | Path, fold: int) -> FoldPartition:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    matches = [entry for entry in manifest.get("outer_folds", []) if entry.get("fold") == fold]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest entry for fold {fold}, found {len(matches)}")
    entry = matches[0]
    partition = FoldPartition(
        fold=fold,
        split_seed=int(manifest["split_seed"]),
        run_seed=int(manifest["run_seed"]),
        dataset_sha256=str(manifest["dataset_sha256"]),
        train_document_ids=tuple(str(x) for x in entry["train_document_ids"]),
        validation_document_ids=tuple(str(x) for x in entry["validation_document_ids"]),
        test_document_ids=tuple(str(x) for x in entry["test_document_ids"]),
    )
    assert_disjoint_partition(partition)
    return partition
