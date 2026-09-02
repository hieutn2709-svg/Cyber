"""Create aggregate-only integrity diagnostics for the controlled Gate A dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from journal.scsp.data import LabelInventory, audit_windows, load_clean_windows
from journal.scsp.spans import derive_width_cap


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_audit(
    dataset_path: str | Path,
    inventory_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    inventory_path = Path(inventory_path)
    manifest_path = Path(manifest_path)

    inventory = LabelInventory.from_json(inventory_path)
    windows = load_clean_windows(dataset_path, inventory)
    report = audit_windows(windows, inventory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_hash = _sha256(dataset_path)
    expected_hash = str(manifest["dataset_sha256"])
    if observed_hash != expected_hash:
        raise ValueError(
            "controlled dataset hash mismatch: "
            f"observed={observed_hash} expected={expected_hash}"
        )

    by_doc: dict[str, list[Any]] = {}
    for window in windows:
        by_doc.setdefault(window.doc_id, []).append(window)
    all_doc_ids = set(by_doc)

    per_fold: dict[str, Any] = {}
    for fold_entry in manifest["outer_folds"]:
        fold = int(fold_entry["fold"])
        train_ids = tuple(
            str(value) for value in fold_entry["train_document_ids"]
        )
        validation_ids = tuple(
            str(value) for value in fold_entry["validation_document_ids"]
        )
        test_ids = tuple(
            str(value) for value in fold_entry["test_document_ids"]
        )
        _assert_ids_exist(all_doc_ids, train_ids, f"fold {fold} train")
        _assert_ids_exist(
            all_doc_ids,
            validation_ids,
            f"fold {fold} validation",
        )
        _assert_ids_exist(all_doc_ids, test_ids, f"fold {fold} test")
        if (
            set(train_ids) & set(validation_ids)
            or set(train_ids) & set(test_ids)
            or set(validation_ids) & set(test_ids)
        ):
            raise ValueError(
                f"document overlap detected in manifest fold {fold}"
            )

        train_windows = [
            window for doc_id in train_ids for window in by_doc[doc_id]
        ]
        test_windows = [
            window for doc_id in test_ids for window in by_doc[doc_id]
        ]
        test_audit = audit_windows(test_windows, inventory)
        train_spans = [
            span
            for window in train_windows
            for span in window.gold_spans
        ]
        width_cap = (
            derive_width_cap(train_spans, coverage=0.995)
            if train_spans
            else None
        )
        per_fold[str(fold)] = {
            "train_document_count": len(train_ids),
            "validation_document_count": len(validation_ids),
            "test_document_count": len(test_ids),
            "train_window_count": len(train_windows),
            "test_window_count": len(test_windows),
            "train_span_width_cap_99_5": width_cap,
            "test_primary_entity_count": test_audit[
                "primary_entity_count"
            ],
            "test_auxiliary_entity_count": test_audit[
                "auxiliary_entity_count"
            ],
            "test_relation_count": test_audit["relation_count"],
            "test_core_to_core_relation_count": test_audit[
                "core_to_core_relation_count"
            ],
            "test_relations_with_auxiliary_endpoint": test_audit[
                "relations_with_auxiliary_endpoint"
            ],
        }

    report.update(
        {
            "dataset_sha256": observed_hash,
            "manifest_dataset_sha256": expected_hash,
            "manifest_sha256": _sha256(manifest_path),
            "inventory_sha256": _sha256(inventory_path),
            "split_seed": int(manifest["split_seed"]),
            "manifest_run_seed": int(manifest["run_seed"]),
            "per_fold": per_fold,
        }
    )
    return report


def _assert_ids_exist(
    available: set[str],
    ids: tuple[str, ...],
    scope: str,
) -> None:
    missing = sorted(set(ids) - available)
    if missing:
        raise ValueError(
            f"{scope} document IDs are absent from dataset: {missing}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = build_dataset_audit(
        args.dataset,
        args.inventory,
        args.manifest,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
