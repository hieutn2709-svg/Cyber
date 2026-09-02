"""Gate A preflight and leakage guard.

This command prepares a reproducible plain SpanPair run. It deliberately does
not fabricate a dataset adapter or training result when the controlled dataset
is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from journal.scsp.config import GateAConfig
from journal.scsp.splits import load_fold_partition

_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_preflight(
    config_path: str | Path,
    manifest_path: str | Path,
    *,
    fold: int,
    dataset_path: str | Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    _reject_test_label_selection(raw_config)
    config = GateAConfig.from_json(config_path)
    if not _HEX40.fullmatch(config.encoder_revision):
        raise ValueError(
            "encoder_revision must be an immutable 40-character commit hash"
        )

    partition = load_fold_partition(manifest_path, fold)
    if config.split_seed != partition.split_seed:
        raise ValueError(
            f"split_seed mismatch: config={config.split_seed} "
            f"manifest={partition.split_seed}"
        )

    dataset_status = "not_provided"
    dataset_sha256 = partition.dataset_sha256
    resolved_dataset_path: str | None = None
    if dataset_path is not None:
        dataset = Path(dataset_path)
        if not dataset.is_file():
            raise ValueError(f"dataset path is not a file: {dataset}")
        observed_sha256 = sha256_file(dataset)
        if observed_sha256 != partition.dataset_sha256:
            raise ValueError(
                "dataset SHA-256 mismatch: "
                f"observed={observed_sha256} expected={partition.dataset_sha256}"
            )
        dataset_status = "hash_verified"
        dataset_sha256 = observed_sha256
        resolved_dataset_path = str(dataset.resolve())
    elif not dry_run:
        raise ValueError(
            "non-dry-run preparation requires an explicit dataset path"
        )

    resolved = asdict(config)
    resolved["fold"] = fold
    resolved.update(
        {
            "selection_scope": "validation-only",
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "dataset_path": resolved_dataset_path,
            "dataset_sha256": dataset_sha256,
            "dataset_status": dataset_status,
            "split_counts": {
                "train": len(partition.train_document_ids),
                "validation": len(partition.validation_document_ids),
                "test": len(partition.test_document_ids),
            },
            "train_document_ids": list(partition.train_document_ids),
            "validation_document_ids": list(partition.validation_document_ids),
            "test_document_ids": list(partition.test_document_ids),
            "execution_status": (
                "dry_run_only"
                if dry_run
                else "preflight_complete_training_not_started"
            ),
        }
    )
    return resolved


def write_preflight_artifacts(
    preflight: dict[str, Any], output_dir: str | Path
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.json"
    provenance_path = output / "provenance.json"

    resolved_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        key: preflight[key]
        for key in (
            "experiment_name",
            "schema_mode",
            "encoder_model",
            "encoder_revision",
            "split_seed",
            "seed",
            "fold",
            "selection_scope",
            "config_sha256",
            "manifest_sha256",
            "dataset_sha256",
            "dataset_status",
            "execution_status",
        )
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "resolved_config": resolved_path,
        "provenance": provenance_path,
    }


def _reject_test_label_selection(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            child_path = f"{path}.{key}"
            if "test" in normalized and any(
                marker in normalized
                for marker in (
                    "threshold",
                    "select",
                    "tuning",
                    "tuned",
                    "calibrat",
                )
            ):
                raise ValueError(
                    "test-label threshold/selection field is forbidden in "
                    f"Gate A: {child_path}"
                )
            if (
                normalized
                in {"selection_split", "tuning_split", "threshold_split"}
                and str(child).lower() == "test"
            ):
                raise ValueError(
                    f"test split cannot be used for selection: {child_path}"
                )
            _reject_test_label_selection(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_test_label_selection(child, f"{path}[{index}]")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    preflight = build_preflight(
        args.config,
        args.manifest,
        fold=args.fold,
        dataset_path=args.dataset,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    output_dir = args.output_dir or preflight["output_dir"]
    paths = write_preflight_artifacts(preflight, output_dir)
    print(
        json.dumps(
            {
                "status": preflight["execution_status"],
                "resolved_config": str(paths["resolved_config"]),
                "provenance": str(paths["provenance"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
