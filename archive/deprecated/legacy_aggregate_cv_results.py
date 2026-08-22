#!/usr/bin/env python3
"""Validate and aggregate five-fold, five-seed experiment results.

The script intentionally refuses incomplete matrices so that single-split or
partially completed runs cannot be reported as five-fold evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SYSTEMS = ("stixnet-only", "hybrid")
SEEDS = (42, 123, 2024, 3407, 777)
FOLDS = tuple(range(5))
METRICS = (
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "relation_precision",
    "relation_recall",
    "relation_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cv_results"))
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"system", "outer_fold", "seed", *METRICS}
    if not rows:
        raise ValueError("The result file contains no runs")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    parsed = []
    for line_number, row in enumerate(rows, start=2):
        system = row["system"].strip().lower()
        if system not in SYSTEMS:
            raise ValueError(f"Line {line_number}: unsupported system {system!r}")
        fold = int(row["outer_fold"])
        seed = int(row["seed"])
        if fold not in FOLDS or seed not in SEEDS:
            raise ValueError(f"Line {line_number}: unexpected fold/seed ({fold}, {seed})")
        values = {metric: float(row[metric]) for metric in METRICS}
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError(f"Line {line_number}: metrics must be within [0, 1]")
        parsed.append({"system": system, "outer_fold": fold, "seed": seed, **values})
    return parsed


def validate_matrix(rows: list[dict]) -> None:
    observed = [(row["system"], row["outer_fold"], row["seed"]) for row in rows]
    expected = [(system, fold, seed) for system in SYSTEMS for fold in FOLDS for seed in SEEDS]
    duplicates = sorted({key for key in observed if observed.count(key) > 1})
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if duplicates or missing or unexpected or len(rows) != len(expected):
        raise ValueError(
            "Incomplete or duplicated result matrix: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}, "
            f"rows={len(rows)}, expected={len(expected)}"
        )


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate(rows: list[dict]) -> tuple[list[dict], dict]:
    by_fold: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_fold[(row["system"], row["outer_fold"])].append(row)

    fold_means = []
    for system in SYSTEMS:
        for fold in FOLDS:
            group = by_fold[(system, fold)]
            record = {"system": system, "outer_fold": fold, "n_seeds": len(group)}
            record.update({metric: statistics.fmean(row[metric] for row in group) for metric in METRICS})
            fold_means.append(record)

    summary: dict[str, dict] = {"systems": {}, "paired_difference_hybrid_minus_stixnet": {}}
    for system in SYSTEMS:
        system_rows = [row for row in fold_means if row["system"] == system]
        summary["systems"][system] = {
            metric: mean_sd([row[metric] for row in system_rows]) for metric in METRICS
        }

    for metric in METRICS:
        baseline = {
            row["outer_fold"]: row[metric]
            for row in fold_means
            if row["system"] == "stixnet-only"
        }
        hybrid = {
            row["outer_fold"]: row[metric]
            for row in fold_means
            if row["system"] == "hybrid"
        }
        differences = [hybrid[fold] - baseline[fold] for fold in FOLDS]
        summary["paired_difference_hybrid_minus_stixnet"][metric] = {
            **mean_sd(differences),
            "per_fold": differences,
            "positive_folds": sum(value > 0 for value in differences),
            "negative_folds": sum(value < 0 for value in differences),
            "zero_folds": sum(value == 0 for value in differences),
        }
    return fold_means, summary


def format_metric(value: dict[str, float]) -> str:
    return f"{value['mean']:.3f} $\\pm$ {value['sample_sd']:.3f}"


def write_outputs(fold_means: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fold_means.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["system", "outer_fold", "n_seeds", *METRICS])
        writer.writeheader()
        writer.writerows(fold_means)
    (output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"System & EP & ER & EF1 & RP & RR & RF1 \\",
        r"\hline",
    ]
    for system, label in (("stixnet-only", "STIXnet-only"), ("hybrid", "Hybrid (ours)")):
        values = [format_metric(summary["systems"][system][metric]) for metric in METRICS]
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    (output_dir / "cv_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    validate_matrix(rows)
    fold_means, summary = aggregate(rows)
    write_outputs(fold_means, summary, args.output_dir)
    print(f"Validated and aggregated {len(rows)} runs into {args.output_dir}")


if __name__ == "__main__":
    main()
