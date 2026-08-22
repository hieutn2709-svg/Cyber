# Reproducibility and provenance notes

## Scope

This repository supports a controlled, same-split comparison on 52 CTI
documents. It does not reproduce the complete original STIXnet implementation
and does not claim that literature-reported STIXnet scores were recreated.

## Fixed data protocol

- Outer split seed: `11800`.
- Five document-level test folds: `11, 11, 10, 10, 10` documents.
- A report is assigned to exactly one outer fold; its text and annotations are
  never split across partitions.
- In each outer iteration, validation documents come only from the non-test
  outer-training partition.
- The ten core evaluation types are recorded in the manifest and configuration.

Generate and verify the manifest without changing the checked-in files:

```bash
python experiments/build_document_cv.py --check
```

To intentionally regenerate the official files after changing the source
annotations or builder, omit `--check` and review the complete diff:

```bash
python experiments/build_document_cv.py \
  --annotations data/Annotations.json \
  --output-dir experiments/cv_manifest \
  --folds 5 \
  --seed 11800
```

## Model and threshold selection

For every outer fold, joint-training early stopping, relation-refinement early
stopping, decoding parameters, reconciliation choices, and thresholds are
selected using that fold's validation documents only. The selected state is
frozen before the held-out test documents are evaluated. V11, V12, and V13 were
developed sequentially, so later variants were informed by earlier corpus-level
experimental outcomes even though their within-fold thresholds were selected
on validation data.

## Result provenance

`experiments/results/current_results.csv` is the only active primary results
table. It records the same-split baseline, V10, and V13 values supported by the
reviewer experiment record. `variant_provenance.csv` preserves the V10--V13
lineage and labels V11/V12 as development diagnostics.

The former `experiments/cv_results.csv` and `experiments/ultimate_run_fixed/`
files did not contain a trustworthy link from configuration, repository commit,
checkpoint, predictions, and metrics. They are retained verbatim in
`archive/deprecated/` and must not be cited as experimental evidence.

## Remaining provenance gaps

The repository does not currently contain:

- immutable dependency and base-model revision locks;
- independently verifiable V10--V13 training logs and checkpoint hashes;
- per-fold validation selections, frozen thresholds, and test predictions for
  every reported variant;
- a faithful runnable copy of the complete original STIXnet system;
- third-party license files for every redistributed dataset/model artifact.

Until those artifacts are supplied, do not infer missing precision/recall,
per-fold values, hardware/runtime claims, or statistical tests from the means
and standard deviations in the current result table.
