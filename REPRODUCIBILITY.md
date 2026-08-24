# Reproducibility and provenance notes

## Scope

This repository supports a controlled, same-split comparison on 52 CTI
documents. It does not reproduce the complete original STIXnet implementation
and does not claim that literature-reported STIXnet scores were recreated.

## Fixed data protocol

- Outer split seed: `11800`.
- Model/evaluation run seed: `42`.
- Five document-level test folds: `11, 11, 10, 10, 10` documents.
- A report is assigned to exactly one outer fold; its text and annotations are
  never split across partitions.
- In each outer iteration, validation documents come only from the non-test
  outer-training partition.
- The ten core evaluation types are recorded in the manifest and configuration.

The source of truth is the exact run-time numeric-index manifest
`experiments/cv_manifest/fold_manifest_v4.json`, recovered from
`models_checkpoints/reviewer_v10`. The stable document-ID partitions saved in
`run_partitions_seed_42.json` were recovered from each fold's `result.json`.
`document_folds.json`, `document_folds.csv`, `fold_summary.csv`, and
`outer_splits/` are derived from those two archived records plus the annotation
task order.

Verify the mapping without changing the checked-in files:

```bash
python experiments/build_document_cv.py --check
```

To rewrite only the derived files from the archived run manifests, omit
`--check` and review the complete diff:

```bash
python experiments/build_document_cv.py \
  --annotations data/Annotations.json \
  --output-dir experiments/cv_manifest \
  --folds 5 \
  --seed 11800 \
  --run-manifest experiments/cv_manifest/fold_manifest_v4.json \
  --run-partitions experiments/cv_manifest/run_partitions_seed_42.json
```

The previous public manifest was produced by a different label-balancing
algorithm that also used the integer 11800. It was not the run allocation and
has been replaced. A seed identifies a pseudorandom stream only together with
its algorithm and input ordering.

Both manifests cover the same 52 reports, but no former fold 0--4 is identical
to any V13 run fold 1--5, so the run folds cannot be obtained by renumbering the
former folds. V13 fold `k` is the archived numeric fold at index `k-1`; the
corrected derived split is stored as `outer_splits/outer_fold_{k-1}.json`.

The actual run-fold raw relation counts are `134, 106, 138, 111, 85`; clean
evaluable counts are `128, 106, 137, 110, 83`. Fold 3 therefore has 138 raw
relations and 137 retained by the clean windowed dataset. V13's 23 TP and 114
FN are consistent with that retained count.

## Model and threshold selection

For every outer fold, joint-training early stopping, relation-refinement early
stopping, decoding parameters, reconciliation choices, and thresholds are
selected using that fold's validation documents only. The selected state is
frozen before the held-out test documents are evaluated. V11, V12, and V13 were
developed sequentially, so later variants were informed by earlier corpus-level
experimental outcomes even though their within-fold thresholds were selected
on validation data.

## Result provenance

`experiments/results/current_results.csv` is the active primary summary table.
It records the same-split baseline, V10, and V13 values supported by the
reviewer experiment record. The same directory now also contains fold-level
baseline/V10/V11/V12/V13 tables, gold-span and gold-pair diagnostics, per-class
P/R/F1 tables, and saved V13 test predictions. `variant_provenance.csv`
preserves the V10--V13 lineage and labels V11/V12 as development diagnostics.

The former `experiments/cv_results.csv` and `experiments/ultimate_run_fixed/`
files did not contain a trustworthy link from configuration, repository commit,
checkpoint, predictions, and metrics. They are retained verbatim in
`archive/deprecated/` and must not be cited as experimental evidence.

## Remaining provenance gaps

The repository or linked artifact set does not yet provide:

- immutable dependency and base-model revision locks;
- a checked-in cryptographic hash inventory for every Drive checkpoint;
- raw test predictions for every variant (V13 predictions are checked in;
  earlier variants remain in the linked run folders where available);
- a faithful runnable copy of the complete original STIXnet system;
- third-party license files for every redistributed dataset/model artifact.

Until those artifacts are supplied, do not infer missing precision/recall,
per-fold values, hardware/runtime claims, or statistical tests from the means
and standard deviations in the current result table.
