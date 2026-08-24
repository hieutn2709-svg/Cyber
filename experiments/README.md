# Reviewer experiment protocol

The controlled evaluation uses the exact archived V10 document-level manifest
for every configuration. The split seed is `11800`, the run seed is `42`, and
the five test folds contain 11, 11, 10, 10, and 10 complete CTI reports.

## Build or check the manifest

```bash
python experiments/build_document_cv.py \
  --annotations data/Annotations.json \
  --output-dir experiments/cv_manifest \
  --folds 5 \
  --seed 11800

python experiments/build_document_cv.py --check
```

The builder reads `cv_manifest/fold_manifest_v4.json` and
`cv_manifest/run_partitions_seed_42.json`, validates their mapping against
annotation order, and derives:

- `document_folds.json`: full assignment, split seed, and core-type inventory;
- `document_folds.csv`: one row per report;
- `fold_summary.csv`: raw annotation and clean evaluable fold-level counts;
- `outer_splits/outer_fold_0.json` through `outer_fold_4.json`: disjoint
  train/validation/test document identifiers.

In outer iteration `k`, fold `k` is test data. The five validation documents
are the saved run-time selections from the non-test documents. All remaining
non-test reports are training data. Test documents are not used for model,
checkpoint, decoding, reconciliation, or threshold selection.

## Configuration

The canonical reviewer configuration is
`Configs/reviewer_experiment.json`: RoBERTa, a two-layer BiGRU with hidden size
250, eight attention heads, dropout 0.35, encoder LR `3e-5`, head LR `3e-4`,
weight decay 0.01, joint maximum 30 epochs/patience 5, and relation refinement
12 epochs/patience 4. It also records sequence length 512, physical batch 2,
gradient accumulation 2, emission CE weight 1.5, O-class cap 0.25, relation
distance 96, and entity/role/relation task weights 0.80/0.05/0.15.

The sequence tag head has 61 labels. Relation roles use a separate four-class
head (`O`, `ROLE_1`, `ROLE_2`, `ROLE_BOTH`).

## Results

`results/current_results.csv` is the only primary results table. It contains:

- same-split rule/KB baseline: EF1 0.289, RF1 0.011;
- V10 contextual neural: EF1 0.698, RF1 0.219;
- V13 conservative hybrid: EF1 0.710, RF1 0.222.

`results/variant_provenance.csv` retains V10--V13 lineage. V11 and V12 are
development-stage diagnostics because later variants were informed by earlier
experimental outcomes on the same corpus.

The fold-level CSVs, diagnostics, per-class tables, and V13 test predictions
are enumerated in `results/README.md`.

The old 50-row results matrix and fixed-run bundle are unsupported as current
evidence and have been moved verbatim to `archive/deprecated/`.
