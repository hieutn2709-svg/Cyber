# Document-level five-fold evaluation

Reviewer requirement 1 is addressed with five leakage-safe document folds.
The fold assignment is deterministic and uses entity/relation counts only to
improve balance.  It never splits a report into multiple partitions.

Build the manifest:

```bash
python experiments/build_document_cv.py \
  --annotations Data/Annotations.json \
  --output-dir experiments/cv_manifest \
  --folds 5 \
  --seed 42
```

The generated files are:

- `document_folds.json`: complete auditable manifest;
- `document_folds.csv`: one row per report;
- `fold_summary.csv`: fold-level corpus counts;
- `outer_splits/outer_fold_0.json` ... `outer_fold_4.json`: nested
  train/validation/test assignments.

For outer fold `k`, fold `k` is test data. Five complete reports are selected
as validation data only from the remaining four folds; all other non-test
reports are training data. The same split manifest must be used for
STIXnet-only and the full hybrid. Run each configuration with seeds `42`,
`123`, `2024`, `3407`, and `777`. This produces 50 core runs:

```
5 outer folds x 5 model seeds x 2 systems = 50 runs
```

Hyperparameters and reconciliation settings must be selected from the
validation partition only.  Test-fold predictions must be saved for exact-
match scoring and document-level audit.  Do not infer cross-validation results
from the earlier single-split table.

After all 50 core runs have produced exact-match metrics, populate a copy of
`cv_results_template.csv` and aggregate it with:

```bash
python experiments/aggregate_cv_results.py \
  --input path/to/completed_cv_results.csv \
  --output-dir experiments/cv_results
```
