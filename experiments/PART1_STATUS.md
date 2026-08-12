# Reviewer requirement 1: current status

## Completed

- Recovered the 52-report Label Studio corpus and source-document identifiers.
- Built deterministic five-fold assignments at report level.
- Balanced folds approximately using entity and relation label counts.
- Verified zero train/validation/test overlap in every outer iteration.
- Verified that each report is held out as test data exactly once.
- Generated an auditable document manifest, nested split files, a manuscript
  methods paragraph, and a strict result aggregator for 50 core runs.

Fold sizes are 11, 11, 10, 10, and 10 reports. Their entity/relation counts are:

| Fold | Reports | Entity mentions | Relation instances |
| ---: | ---: | ---: | ---: |
| 0 | 11 | 314 | 121 |
| 1 | 11 | 264 | 118 |
| 2 | 10 | 281 | 109 |
| 3 | 10 | 267 | 118 |
| 4 | 10 | 276 | 108 |

## Required before inserting numerical claims

The recovered public/prototype training pipeline cannot reproduce the paper's
reported end-to-end results:

1. `train_100epochs.py` ignores the annotated relation types and assigns
   `rel_labels=0` to every example.
2. It uses a random 80/10/10 split over 52 processed records rather than the
   new document-fold manifest.
3. It saves a checkpoint but performs no final exact-match entity/relation
   evaluation.
4. The repository does not contain a faithful runnable STIXnet-only baseline
   or the complete end-to-end evaluation code used to obtain Table IV.
5. The archived offline result is explicitly a token-embedding fallback and is
   not comparable to the RoBERTa-BiGRU-CRF model or the paper.

Therefore the new cross-validation F1 values must not be inferred from the old
single-split values. The original GPU training/evaluation implementation that
generated Table IV, including STIXnet-only and full-hybrid predictions, is
required to complete the 50-run matrix.

Expected matrix:

```
5 outer folds x 5 seeds x 2 systems = 50 runs
```

Populate `cv_results_template.csv` only from saved test-fold predictions and
exact-match evaluation. Then run:

```bash
python experiments/aggregate_cv_results.py \
  --input path/to/completed_cv_results.csv \
  --output-dir experiments/cv_results
```

The aggregator rejects missing, duplicated, unexpected, or out-of-range runs.
