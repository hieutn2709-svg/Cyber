# Deprecated and unsupported historical artifacts

Files in this directory are retained solely for audit history. They are not
active configurations, datasets, results, or evidence for current claims.

- `unsupported_cv_results.csv` contains a former 50-row result matrix without
  adequate run-level provenance for the reported values.
- `unsupported_ultimate_run_fixed/` contains a former run bundle with
  internally unverifiable commit/hash/environment claims and unsupported
  metrics.
- `legacy_train_model.ipynb` uses a random record-level split, legacy training
  settings, and combined entity/role labels.
- `legacy_train_bieos.json` stores the corresponding combined-role label export
  and legacy numeric label identifiers.
- `legacy_aggregate_cv_results.py` and `legacy_cv_results_template.csv` encode
  the unsupported former 50-run aggregation protocol.

Do not cite, aggregate, or copy values from this directory into active
documentation. The files are intentionally preserved verbatim so repository
history remains inspectable.
