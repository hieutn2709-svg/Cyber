# Result provenance

`current_results.csv` contains the only active primary numerical claims. The
three rows share the official document-level fold manifest. Model and threshold
selection was validation-only within each outer fold.

`variant_provenance.csv` preserves the V10--V13 development sequence. V11 and
V12 values are retained to document provenance, but are classified as
development diagnostics rather than independent confirmatory ablations because
later variants were designed after earlier outcomes on the same corpus were
known.

These tables do not imply the presence of per-fold predictions, checkpoints,
or run logs. Those artifacts are currently missing. No unrecorded precision,
recall, per-fold, hardware, or runtime values should be inferred.
