# Result provenance

`current_results.csv` contains the active primary summary claims. The three rows
share the archived V10 run manifest now translated to stable document IDs.
Model and threshold selection was validation-only within each outer fold.

`variant_provenance.csv` preserves the V10--V13 development sequence. V11 and
V12 values are retained to document provenance, but are classified as
development diagnostics rather than independent confirmatory ablations because
later variants were designed after earlier outcomes on the same corpus were
known.

Fold-level evidence in this directory:

- `rule_kb_baseline_per_fold.csv`
- `contextual_neural_per_fold.csv`
- `naive_hybrid_per_fold.csv`
- `per_type_calibrated_hybrid_per_fold.csv`
- `five_fold_results_conservative_hybrid.csv`
- `five_fold_summary.csv`
- `gold_span_diagnostic_per_fold.csv`
- `gold_pair_type_accuracy_per_fold.csv`
- `per_class/entity_per_class_by_fold.csv`
- `per_class/relation_per_class_by_fold.csv`
- `predictions/conservative_hybrid/fold_*_seed_42/test_predictions.jsonl`

For the five run folds, raw annotation relation counts are
`134, 106, 138, 111, 85`, whereas clean evaluable counts are
`128, 106, 137, 110, 83`. The fold-level V13 CSV uses the latter denominator.
Large checkpoints and logs are linked in the root README. No unrecorded
hardware, runtime, or missing-variant prediction value should be inferred.
