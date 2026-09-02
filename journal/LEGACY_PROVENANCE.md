# Legacy V10/V13 provenance boundary

This file defines how the journal extension treats the archived V10 contextual
model and V13 conservative hybrid results. Its purpose is to prevent a future
journal manuscript from overstating reproducibility of legacy training code.

## Supported legacy claims

The following same-split results remain supported by checked-in fold-level
artifacts and saved predictions:

| System | Entity F1 | Relation F1 | Journal role |
| --- | ---: | ---: | --- |
| Rule/KB same-split baseline | 0.289 | 0.011 | archived comparator |
| V10 contextual neural | 0.698 | 0.219 | archived comparator |
| V13 conservative hybrid | 0.710 | 0.222 | archived comparator |

The fixed evaluation protocol uses 52 documents, five document-level outer
folds of 11/11/10/10/10 documents, split seed 11800, and run seed 42. The
checked-in manifest and result tables remain the source of truth for these
numbers.

## Evidence available for V10

The linked `models_checkpoints/reviewer_v10` archive contains fold-level result
records, histories, relation-refinement histories/summaries, saved test
predictions, and checkpoints. The fold-1 run log records:

- `max_epochs=12`;
- a validation-selected entity decoder and relation threshold;
- a dedicated `REL-REFINE` stage;
- early stopping after four stale refinement epochs;
- final held-out entity/relation metrics.

The fold-1 `result.json` identifies the run as script version
`reviewer-v4-v10`, records `epochs_requested=12`, and stores the selected
relation-refinement epoch and threshold. The archived resolved training
specification records the relation existence/type and refinement settings,
including negative sampling, focal-loss parameters, a 96-token maximum
relation distance, relation-existence/type loss weights, and a 12-epoch
relation-refinement cap.

## Source-code boundary

The current public `Joint_model/joint_model.py` is not sufficient by itself to
re-run the complete V10 relation path. It exposes the sequence-tag and token
role heads but does not contain the full relation-existence/type/refinement
implementation evidenced by the archived run records.

A forensic check was performed over:

1. the current journal branch;
2. the repository tree from the 2026-08-16 public upload commit;
3. the directly accessible root of the Drive `reviewer_v10` run archive;
4. the directly accessible `reviewer_v10/pilot_runs/fold_1_seed_42` artifacts;
5. the earlier `reviewer_v4_fixed` artifact directory.

Those locations contain the notebook-era prototype, run configuration,
checkpoints, predictions, histories, diagnostics, and result records, but no
verified source snapshot that can presently be tied one-to-one to the complete
`reviewer-v4-v10` relation-refinement execution.

Therefore the journal project MUST NOT recreate missing V10 relation code from
memory or from hyperparameters alone and then describe it as the original run.
If an exact source snapshot is recovered later, it must be verified against the
archived predictions/metrics before its status is upgraded.

## Journal policy

Until that verification exists:

- V10 and V13 are **archived comparators**, evaluated from their preserved
  predictions and fold-level evidence;
- their numerical results remain valid only for the documented fixed manifest
  and scoring protocol;
- they are not described as clean-room end-to-end reproducible training
  pipelines;
- no missing code is reconstructed merely to strengthen a reproducibility
  claim;
- all new journal methods (SpanPair and SCSP variants) must be fully runnable
  from checked-in source, immutable configuration, model/data revisions, and
  saved raw predictions.

## Separation from the new journal contribution

The journal contribution is intentionally independent of the incomplete legacy
training source. V10/V13 provide historical same-split comparison points. The
new SCSP implementation will have its own source tree, tests, configuration,
run metadata, calibration artifacts, and predictions. This separation makes it
possible to improve reproducibility rather than inheriting undocumented legacy
behavior.
