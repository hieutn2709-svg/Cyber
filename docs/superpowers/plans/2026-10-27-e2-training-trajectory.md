# E2 training trajectory observation — frozen 2026-10-27

## Purpose and scope
Measure directed boundary-pair label collisions during actual optimizer training,
separately from the completed fixed-checkpoint snapshot. This is a new diagnostic
run, not recovered historical batches and not an intervention comparison.

## Frozen execution
- Source parent: 96be16890ef956dba142f3550d19ed4652a20299. Run at the commit
  adding this protocol; hold HEAD fixed until both runs finish.
- Existing matched encoder CLI, packages roberta and securebert; mode dev only.
- Existing matched70 corpus lock, package preflight and archived overfit/smoke
  prerequisites must pass, including pinned pretrained file SHA-256 checks.
- Initialize from original pinned pretrained encoder weights, not selected
  finetuned weights. Fold 1, seed 42, split seed 11800; unchanged settings.
- 12 epochs each; dev has no early stopping. Select checkpoint and threshold
  using the existing validation rule/grid. No full/test forward or scoring.
- CPU, four Torch threads, one interop thread; deterministic settings from the
  existing runner. Same torch 2.14.1+cpu and transformers 5.18.0.
- Enable --train-pair-diagnostics; retain existing sampler, objective, candidates,
  optimizer, accumulation, threshold selection, annotations and partition.
- Separate output directories: e2_trajectory/runs/roberta and securebert.
  Atomic recovery is used only for an incomplete run with matching identity.
- Run sequentially to bound memory. Persist outputs and selected checkpoints.

## Measurements and checks
For every epoch and package, require 49 train window records from the same
36 train documents. Aggregate positive/negative pairs, boundary groups, collision
groups, collision positive/negative instances, windows/documents affected, source
self edges and long positives separately for eligible pools and sampled batches.
Report explicit denominators; repeated windows/epochs are not independent samples.
Check each history epoch, seed schedule, hashes and split membership; verify
completed summary says dev_complete and test_evaluated=false. Preserve per-window
records and sampled-order digests, selected checkpoints, environment and config.
Keep validation quality metrics descriptive and label these as new runs; do not
assert exact historical numerical replay, causal benefit, significance, general
encoder ranking or CUDA parity. Existing synthetic parity tests do not establish
real pretrained backward parity. No on/off duplicate training is claimed.
Do not modify gold; human adjudication remains pending.

## Execution ledger
1. Restore version2 matched70 archive and smoke checkpoints; verify corpus and
   prerequisites; restore exact pretrained assets; run repository tests.
2. Execute both runs at frozen source with diagnostics; retain epoch recovery.
3. Aggregate and verify all 24 epoch histories; inspect selected checkpoints and
   document limits. Obtain independent review of evidence and reporting.
4. Save reproducibility report and selected checkpoints, update draft PR5.
   Do not merge or perform test evaluation.

## Recovery ruling
The previous scratch environment was pruned before a new trajectory was started.
Restore completed source artifacts; do not repeat prior E1/D0/snapshot experiments.
Historical engineering gates are reused with their original provenance.
