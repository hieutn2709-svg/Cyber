# Reviewer reproducibility status

## Verified repository records

- The corpus contains 52 complete reports.
- The official outer manifest uses split seed 11800.
- The archived run partitions use model/evaluation seed 42.
- Fold sizes are 11, 11, 10, 10, and 10 reports.
- Train, validation, and test identifiers are pairwise disjoint in each outer
  iteration, and every report is held out exactly once.
- The active model configuration records the reviewed optimization and
  early-stopping settings.
- Task-loss weights are 0.80/0.05/0.15 for entity/role/relation.
- The active tag inventory is 61 sequence tags plus a separate four-class role
  head.
- Primary result claims are restricted to the supported baseline, V10, and V13
  rows in `results/current_results.csv`.
- Fold-level results, diagnostics, per-class tables, and V13 test predictions
  are present under `results/`.

## Interpretation

The rule/knowledge-based comparator is a same-split, STIXnet-inspired
reproduction. It is not the complete original STIXnet system. Original STIXnet
literature scores must be cited as literature-reported results and must not be
presented as outputs of this repository.

V11 and V12 are retained as sequential development diagnostics, not independent
confirmatory ablations. Within-fold model and threshold selection is
validation-only, but later variant designs were informed by earlier outcomes on
the same corpus.

## Remaining limits

Large checkpoints and run logs are linked from the root README and are shared
read-only. The repository still lacks immutable environment/base-model revision
locks, a complete checkpoint-hash inventory, predictions for every earlier
variant, and complete third-party licensing records. No missing value should be
inferred or synthesized.
