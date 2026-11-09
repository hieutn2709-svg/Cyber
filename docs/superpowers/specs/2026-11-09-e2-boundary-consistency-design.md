# E2 boundary-consistency ablation

User instruction: continue implementation of the next CTIQ2 steps after completed E2 observations. This document records implementation choices, not new empirical results. Parent:63a31ba67089ea2bb7bf0016ff91f9f0fa4e0921.

## Intent and approaches
Measure a single training-negative intervention before hard-negative mining. Prefiltering the eligible pool would change many draws even in windows whose original batch has no collision. Dropping drawn conflicts without replacement would reduce the negative budget. Chosen implementation: reproduce the existing draw, retain every safe sampled negative, replace only sampled negatives sharing a directed document/boundary key with a selectable positive. Draw replacements uniformly without replacement from safe, previously unsampled eligible negatives using the same local RNG after the baseline draw; restore original eligible-pool order. If insufficient replacements exist, keep all available safe negatives and report the shortfall; never duplicate, relabel or broaden distance constraints.

## Contract
- Opt-in `--relation-negative-policy boundary_replace`; default `typed` retains the exact original algorithm/output/metadata, including recovery compatibility at unchanged source.
- Boundary keys contain document ID and ordered source/target start/end, excluding types. Preserve reverse-direction negatives unless that direction itself has a positive. Preserve every positive, including long-distance positives; existing self-edge omission unchanged.
- Never change entity candidates, loss weights, architecture, inference, gold, threshold selection, seed schedule or global RNG.
- Diagnostics retain schema1 `eligible` as the original pre-intervention pool and `sampled` as the actual loss batch; opt-in policy metadata adds original sampled conflicts removed, replacements, shortfall, safe-pool size and baseline batch digest. Zero collision required for final sampled batch.
- Policy is recorded in run config, recovery identity, selected checkpoint and run summary; mismatched policy cannot resume. Generic full/test execution rejects this policy; matched wrappers remain dev/smoke/overfit only.
- Run fixture loss/backward/optimizer/recovery tests and an independent exhaustive sampler oracle before claiming engineering readiness. Do not call fixture tests pretrained training or quality evidence.

## Scientific run protocol
First comparison: both encoders × typed/boundary_replace × seed42,12 dev epochs, fold1/splitseed11800, CPU4/interop1, diagnostics enabled. Initialize all four runs from their pinned pretrained weights, run all arms in the same runtime/source, keep matched70 gold v1 and original validation grid. Historical trajectories are context only, not the new paired control. Freeze protocol before execution; report every run including declines. If a budget shortfall occurs, report its magnitude as an intervention consequence.

Report strict relation/entity P/R/F1, TP/FP/FN, candidate coverage, conflict/removal/replacement/shortfall counts, and optimizer state consistency. Five validation documents are exploratory and used for selection. No significance/generalization/causal encoder-ranking claim; no full/test or merge. Human adjudication remains pending. Larger multi-seed execution follows the pilot, not adaptive tuning on validation.

## Current deliverable
Implement, test and independently review the policy and all CLI/recovery wiring; commit reproducible protocol and engineering report to the existing draft PR. New pretrained experiment execution must be separately reported with actual artifacts; this engineering delivery does not invent its results.
