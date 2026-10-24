# Approved matched70 seed43/44 repeats

User approved continuing the next step after the baseline/cap96 real-forward confirmation on 2026-10-10.

Bounded design: retain seed42 gates/results and both downstream baselines. A separate CLI accepts only43/44 and dev; immutable document split stays fold1/splitseed11800. Train cap128 fresh from pinned upstream pretrained weights for12epochs. Select checkpoint and relation threshold using the existing baseline relation-first rule. Evaluate cap96 at that exact checkpoint and baseline-selected threshold, without threshold reselection. No test/full mode and no negative-sampling changes. Seed42 threshold remains0.9; seed43/44 thresholds follow their baseline rule and need not equal0.9.

1. Restore exact published tree9bbbe8d67fd818b981b1b4aa970aac51b09e2af6 after scratch reset; verify archived results and pretrained weight hashes.
2. Add isolated seed CLI and pure contract helpers with RED/GREEN tests. Preserve all old guards. Derive per-run config/package with new seed and config hash, referencing original seed42 preflight package by hash. Require complete matching seed42 dev evidence. Atomic epoch recovery and writer lock stay enabled.
3. Run regression suite and independent review; commit source before training so recovery identity remains stable.
4. Four dev runs: roberta43/44,securebert43/44. Each output directory separate. Engineering preflight/overfit/smoke from seed42 are reused as package checks, never mislabeled as new per-seed experiments.
5. Evaluate each selected checkpoint at caps128/96 and baseline-selected threshold. Combine with seed42; report all runs, mean/sample-SD, paired deltas and individual failures. Three seeds on five validation documents do not establish significance or generalization.
6. Save selected checkpoints and outputs, publish report into existing draftPR; no merge.

Review focus: seed must propagate into model RNG, negative sampling, shuffling, config hash and checkpoint/recovery metadata; split seed must not change. Prevent full/test and unknown seeds. Original seed42 artifacts must not be overwritten. New runs initialize pretrained, not seed42 fine-tuned. Resume verifies exact identity and original baseline evidence. Report incomplete work honestly.
