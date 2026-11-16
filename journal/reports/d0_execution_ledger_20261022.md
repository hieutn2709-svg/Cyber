# D0 execution ledger — 2026-10-22

Authority: `matched70_next_experiments_20261019.md`, D0 section; user authorized continuation. Base commit 5055089de52ea170d2dca33fa97fa3a56fd29f07. Work happens in a fresh isolated clone of the existing draft branch.

1. Restore six selected checkpoints, source inputs and prior review; verify original SHA values. Completed: six checkpoint hashes and 167 archive entries match; outer archives pass CRC; two pinned encoder config hashes match.
2. Reconstruct optional diagnostic hook and tests, run the full test suite, publish the code checkpoint.
3. Run six selected checkpoints × caps128/96 on five validation documents only. Require full on/off WindowInference equality, unchanged weights/RNG, archived prediction identities/order/gold and strict metrics. Export all candidate posteriors and all relation logits.
4. Analyze all120 primary mentions per configuration and the22 persistent relations. Reconcile prior outcome stages, distinguish repeated endpoints from unique mentions, report top2 as retrospective gold-informed coverage.
5. Review, save reproducibility archive and reports, update draft PR. No merge, training, full/test or gold edits.

Recovery note: Oct19/Oct20 transient D0 code/results were lost before durable saving. They are not current deliverables. This run reconstructs and re-verifies them; historical reports/checkpoints remain unchanged.

Ruling: numerical archive comparison and logging on/off parity are separate gates. Exact archived scores were not reproducible in the interrupted Oct20 attempt. A post-observation absolute guard of1e-5 was introduced then and also failed for SecureBERT42. Preserve it as a reported failure, never enlarge it to fit observations. Explicit diagnostic-only mode may measure drift while still requiring identical decisions/metrics and exact on/off parity. Its status must disclose any failed score guard; it does not resolve the source of numerical drift. Cost if wrong: diagnostics may reflect a numerically different execution; no quality improvement or exact archived reproducibility will be claimed.

Pre-flight: hook emits local inclusive token spans; runner supplies global token/character coordinates; analyzer consumes emitted coordinates. NONE + inventory order determines all probabilities; ranking uses the official pruner. Archived predictions, not the known truncated SecureBERT42 cap128 gzip, are the reference. New full logits will be dated as new output, not restored missing bytes.


## Final delivery checkpoint

- Restored published D0 code after another transient session reset. Earlier unsaved fixes/results were not counted as final.
- Independent review identified incomplete identity/config gates and damaged gzip streams. Added focused tests, watched5 failures, implemented gates, atomic gzip roundtrip verification and explicit blocked-case capture. Full suite:257 passed,65 subtests. Published code a0992d7fc32d8f173a808552aa364bbf5a9adf1b.
- Ruling: keep default complete-matrix analysis; an explicit partial mode may report only verified cases and requires all six cap128 baselines. It never accepts failed prediction identities. Cost if wrong: incomplete cap96 observations could be mistaken for a complete comparison; matrix_complete and missing_configurations are mandatory. Final delivery uses strict complete-matrix mode, not partial mode.
- Fresh final six-checkpoint run completed12/12 cases. All current on/off inference, weights and RNG match; all archived prediction identities/order and exported probabilities match exactly with max delta0. Every new gzip stream fully reads and matches recorded SHA/count. Earlier numerical drift cause remains unresolved; no bound was enlarged.
- Final analysis:97,848 proposals;292,272 pairs;1,440 primary observations;528 persistent endpoint observations;264 persistent relation observations. Cap128 endpoint reasons174 retained,52 wrong type,29 threshold,9 cap.34 unique character-identified mentions; top2 at retained boundaries210/264 is gold-informed only.
- Final: minor (deferred): analyzer does not independently regenerate pair sets/distances; it crosschecks snapshot pairs against exported logits, while full on/off inference parity preserves official pair-generation behavior. No missing pair was observed.
- Original truncated historical gzip unchanged. New complete SecureBERT42 cap128 logits comprise34,136 pairs. No training/full/test/gold edit/merge.
