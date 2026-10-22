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
