# E1-MULTI validation ablation — 2026-10-24

Implemented gold-free offline decoding of retained top-2 entity alternatives over frozen D0 pairs. Prediction-only artifacts are written and hashed before gold is read. The existing strict scorer is reused.

All six cap128 configurations reproduce the entire top1 baseline prediction payload exactly (order, metadata, identities and scores). Top1 is a subset of top2. No new training/forward, gold changes or full/test evaluation.

| Encoder | Mean relation F1 top1 | Mean relation F1 top2 | Paired delta (percentage points) |
|---|---:|---:|---:|
| RoBERTa | 29.916% | 11.987% | -17.929 |
| SecureBERT | 24.271% | 9.580% | -14.691 |

Mean is over three seeds; all six individual F1 values fall. Summed repeated observations: TP 103→113, FP 377→1807, predictions480→1920. Do not pool these as independent samples. Unrestricted top2 expansion is rejected as a quality improvement; selective/consistent decoding remains a separate untested proposal.

Verification: 15 relevant tests pass, including 11 new tests. Independent review verified all18 input hashes, all12 prediction hashes, exact baseline parity, subsets and strict rescoring; no important findings. Full repository discovery is not green in this runtime: baseline218 tests and changed229 tests both have the same20 failures,26 errors,12 skips, with missing torch/tokenizers/pytest. This does not establish neural regression coverage. An initial aborted replay caught incorrect proposal_source metadata; a RED→GREEN regression fixed it before the valid matrix.

Human adjudication is pending. Scores use unchanged development gold and have no held-out generalization claim. Timing is a single CPU decode with tracemalloc, excludes input loading/scoring and neural forward; measured memory is traced Python allocations, not process RSS.

Reproduce: `python3 -m journal.scripts.run_e1_multi --d0 /ABS/D0_EXTRACTED --output /ABS/NEW_OUTPUT` using the raw JSONL payloads from the D0 handoff. Source hashes and exact inputs are recorded in the result provenance. Full results/per-document metrics, predictions, added-TP inventory, report and verification logs are in the separate 2026-10-24 E1 result handoff.
