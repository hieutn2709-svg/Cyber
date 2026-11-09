# E2 checkpoint-snapshot execution plan

> **For agentic workers:** Use inline execution with an independent final review.

**Goal:** Obtain bounded real-input collision measurements and close integration coverage.
**Spec:** docs/superpowers/specs/2026-10-26-e2-checkpoint-snapshot.md
**Architecture:** Unchanged training loss at fixed selected weights, observation only.
**Tech stack:** Existing PyTorch/Transformers CPU runtime and standard Python analysis.

## Tasks
- [x] Restore/authenticate two selected checkpoints and source package; preserve all originals.
- [x] Add wrapper forwarding and enabled recovery/next-epoch regression tests (3 pass).
- [x] Run both 49-window train-mode snapshots with exact on/off checks and tensor integrity.
- [x] Independently reconstruct eligible/sample counts from saved sampler inputs.
- [x] Complete review, update draft PR5, save report/evidence archive.

## Ledger
Base223215e3a0910cabfd38e2b6f1d0f0a7be8abc6c; existing isolated worktree retained.
No production code changes. Standalone snapshot script belongs to analytical evidence.
Added regression tests verify existing behavior; no artificial production defect
was introduced to force a RED phase. Initial seed-wrapper fixture missed the source
summary file required by evidence hashing; corrected fixture, all3 tests pass.
Ruling: fixed-checkpoint measurement is a staged alternative to a new optimizer run;
report this scope explicitly. Cost if misinterpreted: invalid historical-training
or improvement claims. Real training-trajectory measurement remains outstanding.

Both snapshots complete:98/98 exact on/off comparisons, weights unchanged.
Counts independently reconstructed in all98windows: eligible18/71, sampled1/0.
Full suite272/272 passed, also independently reproduced. Review no findings;
prior minor closed. No production code changes or new optimizer training run.
Final: Ruling: independent reviewer did not rerun encoder forwards — recorded exact
assertions plus completed artifacts and independent re-enumeration support the
bounded result, not independent forward replication. Cost if overstated: false
replication claim. Historical trajectory/pretrained backward/GPU/quality and
performance claims remain excluded, as detailed in the report.

Delivery complete: report/tests published ata433477ac2f9629780648e41dc02843358676b83.
CTI_Q2_E2_Ket_qua_do_checkpoint_20261026.zip saved successfully (19 entries;
CRC and all SHA256 checked). Artifact SHA25633c43dba7e09f201c7c95a18afcb233caf7aafb46e3546dcdd781883263cddb9.
Do not rerun this completed snapshot to resume: next work is trajectory measurement.
