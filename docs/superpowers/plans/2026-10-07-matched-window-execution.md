# Matched-window implementation and execution

> Required skill: superpowers:executing-plans. Native execution already selected.

**Goal:** Build approved paired 70-window corpus and continue guarded baseline.
**Spec:** docs/superpowers/specs/2026-10-07-securebert-matched-window-amendment.md
**Architecture:** Separate matched builder and wrapper; reuse verified materialization and Gate A training.

## Global Constraints
Pinned packages and original 67-window corpus immutable. Identical character slices, no truncation. Preserve source IDs, coordinates, roles, relations, document order and Fold 1/11800/42. No full/test. User approved RoBERTa dev rerun after gates. No metrics without actual run.

## Review Focus
Native coordinates must never reuse reference indices. Equal counts can conceal changed membership. Manifest substitution must not bypass corpus/split/package hashes. Delegation and checkpoint loading preserve dev-only and package guards. Review whitespace boundaries and paired atomic failure.

### Task 1: Matched corpus
**Files:** journal/scsp/matched_windows.py; journal/scripts/build_matched_encoder_datasets.py; tests/test_matched_windows.py.
**Interfaces:** Frozen rows and source -> shared character contract, native rows, atomic paired bundle.
- [ ] Test largest safe split, forbidden entity/relation cuts, asymmetric capacity, IDs/coords/roles, native coordinates, failed pair writes.
- [ ] Run unittest tests.test_matched_windows. Expected: missing implementation RED.
- [ ] Implement independent slicing, exact ancestor reconstruction/hash validation, common manifest, frozen partition copies changing dataset SHA only.
- [ ] Run focused tests. Expected: GREEN. Commit; generate real datasets. Expected 52/70/1353/28/564/561, zero truncation, approved three cuts.
- [ ] Record evidence in ledger.

### Task 2: Runtime and guarded training
**Files:** journal/scsp/matched_runtime.py; journal/scripts/preflight_matched_encoders.py; journal/scripts/train_matched_encoder.py; optional Gate A provenance hooks; paired configs; tests/test_matched_runtime.py.
**Interfaces:** Task 1 bundle -> verified package/config/split -> preflight -> overfit/smoke/dev with package-bound checkpoints.
- [ ] Test changed downstream settings, wrong package/dataset/split hashes, vocabulary/position mismatch, full prohibition, checkpoint mismatch.
- [ ] Run unittest tests.test_matched_runtime. Expected: RED.
- [ ] Implement fail-closed contracts, pinned trust_remote_code=False loading, no-gradient preflight, guarded delegation; preserve Gate A defaults.
- [ ] Run focused and full unittest suite. Expected: GREEN. Commit and ledger.

### Task 3: Execute gates and review
**Files:** Runtime artifacts outside repository; tracked execution report.
**Interfaces:** Validated bundles -> actual preflight/diagnostic/run artifacts or observed blocker.
- [ ] Execute pinned model preflight, then overfit, smoke, regression and approved RoBERTa dev when gates pass. Expected: actual evidence or precise external blocker, no weakened guards.
- [ ] One independent final review; fix Important/Critical RED -> GREEN; save results and publish feature PR update without merge.
- [ ] Record hashes, counts and remaining gates. Expected: no fabricated metrics.
