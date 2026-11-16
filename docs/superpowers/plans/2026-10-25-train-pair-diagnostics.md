# Training-pair diagnostics implementation plan

> **For agentic workers:** Use superpowers:executing-plans; continue inline.

**Goal:** Measure eligible and sampled collisions without changing training.
**Architecture:** Observe existing sampler lists; optional dictionary forwarded through loss and epoch loop.
**Tech stack:** Python stdlib and PyTorch.
**Spec:** docs/superpowers/specs/2026-10-25-train-pair-diagnostics.md

## Global constraints
Default off. No gold/sampling/architecture/selection changes. No full/test run.
Window observations must not be presented as independent samples or historical rates.

## Review focus
Long positives, reverse direction, multiple endpoint types, empty/zero-positive
windows, and history/recovery/CLI flag propagation.

## Task 1: Sampler observation
- [x] Write tests/test_training_pair_diagnostics.py; watch missing interface fail.
- [x] Implement training_pair_diagnostics.py and optional output in candidates.py.
- [x] Verify new and existing candidate tests: 11 passed.

## Task 2: Training integration
- [x] Write integration tests; watch missing arguments/flag fail.
- [x] Forward output through training.py and train_gate_a; matched wrappers opt in.
- [x] Verify exact CPU loss/gradient/RNG/optimizer parity and epoch history.
- [x] Compare full suite at base and changed revisions in the same environment.

## Task 3: Review and delivery
- [x] Fresh independent review of whole change; fix important findings.
- [x] Commit/update authorized draft PR5; save evidence report.

## Execution ledger
Base c22e5e2356b56e0e9e9fa591cd1bf8ca0d00a34a; existing isolated worktree reused.
Pre-flight: Task2 consumes Task1 diagnostics dict; no tensors/RNG enter summary.
User continuation authorizes this reversible next step from the E2 audit.
Previous unsaved session edits were lost on restart; restored from visible work
and all verification rerun. No prior test outcome substitutes for fresh evidence.
Task1 complete: six new tests RED missing keyword -> GREEN; 11 candidate tests pass.

Task2 complete: missing API/CLI RED -> GREEN; 32 targeted tests pass.
Full-suite verification: baseline 259/259; changed 269/269, no skips/errors.
CPU PyTorch2.14.1+cpu; NumPy2.3.5 from existing runtime; synthetic tests only.

Final review: no Critical/Important issues; approved including strengthened test.
Independent checks: 16 diagnostics/integration/recovery tests; 240 comparisons to
base sampler; both-endpoint type multiplicity, reverse collisions, type truncation;
enabled save/restore and exact next-epoch parity; changed observation identity rejected.
Final: minor (deferred): committed tests parse CLI but do not exercise matched-wrapper
forwarding or diagnostics-specific recovery metadata. Reviewer inspected wiring and
independently tested recovery; add committed boundary coverage before broader rollout.
Final: Ruling: historical prevalence, pretrained and GPU/AMP parity remain unmeasured
and explicitly excluded from claims — synthetic CPU tests establish engineering
parity only — cost if generalized incorrectly would be invalid research conclusions.
Final: Ruling: production-scale overhead and real-package wrapper execution remain
unmeasured — bounded examples and default-off mode limit exposure — a separately
identified pilot must measure these before operational rollout.
Final: Ruling: reviewer did not audit old PR changes or repeat full-suite comparison
— this review covers only c22e5e2 onward; author-run full logs are retained — historical
review coverage must not be inferred from approval of this delta.

Task3 complete: code published at3d7aa16f3a0ea822fa12dfdca2bf890b86a39eb8; PR5 remains draft.
Evidence archive CTI_Q2_E2_Chan_doan_huan_luyen_20261025.zip saved successfully;
14 entries, CRC and all payload SHA256 verified. Final suite269/269 passed.
Next incomplete work is real-corpus training collision measurement, not this instrumentation.
