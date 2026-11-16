# E2 boundary consistency implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans inline, with one fresh final review.

**Goal:** opt-in conflict replacement with unchanged baseline and explicit provenance.
**Architecture:** sampler intervention after the original draw; pass one policy through training and matched CLIs; include it in recovery identity/artifacts only when enabled.
**Tech Stack:** existing Python/PyTorch/pytest; no production dependencies added.
**Spec:** docs/superpowers/specs/2026-11-09-e2-boundary-consistency-design.md

## Global constraints
Keep default typed behavior exact; do not modify gold/inference/loss/seed schedule. No full/test or merge. Keep fixture evidence separate from pretrained execution.

## Review focus
Directional and document boundary isolation; depleted replacement pools; zero positives/zero budget; default-path compatibility; cross-policy recovery rejection and both wrapper paths. Tests below cover each.

## Task1: sampler
- [x] Add tests in tests/test_boundary_consistency.py for safe pair retention, budget refill, pool exhaustion, direction/document keys, deterministic/global RNG and default equivalence.
- [x] Run tests and observe unsupported-policy failure.
- [x] Add `negative_policy='typed'` to build_training_pairs in journal/scsp/candidates.py; reject unknown values; implement exact baseline draw followed by safe replacements and opt-in diagnostics.
- [x] Run sampler tests; verify against independent exhaustive oracle.

## Task2: training and provenance
- [x] Add failing integration tests for real loss/backward/AdamW, epoch forwarding, policy metadata/guards, both matched wrappers and exact resume/reject mismatched policy.
- [x] Pass relation_negative_policy through compute_training_window_loss, _train_epoch, _run_overfit, run and wrappers. A helper validates policy and returns opt-in identity metadata before outputs/forward.
- [x] Run focused and full repository tests; no regression allowed.

## Task3: review and delivery
- [x] Fresh independent read-only code review; fix important findings with regression tests.
- [x] Record actual tests and scientific execution status in journal/reports; update draft PR, no merge.

## Ledger
Ruling: work in the fresh isolated clone on the already authorized continuation branch; no second worktree required. User asked to implement next steps; use the documented minimal intervention and existing stack without another approval loop. Scientific quality results require separate completed runs; engineering readiness is the bounded deliverable here.

Task1 complete:4 sampler tests RED (unsupported negative_policy) thenGREEN;100seed repair oracle. Task2 complete:integration RED before wiring thenGREEN; final suite290tests+100subtests. Immutableparent suite281+96. Independentparent compatibility504cases passed. Nativefixture smoke persisted bothpolicy artifacts correctly.

Final review: no Critical/Important. Minor oracle wording corrected; no remaining deferred minor. Ruling: scientific effectiveness and CUDA were outside reviewer evidence — retain explicit limits; neither is claimed. Fullsuite independently executed by primary agent.
