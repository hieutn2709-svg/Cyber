# SCSP Gate A Verified Dataset Adapter Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the hash-verified controlled V4 clean-window dataset to the new plain SpanPair Gate A implementation without changing the fixed document folds or silently dropping relation endpoints.

**Architecture:** Treat the 10 journal core entity types as the primary entity-evaluation inventory, while retaining the five additional annotated entity types as auxiliary trainable endpoint labels. This preserves all 564 evaluable relation instances used by the fixed manifest. The adapter exposes local content-token bounds, trainable spans, primary spans, and relations without importing any legacy training implementation.

**Tech Stack:** Python 3.10+, standard library, existing `journal.scsp` dataclasses/tests.

**Spec:** `journal/SCSP_DESIGN.md` plus the data-scope correction documented in `journal/DATA_SCOPE_DECISION.md`.

## Global Constraints

- Controlled dataset SHA-256 must equal `190d3136edba33d89ee58f533e2d12cc6cac2842323e3168f6e3e0e71af72c48`.
- Preserve `doc_seq_index`, `doc_id`, and `window_index`; split assignment is document-level only.
- Primary entity metrics use exactly the ten journal core types.
- Auxiliary entity labels are trainable only to preserve annotated endpoint coverage; they are not counted in primary core Entity F1.
- Relation inventory contains the 13 relation labels present in the controlled clean-window dataset.
- Do not use test labels to choose span width, candidate budget, distance cutoff, thresholds, or training hyperparameters.

---

### Task 1: Versioned label inventory and data-scope decision

**Files:**
- Create: `journal/configs/gate_a_label_inventory.json`
- Create: `journal/DATA_SCOPE_DECISION.md`
- Test: `tests/test_scsp_data_adapter.py`

- [ ] Write failing tests for primary/auxiliary/trainable inventory separation and duplicate-label rejection.
- [ ] Verify RED.
- [ ] Implement immutable inventory loading.
- [ ] Verify GREEN.

### Task 2: Clean-window dataset adapter

**Files:**
- Create: `journal/scsp/data.py`
- Test: `tests/test_scsp_data_adapter.py`

- [ ] Write failing synthetic-fixture tests for content bounds, primary vs auxiliary spans, relation endpoint mapping, length validation, and document identity.
- [ ] Verify RED.
- [ ] Implement the minimal adapter.
- [ ] Verify GREEN.

### Task 3: Controlled-dataset audit artifact

**Files:**
- Create: `journal/scripts/audit_gate_a_dataset.py`
- Create: `journal/data/controlled_dataset_audit.json`
- Test: `tests/test_scsp_data_adapter.py`

- [ ] Write a failing test for deterministic audit counts on a synthetic fixture.
- [ ] Verify RED.
- [ ] Implement audit aggregation with no raw text output.
- [ ] Verify GREEN.
- [ ] Run the audit on the hash-verified controlled dataset and commit only aggregate counts/hash, never source report text.

## Acceptance criteria

- The real controlled file loads without adapter errors.
- Audit reports 67 windows and 52 documents.
- Primary entity count is 1,353; auxiliary entity count is 28.
- Total clean-window relations are 564; exactly 3 have at least one auxiliary endpoint.
- Fixed test-fold relation counts remain 128/106/137/110/83.
- No raw CTI text is added to the public repository by the audit.
