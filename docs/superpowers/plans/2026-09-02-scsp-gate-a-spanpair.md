# SCSP Gate A Plain SpanPair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully re-runnable, no-schema Gate A foundation for plain SpanPair-RoBERTa CTI extraction that preserves the fixed 52-document manifest and produces endpoint/pair bottleneck diagnostics plus independently rescorable prediction artifacts.

**Architecture:** New journal code lives only under `journal/scsp/` and does not modify legacy inference/training code. Gate A is deliberately split into pure-Python data/split/candidate/diagnostic components and optional PyTorch/Transformers model components so correctness can be unit-tested without downloading a model. Schema constraints, deterministic rescue, and calibration are explicitly excluded from Gate A.

**Tech Stack:** Python 3.10+, standard library, PyTorch for model components, Hugging Face Transformers only at runtime for RoBERTa encoding, `unittest`/`pytest` compatible tests.

**Spec:** `journal/SCSP_DESIGN.md`

## Global Constraints

- Work only on branch `journal/scsp-q2-gate-a-spanpair`; do not modify `master`.
- Use fixed manifest `experiments/cv_manifest/run_partitions_seed_42.json`; no resplitting for the primary development comparison.
- Gate A must contain no STIX schema mask, probabilistic compatibility term, deterministic rescue, or test-set threshold tuning.
- Development seed is `42`; fold partitions must remain document-disjoint.
- Every emitted prediction record must retain document ID, fold, split, span/pair coordinates, labels/scores, and run/config provenance.
- Diagnostics must expose proposal span recall, post-pruning span recall, pair recall before/after distance filtering, positive/negative candidate ratio, and relation-type/distance buckets.
- All hyperparameter choices that affect test predictions must come from config or validation-only selection and be serialized.

---

### Task 1: Immutable Gate A configuration and split guard

**Files:**
- Create: `journal/scsp/__init__.py`
- Create: `journal/scsp/config.py`
- Create: `journal/scsp/splits.py`
- Create: `journal/configs/gate_a_plain_spanpair.json`
- Test: `tests/test_scsp_gate_a_config_splits.py`

**Interfaces:**
- Produces `GateAConfig.from_json(path)` returning an immutable dataclass.
- Produces `load_fold_partition(manifest_path, fold)` and `assert_disjoint_partition(partition)`.

- [ ] Write failing tests that assert config immutability, `schema_mode == "none"`, seed/fold defaults, and rejection of overlapping train/validation/test document IDs.
- [ ] Run `python -m unittest tests.test_scsp_gate_a_config_splits -v` and verify RED because modules do not exist.
- [ ] Implement minimal immutable config + manifest partition loader/guard.
- [ ] Re-run the test and verify GREEN.
- [ ] Commit.

### Task 2: Span candidate construction and pruning diagnostics

**Files:**
- Create: `journal/scsp/structures.py`
- Create: `journal/scsp/spans.py`
- Test: `tests/test_scsp_spans.py`

**Interfaces:**
- Produces immutable `GoldSpan`, `SpanCandidate`, and `SpanDiagnostics` records.
- Produces `derive_width_cap(gold_spans, coverage=0.995)`.
- Produces `enumerate_spans(token_count, max_width)`.
- Produces `prune_span_candidates(candidates, max_candidates, min_entity_score)`.
- Produces `span_recall(gold_spans, candidates)` and bucketed recall diagnostics.

- [ ] Write failing tests for 99.5%-coverage width policy, valid inclusive span boundaries, deterministic pruning order, and exact typed-span recall.
- [ ] Run targeted tests and verify RED.
- [ ] Implement minimal span logic without any schema knowledge.
- [ ] Re-run targeted tests and verify GREEN.
- [ ] Commit.

### Task 3: Ordered entity-pair generation and pair-recall diagnostics

**Files:**
- Create: `journal/scsp/pairs.py`
- Create: `journal/scsp/diagnostics.py`
- Test: `tests/test_scsp_pairs_diagnostics.py`

**Interfaces:**
- Produces immutable `GoldRelation` and `PairCandidate` records.
- Produces `generate_ordered_pairs(spans, max_token_distance=None)`.
- Produces `pair_distance(a, b)`.
- Produces `pair_recall(gold_relations, pairs)` plus relation-type and distance-bucket summaries.
- Produces candidate positive/negative ratio when gold relations are supplied.

- [ ] Write failing tests proving directionality `(i,j) != (j,i)`, distance filtering behavior, and pair-recall loss caused only by filtering.
- [ ] Run targeted tests and verify RED.
- [ ] Implement minimal pair and diagnostic logic.
- [ ] Re-run targeted tests and verify GREEN.
- [ ] Commit.

### Task 4: Plain SpanPair neural heads with no schema path

**Files:**
- Create: `journal/scsp/model.py`
- Create: `journal/scsp/losses.py`
- Test: `tests/test_scsp_model_losses.py`

**Interfaces:**
- Produces `SpanPooler`, `SpanEntityHead`, `PairRepresentation`, `RelationExistenceHead`, and `RelationTypeHead` as small PyTorch modules.
- Model accepts already-computed token states in unit tests; the RoBERTa wrapper remains a runtime integration point so tests require no network/model download.
- Produces focal/weighted binary existence loss and positive-pair relation-type cross-entropy.

- [ ] Write failing tests for tensor shapes, ordered-pair asymmetry, no-schema forward signature, finite losses, and zero-positive relation batches.
- [ ] Run targeted tests and verify RED.
- [ ] Implement minimal PyTorch heads.
- [ ] Re-run targeted tests and verify GREEN when PyTorch is present; skip cleanly with a documented reason otherwise.
- [ ] Commit.

### Task 5: Prediction serialization and independent rescoring

**Files:**
- Create: `journal/scsp/serialization.py`
- Create: `journal/evaluation/rescore_gate_a.py`
- Test: `tests/test_scsp_serialization_rescore.py`

**Interfaces:**
- Produces JSONL records with run ID, git commit, dataset hash, config hash, fold, seed, split, document ID, predicted/gold spans and relations.
- Produces strict micro entity and typed-endpoint relation P/R/F1 by rescoring saved JSONL only.

- [ ] Write failing tests that round-trip JSONL and reproduce exact strict entity/relation counts from a synthetic two-document fixture.
- [ ] Run targeted tests and verify RED.
- [ ] Implement serializer/rescorer.
- [ ] Re-run targeted tests and verify GREEN.
- [ ] Commit.

### Task 6: Gate A dry-run harness and leakage guards

**Files:**
- Create: `journal/scripts/run_gate_a.py`
- Create: `journal/README_GATE_A.md`
- Test: `tests/test_scsp_gate_a_runner.py`

**Interfaces:**
- `python journal/scripts/run_gate_a.py --config journal/configs/gate_a_plain_spanpair.json --manifest experiments/cv_manifest/run_partitions_seed_42.json --fold 1 --dry-run`
- Dry-run validates config, manifest, dataset path/hash metadata, output directories, and confirms `schema_mode=none` without training.
- Non-dry-run requires an explicit dataset path and writes resolved config/provenance before model execution.

- [ ] Write failing tests for dry-run manifest validation, rejection of schema modes other than `none`, and refusal to write test-selected thresholds into resolved config.
- [ ] Run targeted tests and verify RED.
- [ ] Implement runner/provenance preflight.
- [ ] Re-run targeted tests and verify GREEN.
- [ ] Run all new Gate A tests plus existing `test_journal_foundation.py`.
- [ ] Commit.

## Gate A completion criteria

Gate A is complete only when one real fold/seed can train end-to-end on the controlled dataset and its saved predictions can be independently rescored. This plan intentionally implements and verifies the reusable foundation first. A real GPU training run is a separate execution checkpoint because the controlled dataset currently used by the archived V10/V13 runs is not committed as a normal repository data file and must be supplied explicitly without fabricating results.
