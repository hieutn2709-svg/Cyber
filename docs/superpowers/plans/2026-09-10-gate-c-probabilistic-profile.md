# Gate C Probabilistic Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a decoding-only Probabilistic Profile evaluator that propagates conditional non-NONE entity-type uncertainty into relation-type decoding on the frozen Gate A model, while preserving Gate A candidates, entity predictions, relation-existence scores, raw relation logits, and the frozen Gate B profile.

**Architecture:** Add a pure probabilistic compatibility/math layer, then an inference-side posterior extractor that calls the existing frozen Gate A `infer_window()` and performs a second read-only entity-head pass only over already-retained spans. Add a document decoder and deterministic artifact/diagnostic layer, then an evaluation-only CLI that searches the prespecified beta × threshold validation grid. Test remains inaccessible in `dev` and must not be opened until Gate C behavior is frozen.

**Tech Stack:** Python 3.11+, `unittest`, dataclasses, JSON/JSONL, PyTorch, existing `journal.scsp` Gate A/Gate B modules.

**Spec:** `docs/superpowers/specs/2026-09-10-gate-c-probabilistic-profile-design.md`

## Global Constraints

- Work only on branch `journal/scsp-q2-gate-c-prob-profile`; never modify or merge `master` without explicit approval.
- Parent frozen Hard Profile commit is `2e74e98231c3c5bb1b0db4d826602b61b71ba07d`.
- Frozen Gate A checkpoint lineage is `b4033edbaf2150605c286a36e4b0564d75b0ac91`.
- Keep `max_span_candidates = 128`, `max_relation_token_distance = 96`, `max_epochs = 12`, and relation-existence threshold grid `0.85–0.99`.
- Gate C is decoding-only: do not retrain or change encoder, entity head, relation-existence head, relation-type head, losses, negative sampling, candidate enumeration, span pruning, relation candidate generation, checkpoint selection, relation-existence logits, or raw relation-type logits.
- Main posterior is conditional non-NONE: `q(a) = P(a) / (1 - P(NONE))` over all 15 trainable entity types.
- `file-paths`, `sha256s`, and `tactic` are endpoint-unresolved and neutral in compatibility (`M=1` for hypothetical combinations containing them).
- `used-in` is relation-label unresolved/pass-through and has compatibility score exactly `1`.
- Compatibility tensor semantics come only from frozen Gate B canonicalization/profile files; validation/test labels never create or revise rules.
- `epsilon = 1e-8` is fixed; beta grid is exactly `[0.0, 0.25, 0.5, 1.0, 2.0]`.
- Validation selection is lexicographic: higher strict all-relation F1, then higher primary entity F1, then lower beta, then lower threshold.
- `beta=0` must reproduce Gate A relation-type decisions and final predictions exactly.
- `dev` evaluates validation only and must not request, infer, score, or write test artifacts.
- Protected files must remain unchanged: `journal/scsp/losses.py`, `journal/scsp/model.py`, `journal/scsp/runtime_model.py`, `journal/scsp/training.py`.
- TDD is mandatory for every production behavior change: RED test, verify RED, minimal GREEN, verify GREEN, then commit.

## File Structure

- Create `journal/scsp/gate_c.py`: pure posterior validation, probabilistic compatibility, logit adjustment, deterministic relation-type decision, document aggregation.
- Create `journal/scsp/gate_c_inference.py`: read-only Gate A inference wrapper plus full retained-span posterior extraction/parity checks.
- Create `journal/scsp/gate_c_artifacts.py`: deterministic span/posterior and pair/compatibility JSONL plus prediction-side diagnostics.
- Create `journal/scripts/evaluate_gate_c_probabilistic_profile.py`: provenance guards, validation beta × threshold selection, dev/full split guard, artifact writing.
- Create `tests/test_scsp_gate_c_math.py`: posterior/compatibility/logit math.
- Create `tests/test_scsp_gate_c_inference.py`: posterior extraction and frozen Gate A parity.
- Create `tests/test_scsp_gate_c_decoder.py`: document-level decoding and beta=0 identity.
- Create `tests/test_scsp_gate_c_artifacts.py`: deterministic strict-JSON artifacts and diagnostics.
- Create `tests/test_gate_c_probabilistic_profile_cli.py`: mode guards, frozen hyperparameters, selection tie-breaks, provenance/test-split protection.

---

### Task 1: Conditional non-NONE posterior and probabilistic compatibility math

**Files:**
- Create: `tests/test_scsp_gate_c_math.py`
- Create: `journal/scsp/gate_c.py`

**Interfaces:**
- Consumes: frozen `CanonicalizationTable`, `TaskRelationshipProfile`, relation inventory, entity inventory, endpoint posterior vectors.
- Produces:
  - `EntityTypePosterior`
  - `ProbabilisticPairDecision`
  - `conditional_non_none_posterior(probabilities, entity_types, *, span_key, tolerance=1e-8)`
  - `probabilistic_compatibility(source, target, relation_types, *, canonicalization, profile)`
  - `adjust_relation_type_logits(raw_logits, compatibility_scores, *, beta, epsilon=1e-8)`
  - `select_relation_type(adjusted_logits)`

- [ ] **Step 1: Write RED tests for posterior normalization and degenerate mass**

```python
import unittest

from journal.scsp.gate_c import conditional_non_none_posterior


class GateCPosteriorMathTests(unittest.TestCase):
    def test_conditional_non_none_posterior_normalizes(self) -> None:
        result = conditional_non_none_posterior(
            [0.20, 0.40, 0.30, 0.10],
            ["intrusion-set", "threat-actor", "malware"],
            span_key=("doc", 1, 2),
        )
        self.assertAlmostEqual(sum(result.conditional_probabilities), 1.0, places=12)
        self.assertAlmostEqual(result.none_probability, 0.20, places=12)
        self.assertAlmostEqual(result.entity_probability, 0.80, places=12)
        self.assertEqual(result.top1_entity_type, "intrusion-set")

    def test_conditional_non_none_posterior_rejects_zero_non_none_mass(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-NONE mass"):
            conditional_non_none_posterior(
                [1.0, 0.0, 0.0], ["malware", "tool"], span_key=("doc", 0, 0)
            )
```

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
python -m unittest tests.test_scsp_gate_c_math.GateCPosteriorMathTests -v
```

Expected: import/module failure because Gate C math does not exist yet.

- [ ] **Step 3: Implement minimal immutable posterior type and normalization**

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class EntityTypePosterior:
    span_key: tuple[object, ...]
    entity_types: tuple[str, ...]
    conditional_probabilities: tuple[float, ...]
    none_probability: float
    entity_probability: float
    top1_entity_type: str


def conditional_non_none_posterior(
    probabilities: Sequence[float],
    entity_types: Sequence[str],
    *,
    span_key: tuple[object, ...],
    tolerance: float = 1e-8,
) -> EntityTypePosterior:
    probs = tuple(float(x) for x in probabilities)
    labels = tuple(str(x) for x in entity_types)
    if len(probs) != len(labels) + 1:
        raise ValueError("probability vector must be NONE + entity types")
    if not labels:
        raise ValueError("entity_types must be non-empty")
    if any(x < -tolerance or x > 1.0 + tolerance for x in probs):
        raise ValueError("entity probabilities must lie in [0, 1]")
    if abs(sum(probs) - 1.0) > tolerance:
        raise ValueError("entity probabilities must sum to one")
    entity_mass = 1.0 - probs[0]
    if entity_mass <= tolerance:
        raise ValueError("non-NONE mass is numerically degenerate")
    conditional = tuple(max(0.0, x) / entity_mass for x in probs[1:])
    normalizer = sum(conditional)
    conditional = tuple(x / normalizer for x in conditional)
    top1 = max(range(len(labels)), key=lambda i: (conditional[i], -i))
    return EntityTypePosterior(
        span_key=span_key,
        entity_types=labels,
        conditional_probabilities=conditional,
        none_probability=float(probs[0]),
        entity_probability=float(entity_mass),
        top1_entity_type=labels[top1],
    )
```

- [ ] **Step 4: Verify GREEN for posterior tests**

Run the same command; expected `OK`.

- [ ] **Step 5: Add RED tests for compatibility semantics**

Use small frozen-style canonicalization/profile fixtures and assert:

```python
self.assertAlmostEqual(
    probabilistic_compatibility(src_one_hot, dst_one_hot, ["uses"], canonicalization=table, profile=profile)[0],
    1.0,
)
self.assertAlmostEqual(
    probabilistic_compatibility(src_incompatible, dst_incompatible, ["uses"], canonicalization=table, profile=profile)[0],
    0.0,
)
self.assertEqual(
    probabilistic_compatibility(src_any, dst_any, ["used-in"], canonicalization=table, profile=profile),
    (1.0,),
)
```

Also construct a posterior with positive mass on `tactic` and prove that the unresolved mass contributes neutrally rather than being zeroed.

- [ ] **Step 6: Implement compatibility calculation using frozen Gate B semantics**

For every relation label and every source/target entity-type pair:

```python
rule = canonicalize_relation(relation_label, canonicalization)
if rule.status == "unresolved":
    m = 1.0
elif source_type in profile.unresolved_entity_types or target_type in profile.unresolved_entity_types:
    m = 1.0
else:
    lookup_source, lookup_target = (
        (target_type, source_type) if rule.swap_endpoints else (source_type, target_type)
    )
    m = float((lookup_source, rule.label, lookup_target) in profile.allowed_triples)
score += q_source * q_target * m
```

Validate that endpoint inventories match, every score is within `[0, 1]` up to `1e-12`, then clamp only tiny floating error to the closed interval.

- [ ] **Step 7: Add RED tests for beta adjustment, epsilon, ties, and beta=0 identity**

```python
self.assertEqual(
    adjust_relation_type_logits([2.0, -1.0], [0.2, 1.0], beta=0.0),
    (2.0, -1.0),
)
adjusted = adjust_relation_type_logits([2.0], [0.25], beta=1.0)
self.assertAlmostEqual(adjusted[0], 2.0 + math.log(0.25), places=12)
self.assertEqual(select_relation_type([3.0, 3.0, 2.0]), 0)
```

Assert negative beta, non-positive epsilon, mismatched vector lengths, empty relation inventory, and compatibility outside `[0,1]` raise `ValueError`.

- [ ] **Step 8: Implement minimal logit adjustment and deterministic argmax**

```python
@dataclass(frozen=True, slots=True)
class ProbabilisticPairDecision:
    compatibility_scores: tuple[float, ...]
    adjusted_type_logits: tuple[float, ...]
    selected_index: int


def adjust_relation_type_logits(raw_logits, compatibility_scores, *, beta: float, epsilon: float = 1e-8):
    logits = tuple(float(x) for x in raw_logits)
    scores = tuple(float(x) for x in compatibility_scores)
    if not logits or len(logits) != len(scores):
        raise ValueError("relation logits and compatibility scores must align and be non-empty")
    if beta < 0.0:
        raise ValueError("beta must be >= 0")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if beta == 0.0:
        return logits
    import math
    return tuple(z + beta * math.log(max(c, epsilon)) for z, c in zip(logits, scores))


def select_relation_type(adjusted_logits):
    values = tuple(float(x) for x in adjusted_logits)
    if not values:
        raise ValueError("adjusted logits must be non-empty")
    return max(range(len(values)), key=lambda i: (values[i], -i))
```

- [ ] **Step 9: Run Task 1 tests and verify GREEN**

```bash
python -m unittest tests.test_scsp_gate_c_math -v
```

- [ ] **Step 10: Commit Task 1**

```bash
git add journal/scsp/gate_c.py tests/test_scsp_gate_c_math.py
git commit -m "feat: add Gate C probabilistic compatibility math"
```

---

### Task 2: Retained-span posterior extraction with frozen Gate A parity

**Files:**
- Create: `tests/test_scsp_gate_c_inference.py`
- Create: `journal/scsp/gate_c_inference.py`

**Interfaces:**
- Consumes: model, `WindowExample`, inventory, width cap, Gate A config/training config, device.
- Reuses without modification: `journal.scsp.training.infer_window`, `_window_tensors`, `_span_tensor`.
- Produces:
  - `GateCWindowInference(base: WindowInference, posterior_by_typed_key: Mapping[tuple, EntityTypePosterior])`
  - `infer_gate_c_window(...)`
  - `infer_gate_c_split(...)`

- [ ] **Step 1: Write RED test proving Gate C delegates authoritative candidates/pairs to Gate A**

Patch `journal.scsp.gate_c_inference.infer_window` with a fake `WindowInference`, use a fake model whose encode/span/entity calls return deterministic tensors, then assert:

```python
result = infer_gate_c_window(...)
self.assertIs(result.base, fake_window_inference)
self.assertEqual(tuple(result.posterior_by_typed_key), tuple(span.typed_key for span in fake_window_inference.predicted_spans))
```

The test must fail before the module exists.

- [ ] **Step 2: Add RED tests for top-1 and entity-score parity guards**

Construct a retained span labelled `malware` but posterior whose top-1 is `tool`; expect `ValueError("top-1 parity")`. Construct matching top-1 but mismatched `entity_score`; expect `ValueError("entity-score parity")`.

- [ ] **Step 3: Implement the wrapper and second read-only entity-head pass**

Pseudo-code to implement literally:

```python
base = infer_window(
    model, window, inventory,
    width_cap=width_cap,
    base_config=base_config,
    relation_chunk_size=training_config.relation_chunk_size,
    device=device,
)
if not base.predicted_spans:
    return GateCWindowInference(base=base, posterior_by_typed_key=MappingProxyType({}))
with torch.no_grad():
    input_ids, attention_mask = _window_tensors(window, device)
    token_states = model.encode(input_ids, attention_mask)[0]
    span_reps = model.span_pooler(token_states, _span_tensor(base.predicted_spans, device))
    logits = model.heads.entity_head(span_reps)
    probs = torch.softmax(logits.detach(), dim=-1).cpu().tolist()
```

For each already-retained span, call `conditional_non_none_posterior(...)`, then assert:

```python
posterior.top1_entity_type == span.label
abs(posterior.entity_probability - span.entity_score) <= parity_tolerance
```

Never call pruning or candidate generation in the posterior pass.

- [ ] **Step 4: Run inference tests and verify GREEN**

```bash
python -m unittest tests.test_scsp_gate_c_inference -v
```

- [ ] **Step 5: Add deterministic split wrapper test**

`infer_gate_c_split()` must preserve input window order and call `infer_gate_c_window()` exactly once per window.

- [ ] **Step 6: Commit Task 2**

```bash
git add journal/scsp/gate_c_inference.py tests/test_scsp_gate_c_inference.py
git commit -m "feat: extract frozen Gate A entity posteriors for Gate C"
```

---

### Task 3: Probabilistic document decoder and exact beta=0 Gate A identity

**Files:**
- Create: `tests/test_scsp_gate_c_decoder.py`
- Modify: `journal/scsp/gate_c.py`

**Interfaces:**
- Consumes: `Sequence[GateCWindowInference]`, relation inventory, beta, threshold, frozen canonicalization/profile, metadata.
- Produces: `build_probabilistic_prediction_records(...) -> tuple[PredictionRecord, ...]`.
- Must mirror Gate B/Gate A aggregation, coordinate conversion, deduplication, sorting, and relation-existence sigmoid exactly.

- [ ] **Step 1: Write RED test that beta=0 preserves raw argmax and Gate A-style emitted relation**

Create one scored pair with raw relation logits `[5.0, 4.0]`, compatibility `[0.01, 1.0]`, beta `0.0`, threshold below its existence score. Assert selected project label is the first raw class.

- [ ] **Step 2: Write RED test that positive beta can change only relation type**

With the same endpoints/pair/existence score, use beta `2.0` and a strongly incompatible first class. Assert:

```python
self.assertEqual(gate_c_record.predicted_spans, baseline_record.predicted_spans)
self.assertAlmostEqual(gate_c_relation.relation_score, baseline_relation.relation_score)
self.assertEqual(gate_c_relation.source, baseline_relation.source)
self.assertEqual(gate_c_relation.target, baseline_relation.target)
self.assertNotEqual(gate_c_relation.label, baseline_relation.label)
```

- [ ] **Step 3: Implement decoder by adapting Gate B aggregation without hard masking**

For each above-threshold scored pair:

```python
source_posterior = inference.posterior_by_typed_key[scored_pair.pair.source.typed_key]
target_posterior = inference.posterior_by_typed_key[scored_pair.pair.target.typed_key]
compatibility = probabilistic_compatibility(
    source_posterior,
    target_posterior,
    relation_labels,
    canonicalization=canonicalization,
    profile=profile,
)
adjusted = adjust_relation_type_logits(
    scored_pair.type_logits,
    compatibility,
    beta=beta,
    epsilon=epsilon,
)
selected_index = select_relation_type(adjusted)
```

Then construct `PredictedRelation` with the original project label/direction and the unchanged sigmoid(existence_logit) score. Deduplicate by strict key exactly as Gate B does.

- [ ] **Step 4: Add RED test for missing posterior key and inventory mismatch**

Expect explicit `ValueError`, never silent fallback to top-1 hard profile.

- [ ] **Step 5: Verify decoder GREEN**

```bash
python -m unittest tests.test_scsp_gate_c_decoder -v
```

- [ ] **Step 6: Add an exact beta=0 structural parity test against Gate A aggregator**

Using synthetic `WindowInference` fixtures, compare serialized `PredictionRecord` payloads from the existing Gate A prediction builder and Gate C `beta=0` at multiple thresholds (`0.85`, `0.96`, `0.99`). Require exact equality except allowed run metadata fields if the existing builder embeds a different run id; normalize only those metadata fields in the test, never prediction content.

- [ ] **Step 7: Commit Task 3**

```bash
git add journal/scsp/gate_c.py tests/test_scsp_gate_c_decoder.py
git commit -m "feat: add Gate C probabilistic relation decoder"
```

---

### Task 4: Deterministic posterior/pair artifacts and diagnostics

**Files:**
- Create: `tests/test_scsp_gate_c_artifacts.py`
- Create: `journal/scsp/gate_c_artifacts.py`

**Interfaces:**
- Produces:
  - `build_gate_c_span_rows(...)`
  - `write_gate_c_span_jsonl(...)`
  - `build_gate_c_scored_pair_rows(..., beta, epsilon=1e-8)`
  - `write_gate_c_scored_pair_jsonl(...)`
  - `build_gate_c_diagnostics(..., beta, threshold, epsilon=1e-8)`
- Artifact builders are prediction-side only; they must not read gold labels.

- [ ] **Step 1: Write RED test for exact span posterior artifact schema**

Expected keys:

```python
{
    "document_id", "window_index", "start", "end",
    "top1_entity_type", "entity_probability", "none_probability",
    "entity_types", "conditional_non_none_posterior"
}
```

Assert conditional posterior sums to one and row order is deterministic by window iteration then retained-span order.

- [ ] **Step 2: Write RED pair artifact test**

Required keys:

```python
{
    "document_id", "window_index", "source", "target", "token_distance",
    "existence_logit", "relation_types", "raw_relation_type_logits",
    "compatibility_scores", "beta", "epsilon",
    "adjusted_relation_type_logits", "selected_index", "selected_project_label"
}
```

Assert adjusted logits recompute exactly from raw logits, compatibility, beta, epsilon.

- [ ] **Step 3: Implement deterministic strict-JSON writers**

Use:

```python
json.dumps(
    row,
    sort_keys=True,
    ensure_ascii=False,
    allow_nan=False,
    separators=(",", ":"),
)
```

Never serialize NaN/Infinity. Compatibility/logit functions must fail before a non-finite value reaches the writer.

- [ ] **Step 4: Add RED diagnostic tests**

Diagnostics must include at least:

```text
retained_span_count
posterior_top1_parity_mismatch_count
posterior_max_normalization_deviation
entity_score_parity_max_abs_difference
compatibility_summary_by_project_relation
mean_unresolved_source_posterior_mass
mean_unresolved_target_posterior_mass
candidate_relation_count
argmax_change_count_vs_gate_a
argmax_change_rate_vs_gate_a
above_threshold_pair_count
above_threshold_argmax_change_count_vs_gate_a
transition_counts
emitted_relation_count
```

Ensure diagnostics do not access `window.gold_spans` or `window.gold_relations`; build them entirely from prediction/posterior state.

- [ ] **Step 5: Verify artifact tests GREEN and byte determinism**

```bash
python -m unittest tests.test_scsp_gate_c_artifacts -v
```

Write the same fixture twice and assert `read_bytes()` equality.

- [ ] **Step 6: Commit Task 4**

```bash
git add journal/scsp/gate_c_artifacts.py tests/test_scsp_gate_c_artifacts.py
git commit -m "feat: add Gate C posterior artifacts and diagnostics"
```

---

### Task 5: Validation beta × threshold selector

**Files:**
- Create: `tests/test_gate_c_probabilistic_profile_cli.py`
- Create: `journal/scripts/evaluate_gate_c_probabilistic_profile.py`

**Interfaces:**
- Produces:
  - `_VALID_MODES = ("dev", "full")`
  - `_FROZEN_GATE_A_COMMIT = "b4033edbaf2150605c286a36e4b0564d75b0ac91"`
  - `_EXPECTED_THRESHOLD_GRID = tuple(round(v / 100, 2) for v in range(85, 100))`
  - `_EXPECTED_BETA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)`
  - `_EPSILON = 1e-8`
  - `mode_evaluates_test()`
  - `requested_evaluation_splits()`
  - `_select_probabilistic_decoder(...)`
  - provenance guard helpers patterned on Gate B.

- [ ] **Step 1: Write RED tests for mode guard and beta grid constants**

```python
self.assertFalse(mode_evaluates_test("dev"))
self.assertTrue(mode_evaluates_test("full"))
self.assertEqual(requested_evaluation_splits("dev"), ("validation",))
self.assertEqual(_EXPECTED_BETA_GRID, (0.0, 0.25, 0.5, 1.0, 2.0))
self.assertEqual(_EPSILON, 1e-8)
```

- [ ] **Step 2: Write RED selection tie-break tests**

Use a stub scoring function or patch `build_probabilistic_prediction_records` / `_score_records` so candidate grid rows are controlled. Prove lexicographic order:

```text
relation_f1 descending
primary_entity_f1 descending
beta ascending
threshold ascending
```

If all metrics tie, expected selection is `beta=0.0`, `threshold=0.85`.

- [ ] **Step 3: Implement pure grid selector**

Return:

```python
{
    "best": {
        "beta": float,
        "threshold": float,
        "relation_f1": float,
        "primary_entity_f1": float,
    },
    "grid": [... all 75 rows in deterministic beta-major then threshold order ...],
}
```

Compare rows with:

```python
(
    item["relation_f1"],
    item["primary_entity_f1"],
    -item["beta"],
    -item["threshold"],
)
```

- [ ] **Step 4: Verify selector tests GREEN**

```bash
python -m unittest tests.test_gate_c_probabilistic_profile_cli -v
```

- [ ] **Step 5: Commit Task 5**

```bash
git add journal/scripts/evaluate_gate_c_probabilistic_profile.py tests/test_gate_c_probabilistic_profile_cli.py
git commit -m "feat: add Gate C validation decoder selection"
```

---

### Task 6: Frozen-checkpoint evaluator, provenance guards, and dev test-split firewall

**Files:**
- Modify: `journal/scripts/evaluate_gate_c_probabilistic_profile.py`
- Modify: `tests/test_gate_c_probabilistic_profile_cli.py`

**Interfaces:**
- CLI arguments should mirror Gate B: `--mode`, `--gate-a-checkpoint`, `--config`, `--training-config`, `--inventory`, `--canonicalization`, `--profile`, `--manifest`, `--dataset`, `--fold`, `--seed`, `--device`, `--output-dir`.
- `dev` must obtain only validation windows; test partition materialization must occur only inside `if mode_evaluates_test(args.mode)` after validation selection is complete.

- [ ] **Step 1: Write RED provenance guard tests**

Copy the Gate B guard expectations and require failure for:

```text
checkpoint git_commit != frozen Gate A commit
checkpoint dataset hash mismatch
checkpoint config hash mismatch
companion fold mismatch
companion seed mismatch
companion dataset hash mismatch
companion config hash mismatch
checkpoint width_cap mismatch
```

- [ ] **Step 2: Write RED frozen-config/profile guard tests**

Require exact:

```text
max_span_candidates = 128
max_relation_token_distance = 96
max_epochs = 12
threshold grid = 0.85..0.99
beta grid = [0, .25, .5, 1, 2]
epsilon = 1e-8
canonicalization covers all relation types
profile resolved ∪ unresolved endpoint types == all trainable entity types
canonicalization/profile file hashes are reported, not replaced
```

- [ ] **Step 3: Write RED `dev` split-firewall test**

Patch the partition/window accessor so requesting test windows raises an assertion. Call `run()` with `mode="dev"`; the test must complete its mocked validation path without touching test. Also assert no path beginning with `test_` is written.

- [ ] **Step 4: Implement evaluator by following Gate B orchestration but replacing inference/decoder pieces**

Validation flow:

```text
preflight -> load frozen configs/profile -> hash/provenance guards
-> derive width cap from train documents only
-> load frozen Gate A checkpoint/model
-> infer_gate_c_split(validation)
-> select beta × threshold on validation
-> build selected validation predictions
-> score validation
-> write posterior/pair/selection/metrics/diagnostic/provenance artifacts
-> run_summary.test_evaluated = false
```

Do not call test-window selection anywhere in the `dev` path.

- [ ] **Step 5: Implement explicit `full` branch but do not execute it**

Only after validation selection:

```python
if mode_evaluates_test(args.mode):
    test_windows = _windows_for_ids(windows, partition.test_document_ids)
    test_inferences = infer_gate_c_split(...)
    # use already-selected beta and threshold; never reselect on test
```

The existence of code is allowed; running it is not authorized yet.

- [ ] **Step 6: Verify CLI tests GREEN**

```bash
python -m unittest tests.test_gate_c_probabilistic_profile_cli -v
```

- [ ] **Step 7: Commit Task 6**

```bash
git add journal/scripts/evaluate_gate_c_probabilistic_profile.py tests/test_gate_c_probabilistic_profile_cli.py
git commit -m "feat: add validation-guarded Gate C evaluator"
```

---

### Task 7: Run-level artifacts, beta=0 runtime parity, and journal diagnostics

**Files:**
- Modify: `journal/scripts/evaluate_gate_c_probabilistic_profile.py`
- Modify: `journal/scsp/gate_c_artifacts.py`
- Modify: relevant Gate C tests only.

**Interfaces:**
- Required validation files:

```text
validation_predictions.jsonl
validation_entity_posteriors.jsonl
validation_scored_pairs.jsonl
validation_decoder_selection.json
validation_metrics.json
validation_profile_diagnostics.json
environment.json
hashes.json
checkpoint_metadata.json
run_summary.json
```

- [ ] **Step 1: Write RED artifact-name/content tests**

Assert `run_summary.json` contains at least:

```text
status
mode
fold
seed
selection_scope = validation-only
beta
relation_threshold
validation
test_evaluated = false
gate_a_parent_commit
gate_b_parent_commit
git_commit
dataset_sha256
config_sha256
canonicalization_sha256
task_profile_sha256
epsilon
beta_grid
```

- [ ] **Step 2: Add beta=0 parity check to evaluator before accepting validation run**

Within validation inference already in memory, build Gate C `beta=0` records at every threshold and Gate A records at every threshold using the existing Gate A aggregator. Compare prediction content exactly. If any mismatch occurs, raise `RuntimeError("Gate C beta=0 parity failure")` before writing a successful run summary.

Persist a compact parity report:

```json
{
  "checked_thresholds": [0.85, ..., 0.99],
  "prediction_parity": true,
  "mismatch_count": 0
}
```

inside `validation_profile_diagnostics.json` or a dedicated `validation_beta0_parity.json` if this keeps diagnostics clearer.

- [ ] **Step 3: Add diagnostics by beta**

For each beta in the fixed grid report:

```text
argmax_change_count_vs_gate_a
argmax_change_rate_vs_gate_a
above_selected_threshold_change_count
transition_counts
```

Do not use gold labels for these prediction-side counts.

- [ ] **Step 4: Verify all Gate C tests GREEN**

```bash
python -m unittest \
  tests.test_scsp_gate_c_math \
  tests.test_scsp_gate_c_inference \
  tests.test_scsp_gate_c_decoder \
  tests.test_scsp_gate_c_artifacts \
  tests.test_gate_c_probabilistic_profile_cli -v
```

- [ ] **Step 5: Commit Task 7**

```bash
git add journal/scripts/evaluate_gate_c_probabilistic_profile.py journal/scsp/gate_c_artifacts.py tests/test_scsp_gate_c_*.py tests/test_gate_c_probabilistic_profile_cli.py
git commit -m "feat: add Gate C parity and journal diagnostics"
```

---

### Task 8: Verification gate before any Gate C experiment

**Files:**
- No production changes unless a demonstrated defect appears; any defect fix requires a new RED test first.

**Interfaces:**
- Produces evidence only. No validation/test experiment until all checks pass.

- [ ] **Step 1: Compile Gate C files**

```bash
python -m py_compile \
  journal/scsp/gate_c.py \
  journal/scsp/gate_c_inference.py \
  journal/scsp/gate_c_artifacts.py \
  journal/scripts/evaluate_gate_c_probabilistic_profile.py
```

Expected: no output/errors.

- [ ] **Step 2: Run all targeted Gate C tests**

```bash
python -m unittest discover -s tests -p 'test_*gate_c*.py' -v
```

Expected: all tests `OK`.

- [ ] **Step 3: Run full SCSP regression suite**

```bash
python -m unittest discover -s tests -p 'test_scsp*.py' -v
```

Expected: all tests `OK`; record the exact count from fresh output rather than assuming the previous Gate B count.

- [ ] **Step 4: Verify protected Gate A files have no diff from frozen Gate B parent**

```bash
git diff --exit-code 2e74e98231c3c5bb1b0db4d826602b61b71ba07d -- \
  journal/scsp/losses.py \
  journal/scsp/model.py \
  journal/scsp/runtime_model.py \
  journal/scsp/training.py
```

Expected: empty output and exit code 0.

- [ ] **Step 5: Verify frozen Gate B profile files are byte-identical to parent**

```bash
git diff --exit-code 2e74e98231c3c5bb1b0db4d826602b61b71ba07d -- \
  journal/configs/stix/relation_canonicalization_v1.json \
  journal/configs/stix/task_relationship_profile_v1.json
```

Expected: empty output and exit code 0.

- [ ] **Step 6: Verify constants statically**

```bash
python - <<'PY'
from journal.scripts.evaluate_gate_c_probabilistic_profile import (
    _EXPECTED_BETA_GRID,
    _EXPECTED_THRESHOLD_GRID,
    _EPSILON,
)
assert _EXPECTED_BETA_GRID == (0.0, 0.25, 0.5, 1.0, 2.0)
assert _EXPECTED_THRESHOLD_GRID == tuple(round(v / 100, 2) for v in range(85, 100))
assert _EPSILON == 1e-8
print("GATE C CONSTANTS: OK")
PY
```

- [ ] **Step 7: Verify branch and recent commits**

```bash
git branch --show-current
git log --oneline -10
```

Expected branch: `journal/scsp-q2-gate-c-prob-profile`.

- [ ] **Step 8: Stop before experiment**

Do **not** run `--mode full`. After verification evidence is reviewed, the next and only authorized experiment is:

```text
Gate C dev
Fold 1
seed 42
frozen Gate A checkpoint
frozen Gate B profile v1
validation only
```

---

## Plan Self-Review

- Spec coverage: conditional posterior, neutral unresolved endpoint mass, `used-in=1`, probabilistic compatibility, epsilon/beta grid, beta=0 identity, joint validation selection, split guard, artifacts, diagnostics, TDD, and frozen-file guards are all mapped to explicit tasks.
- Placeholder scan: no `TBD`, `TODO`, deferred implementation instruction, or unspecified error-handling step remains.
- Type consistency: `EntityTypePosterior` flows from Task 1 to posterior extraction, compatibility, artifacts, and decoder; `GateCWindowInference` is produced only by Task 2 and consumed by Tasks 3–7; beta/epsilon constants are centralized in the evaluator and passed explicitly to pure decoder/artifact functions.
- Scope: one subsystem only—Gate C decoding/evaluation—so a single plan is appropriate.
