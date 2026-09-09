# Gate B Hard Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate a deterministic, decoding-only STIX 2.1 Hard Profile variant on the frozen Gate A SpanPair model without changing training, relation existence, entity predictions, or candidate generation.

**Architecture:** Add a pure CPU-side schema/profile layer first, then a document-level hard-profile decoder that consumes existing `WindowInference` objects, then a split-specific artifact writer and an evaluation-only CLI that loads a frozen Gate A checkpoint. Validation reuses the prespecified threshold grid; test remains inaccessible unless the user explicitly runs `--mode full` after the profile is frozen.

**Tech Stack:** Python 3.11+, `unittest`, dataclasses, JSON/JSONL, PyTorch only in the evaluation driver, existing `journal.scsp` Gate A modules.

**Spec:** `docs/superpowers/specs/2026-09-09-gate-b-hard-profile-design.md`

## Global Constraints

- Work only on branch `journal/scsp-q2-gate-b-hard-profile`; never modify or merge `master` without explicit approval.
- Parent Gate A checkpoint lineage is `b4033edbaf2150605c286a36e4b0564d75b0ac91`.
- Keep `max_span_candidates = 128`, `max_relation_token_distance = 96`, `max_epochs = 12`, and validation threshold grid `0.85–0.99`.
- Hard Profile is decoding-only: no encoder, loss, negative sampling, candidate generation, checkpoint selection, or relation-existence changes.
- `used-in` is relation-label unresolved/pass-through in v1.
- `file-paths`, `sha256s`, and `tactic` are endpoint-unresolved/pass-through in v1.
- Profile semantics come from OASIS STIX 2.1 plus training-side audit evidence only; validation/test labels never create or revise rules.
- Test evaluation is allowed only in explicit full mode after validation behavior/profile files are frozen.
- TDD is mandatory for every production behavior change: RED test, verify RED, minimal GREEN, verify GREEN, then commit.

---

### Task 1: Relation canonicalization core

**Files:**
- Create: `tests/test_scsp_schema.py`
- Create: `journal/scsp/schema.py`

**Interfaces:**
- Produces: `RelationStatus`, `CanonicalRelation`, `CanonicalizationTable`, `load_relation_canonicalization()`, `canonicalize_relation()`.
- Consumes: JSON canonicalization payloads with `version`, `rules`, and rule fields `project_label`, `canonical_label`, `swap_endpoints`, `status`.

- [ ] **Step 1: Write the first failing canonicalization test**

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from journal.scsp.schema import (
        canonicalize_relation,
        load_relation_canonicalization,
    )
except ImportError:
    canonicalize_relation = None
    load_relation_canonicalization = None


class SchemaTests(unittest.TestCase):
    def _json_file(self, payload: dict) -> tempfile.TemporaryDirectory:
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "payload.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        td.path = path  # type: ignore[attr-defined]
        return td

    def test_relation_canonicalization_maps_deliver_alias(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)
        self.assertIsNotNone(canonicalize_relation)
        with self._json_file({
            "version": 1,
            "rules": [{
                "project_label": "deliver",
                "canonical_label": "delivers",
                "swap_endpoints": False,
                "status": "alias",
            }],
        }) as td:
            table = load_relation_canonicalization(td.path)  # type: ignore[attr-defined]
        result = canonicalize_relation("deliver", table)
        self.assertEqual(result.label, "delivers")
        self.assertFalse(result.swap_endpoints)
        self.assertEqual(result.status, "alias")
```

- [ ] **Step 2: Run the targeted test and verify RED**

Run:

```bash
python -m unittest tests.test_scsp_schema.SchemaTests.test_relation_canonicalization_maps_deliver_alias -v
```

Expected: `FAIL`, specifically because the schema API is not implemented yet.

- [ ] **Step 3: Implement the minimum canonicalization API**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

RelationStatus = Literal["direct", "alias", "inverse", "unresolved"]
_ALLOWED_STATUSES = {"direct", "alias", "inverse", "unresolved"}


@dataclass(frozen=True, slots=True)
class CanonicalRelation:
    label: str | None
    swap_endpoints: bool
    status: RelationStatus


@dataclass(frozen=True, slots=True)
class CanonicalizationTable:
    by_project_label: Mapping[str, CanonicalRelation]


def load_relation_canonicalization(path: str | Path) -> CanonicalizationTable:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("rules"), list):
        raise ValueError("invalid relation canonicalization payload")
    by_label: dict[str, CanonicalRelation] = {}
    for item in payload["rules"]:
        expected = {"project_label", "canonical_label", "swap_endpoints", "status"}
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("malformed canonicalization rule")
        project_label = item["project_label"]
        status = item["status"]
        if not isinstance(project_label, str) or not project_label.strip():
            raise ValueError("project_label must be non-empty")
        if project_label in by_label:
            raise ValueError(f"duplicate project relation label: {project_label}")
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"unknown canonicalization status: {status}")
        canonical_label = item["canonical_label"]
        swap = item["swap_endpoints"]
        if not isinstance(swap, bool):
            raise ValueError("swap_endpoints must be boolean")
        if status == "unresolved":
            if canonical_label is not None or swap:
                raise ValueError("unresolved rules require null label and no swap")
        elif not isinstance(canonical_label, str) or not canonical_label.strip():
            raise ValueError("resolved rules require canonical_label")
        by_label[project_label] = CanonicalRelation(canonical_label, swap, status)
    return CanonicalizationTable(MappingProxyType(by_label))


def canonicalize_relation(
    label: str,
    canonicalization: CanonicalizationTable,
) -> CanonicalRelation:
    try:
        return canonicalization.by_project_label[label]
    except KeyError as exc:
        raise ValueError(f"unknown project relation label: {label}") from exc
```

- [ ] **Step 4: Run the targeted test and verify GREEN**

Run the same command. Expected: `OK`.

- [ ] **Step 5: Add RED tests for inverse, unresolved, duplicate, and malformed rules**

Add separate tests asserting:

```python
self.assertEqual(canonicalize_relation("targeted-by", table).label, "targets")
self.assertTrue(canonicalize_relation("targeted-by", table).swap_endpoints)
self.assertEqual(canonicalize_relation("used-by", table).status, "inverse")
self.assertIsNone(canonicalize_relation("used-in", table).label)
self.assertEqual(canonicalize_relation("used-in", table).status, "unresolved")
```

and `ValueError` for duplicate labels, unknown status, unresolved-with-label, and missing fields.

- [ ] **Step 6: Run schema tests, verify RED, implement only missing validations, verify GREEN**

Run:

```bash
python -m unittest tests.test_scsp_schema -v
```

- [ ] **Step 7: Commit Task 1**

```bash
git add tests/test_scsp_schema.py journal/scsp/schema.py
git commit -m "feat: add Gate B relation canonicalization core"
```

---

### Task 2: Task-profile loader and compatibility semantics

**Files:**
- Modify: `tests/test_scsp_schema.py`
- Modify: `journal/scsp/schema.py`

**Interfaces:**
- Produces: `TaskRelationshipProfile`, `load_task_relationship_profile()`, `is_profile_compatible()`.
- Consumes: profile JSON with `version`, `resolved_entity_types`, `unresolved_entity_types`, and explicit `allowed_triples` objects.

- [ ] **Step 1: Write RED tests for allowed, blocked, relation-unresolved, and endpoint-unresolved cases**

Use a temporary profile containing:

```json
{
  "version": 1,
  "resolved_entity_types": ["intrusion-set", "malware", "tool"],
  "unresolved_entity_types": ["tactic"],
  "allowed_triples": [
    {
      "source_type": "intrusion-set",
      "relation_type": "uses",
      "target_type": "malware",
      "source_note": "STIX 2.1 Appendix B"
    }
  ]
}
```

Assert:

```python
self.assertIs(is_profile_compatible("intrusion-set", "uses", "malware", ...), True)
self.assertIs(is_profile_compatible("intrusion-set", "uses", "tool", ...), False)
self.assertIs(is_profile_compatible("intrusion-set", "used-in", "malware", ...), None)
self.assertIs(is_profile_compatible("tactic", "uses", "malware", ...), None)
```

- [ ] **Step 2: Run the four targeted tests and verify RED**

Expected: missing profile API.

- [ ] **Step 3: Implement immutable profile loading and compatibility**

```python
@dataclass(frozen=True, slots=True)
class TaskRelationshipProfile:
    allowed_triples: frozenset[tuple[str, str, str]]
    resolved_entity_types: frozenset[str]
    unresolved_entity_types: frozenset[str]


def load_task_relationship_profile(path: str | Path) -> TaskRelationshipProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != 1:
        raise ValueError("unsupported task profile version")
    resolved = tuple(payload.get("resolved_entity_types", ()))
    unresolved = tuple(payload.get("unresolved_entity_types", ()))
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("resolved_entity_types must be unique and non-empty")
    if len(set(unresolved)) != len(unresolved):
        raise ValueError("unresolved_entity_types must be unique")
    overlap = set(resolved) & set(unresolved)
    if overlap:
        raise ValueError(f"entity type status overlap: {sorted(overlap)}")
    triples: list[tuple[str, str, str]] = []
    for item in payload.get("allowed_triples", ()):
        expected = {"source_type", "relation_type", "target_type", "source_note"}
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("malformed task-profile triple")
        triple = (item["source_type"], item["relation_type"], item["target_type"])
        if any(not isinstance(value, str) or not value.strip() for value in triple):
            raise ValueError("task-profile triple fields must be non-empty strings")
        if not isinstance(item["source_note"], str) or not item["source_note"].strip():
            raise ValueError("source_note must be non-empty")
        triples.append(triple)
    if len(set(triples)) != len(triples):
        raise ValueError("duplicate task-profile triple")
    known = set(resolved) | set(unresolved)
    if any(source not in known or target not in known for source, _, target in triples):
        raise ValueError("task-profile triple uses undeclared entity type")
    return TaskRelationshipProfile(frozenset(triples), frozenset(resolved), frozenset(unresolved))


def is_profile_compatible(...):
    relation = canonicalize_relation(relation_label, canonicalization)
    if source_type in profile.unresolved_entity_types or target_type in profile.unresolved_entity_types:
        return None
    if source_type not in profile.resolved_entity_types or target_type not in profile.resolved_entity_types:
        raise ValueError("unknown entity type for task profile")
    if relation.status == "unresolved":
        return None
    source, target = (target_type, source_type) if relation.swap_endpoints else (source_type, target_type)
    return (source, relation.label, target) in profile.allowed_triples
```

- [ ] **Step 4: Verify GREEN and add loader validation tests**

Assert duplicate triples, overlapping resolved/unresolved sets, undeclared endpoint types, and unknown runtime endpoint types raise `ValueError`.

- [ ] **Step 5: Commit Task 2**

```bash
git add tests/test_scsp_schema.py journal/scsp/schema.py
git commit -m "feat: add Gate B task-profile compatibility"
```

---

### Task 3: Deterministic hard-profile masking

**Files:**
- Modify: `tests/test_scsp_schema.py`
- Modify: `journal/scsp/schema.py`

**Interfaces:**
- Produces: `HardProfileMaskResult`, `hard_profile_mask()`.

- [ ] **Step 1: Write RED tests for compatible preservation and incompatible masking**

```python
result = hard_profile_mask(
    [9.0, 8.0],
    source_type="intrusion-set",
    target_type="malware",
    relation_types=["targets", "uses"],
    canonicalization=table,
    profile=profile,
)
self.assertEqual(result.compatibility, (False, True))
self.assertEqual(result.masked_logits[0], float("-inf"))
self.assertEqual(result.masked_logits[1], 8.0)
self.assertEqual(result.selected_index, 1)
```

- [ ] **Step 2: Write RED tests for `used-in`, unresolved endpoints, exact ties, and no survivor**

Assert:

```python
# relation-unresolved survives
self.assertIsNone(result.compatibility[used_in_index])

# endpoint-unresolved preserves all logits
self.assertEqual(result.masked_logits, tuple(original_logits))
self.assertTrue(all(value is None for value in result.compatibility))

# exact tie chooses lowest index
self.assertEqual(result.selected_index, 0)

# all resolved classes blocked and no unresolved class => None
self.assertIsNone(result.selected_index)
```

- [ ] **Step 3: Run schema tests and verify RED**

```bash
python -m unittest tests.test_scsp_schema -v
```

- [ ] **Step 4: Implement minimal mask**

```python
@dataclass(frozen=True, slots=True)
class HardProfileMaskResult:
    masked_logits: tuple[float, ...]
    compatibility: tuple[bool | None, ...]
    selected_index: int | None


def hard_profile_mask(...):
    logits = tuple(float(value) for value in relation_logits)
    labels = tuple(relation_types)
    if len(logits) != len(labels) or not labels:
        raise ValueError("relation logits and relation types must align and be non-empty")
    states = tuple(
        is_profile_compatible(
            source_type,
            label,
            target_type,
            canonicalization=canonicalization,
            profile=profile,
        )
        for label in labels
    )
    masked = tuple(float("-inf") if state is False else logit for logit, state in zip(logits, states))
    survivors = [index for index, value in enumerate(masked) if value != float("-inf")]
    selected = max(survivors, key=lambda index: (masked[index], -index)) if survivors else None
    return HardProfileMaskResult(masked, states, selected)
```

- [ ] **Step 5: Verify GREEN and commit Task 3**

```bash
git add tests/test_scsp_schema.py journal/scsp/schema.py
git commit -m "feat: add deterministic Gate B hard-profile mask"
```

---

### Task 4: Versioned STIX 2.1 profile data

**Files:**
- Create: `journal/configs/stix/stix_2_1_normative_notes.json`
- Create: `journal/configs/stix/relation_canonicalization_v1.json`
- Create: `journal/configs/stix/task_relationship_profile_v1.json`
- Create: `journal/configs/stix/README.md`
- Modify: `tests/test_scsp_schema.py`

**Interfaces:**
- Produces repository-owned v1 profile data consumed by Task 2/3 loaders.

- [ ] **Step 1: Write RED repository-data test before creating files**

The test loads the actual three JSON files and asserts:

```python
self.assertEqual(set(table.by_project_label), set(inventory.relation_types))
self.assertEqual(
    profile.resolved_entity_types | profile.unresolved_entity_types,
    set(inventory.trainable_entity_types),
)
self.assertIn(("attack-pattern", "delivers", "malware"), profile.allowed_triples)
self.assertIn(("malware", "authored-by", "intrusion-set"), profile.allowed_triples)
self.assertIn(("tool", "has", "vulnerability"), profile.allowed_triples)
self.assertIn(("threat-actor", "attributed-to", "identity"), profile.allowed_triples)
```

- [ ] **Step 2: Run the repository-data test and verify RED because files do not exist**

- [ ] **Step 3: Create `relation_canonicalization_v1.json` with all 13 project labels**

Use exactly:

```json
{
  "version": 1,
  "rules": [
    {"project_label":"attributed-to","canonical_label":"attributed-to","swap_endpoints":false,"status":"direct"},
    {"project_label":"authored-by","canonical_label":"authored-by","swap_endpoints":false,"status":"direct"},
    {"project_label":"deliver","canonical_label":"delivers","swap_endpoints":false,"status":"alias"},
    {"project_label":"drops","canonical_label":"drops","swap_endpoints":false,"status":"direct"},
    {"project_label":"have","canonical_label":"has","swap_endpoints":false,"status":"alias"},
    {"project_label":"indicates","canonical_label":"indicates","swap_endpoints":false,"status":"direct"},
    {"project_label":"located-at","canonical_label":"located-at","swap_endpoints":false,"status":"direct"},
    {"project_label":"originates-from","canonical_label":"originates-from","swap_endpoints":false,"status":"direct"},
    {"project_label":"targeted-by","canonical_label":"targets","swap_endpoints":true,"status":"inverse"},
    {"project_label":"targets","canonical_label":"targets","swap_endpoints":false,"status":"direct"},
    {"project_label":"used-by","canonical_label":"uses","swap_endpoints":true,"status":"inverse"},
    {"project_label":"used-in","canonical_label":null,"swap_endpoints":false,"status":"unresolved"},
    {"project_label":"uses","canonical_label":"uses","swap_endpoints":false,"status":"direct"}
  ]
}
```

- [ ] **Step 4: Create normative notes and explicit task profile**

Normative notes must cite the OASIS STIX 2.1 OASIS Standard HTML and Appendix B Relationship Summary. The profile must declare:

```json
"resolved_entity_types": [
  "attack-pattern", "campaign", "domain-name", "identity", "indicator",
  "intrusion-set", "location", "malware", "threat-actor", "tool", "url",
  "vulnerability"
],
"unresolved_entity_types": ["file-paths", "sha256s", "tactic"]
```

and the following 55 allowed triples, each with `"source_note": "OASIS STIX 2.1 Appendix B Relationship Summary"`:

```text
attack-pattern delivers malware
attack-pattern targets identity
attack-pattern targets location
attack-pattern targets vulnerability
attack-pattern uses malware
attack-pattern uses tool
campaign attributed-to intrusion-set
campaign attributed-to threat-actor
campaign originates-from location
campaign targets identity
campaign targets location
campaign targets vulnerability
campaign uses attack-pattern
campaign uses malware
campaign uses tool
identity located-at location
indicator indicates attack-pattern
indicator indicates campaign
indicator indicates intrusion-set
indicator indicates malware
indicator indicates threat-actor
indicator indicates tool
intrusion-set attributed-to threat-actor
intrusion-set originates-from location
intrusion-set targets identity
intrusion-set targets location
intrusion-set targets vulnerability
intrusion-set uses attack-pattern
intrusion-set uses malware
intrusion-set uses tool
malware authored-by threat-actor
malware authored-by intrusion-set
malware drops malware
malware drops tool
malware originates-from location
malware targets identity
malware targets location
malware targets vulnerability
malware uses attack-pattern
malware uses malware
malware uses tool
threat-actor attributed-to identity
threat-actor located-at location
threat-actor targets identity
threat-actor targets location
threat-actor targets vulnerability
threat-actor uses attack-pattern
threat-actor uses malware
threat-actor uses tool
tool delivers malware
tool drops malware
tool has vulnerability
tool targets identity
tool targets location
tool targets vulnerability
```

- [ ] **Step 5: Verify repository-data tests GREEN**

```bash
python -m unittest tests.test_scsp_schema -v
```

- [ ] **Step 6: Commit Task 4**

```bash
git add journal/configs/stix tests/test_scsp_schema.py
git commit -m "data: add versioned STIX 2.1 hard profile"
```

---

### Task 5: Hard-profile document decoder without changing Gate A decoder

**Files:**
- Create: `journal/scsp/gate_b.py`
- Create: `tests/test_scsp_gate_b_decoder.py`
- Read only: `journal/scsp/artifacts.py`

**Interfaces:**
- Consumes: existing `WindowInference`, relation inventory, fixed existence threshold, `CanonicalizationTable`, `TaskRelationshipProfile`.
- Produces: `build_hard_profile_prediction_records(...) -> tuple[PredictionRecord, ...]` and deterministic prediction-side diagnostics.

- [ ] **Step 1: Write RED synthetic integration test**

Construct one `WindowInference` where relation logits rank incompatible `targets` above compatible `uses`. Compare with existing `build_prediction_records()` and assert:

```python
self.assertEqual(baseline[0].predicted_spans, hard[0].predicted_spans)
self.assertEqual(baseline[0].predicted_relations[0].relation_score,
                 hard[0].predicted_relations[0].relation_score)
self.assertEqual(baseline[0].predicted_relations[0].label, "targets")
self.assertEqual(hard[0].predicted_relations[0].label, "uses")
```

Also assert the original `ScoredRelationPair.type_logits` tuple is unchanged before/after hard decoding.

- [ ] **Step 2: Verify RED**

```bash
python -m unittest tests.test_scsp_gate_b_decoder -v
```

- [ ] **Step 3: Implement `build_hard_profile_prediction_records()`**

Mirror the proven document aggregation rules in `journal/scsp/artifacts.py`, but replace only the relation-type argmax with `hard_profile_mask()`. Keep existence sigmoid/thresholding, local-to-global conversion, deduplication, relation score, gold aggregation, predicted-span aggregation, and sorting identical to Gate A.

If `selected_index is None`, emit no relation for that scored pair and increment `no_compatible_type_count`. If the selected class has compatibility `None`, emit it unchanged and count it as unresolved/pass-through.

- [ ] **Step 4: Add RED/GREEN tests for inverse labels and unresolved endpoints**

Cases:
- `used-by` compatibility swaps endpoints only for lookup, while the emitted project label remains `used-by` and original source/target direction is preserved in the prediction artifact.
- `tactic` endpoint leaves baseline relation-type argmax unchanged.
- resolved pair with all resolved classes blocked and no unresolved winner emits no relation.

- [ ] **Step 5: Verify decoder tests and SCSP regression suite**

```bash
python -m unittest tests.test_scsp_gate_b_decoder -v
python -m unittest discover -s tests -p "test_scsp*.py" -v
```

- [ ] **Step 6: Commit Task 5**

```bash
git add journal/scsp/gate_b.py tests/test_scsp_gate_b_decoder.py
git commit -m "feat: add Gate B hard-profile document decoder"
```

---

### Task 6: Split-specific scored-pair artifacts and diagnostics

**Files:**
- Create: `journal/scsp/gate_b_artifacts.py`
- Create: `tests/test_scsp_gate_b_artifacts.py`

**Interfaces:**
- Produces: deterministic JSONL writer for prediction-side scored pairs and JSON diagnostics sufficient to reproduce every hard-mask decision without gold labels.

- [ ] **Step 1: Write RED test for artifact content and ordering**

Expected JSONL row keys:

```text
document_id
window_index
source.start/source.end/source.label
 target.start/target.end/target.label
token_distance
existence_logit
relation_type_logits
relation_types
compatibility
masked_relation_type_logits
selected_index
selected_project_label
```

Assert two identical writes are byte-identical.

- [ ] **Step 2: Verify RED and implement writer**

Writer requirements:
- one row per scored pair;
- sorted input traversal inherited from `WindowInference` order;
- `json.dumps(..., sort_keys=True, ensure_ascii=False)` plus newline;
- no gold span/relation fields in the scored-pair artifact.

- [ ] **Step 3: Add diagnostics aggregator**

Return at least:

```python
{
    "candidate_relation_count": int,
    "endpoint_unresolved_pair_count": int,
    "resolved_compatible_class_opportunities": int,
    "resolved_blocked_class_opportunities": int,
    "unresolved_relation_label_opportunities": int,
    "emitted_relation_count": int,
    "emitted_task_profile_compatible_count": int,
    "emitted_unresolved_count": int,
    "emitted_blocked_count": 0,
    "compatibility_rate_resolved_emitted": float,
    "counts_by_project_relation_label": {...},
    "counts_by_canonical_relation_label": {...},
    "counts_by_source_target_entity_type": {...}
}
```

- [ ] **Step 4: Verify artifact tests GREEN and commit Task 6**

```bash
python -m unittest tests.test_scsp_gate_b_artifacts -v
git add journal/scsp/gate_b_artifacts.py tests/test_scsp_gate_b_artifacts.py
git commit -m "feat: add reproducible Gate B profile artifacts"
```

---

### Task 7: Evaluation-only Gate B CLI with test guard

**Files:**
- Create: `journal/scripts/evaluate_gate_b_hard_profile.py`
- Create: `tests/test_gate_b_hard_profile_cli.py`

**Interfaces:**
- CLI modes: `dev` and `full`.
- `dev`: loads frozen Gate A checkpoint, reruns validation inference only, applies hard profile, selects existence threshold on validation using the unchanged grid, writes validation artifacts, `test_evaluated=false`.
- `full`: repeats frozen validation selection, then performs one test inference/evaluation pass, writes separate test artifacts, `test_evaluated=true`.

- [ ] **Step 1: Write RED CLI guard tests before driver code**

```python
self.assertFalse(mode_evaluates_test("dev"))
self.assertTrue(mode_evaluates_test("full"))
with self.assertRaises(ValueError):
    mode_evaluates_test("smoke")
```

Add a source-level or helper-level test proving the test split is requested only from the `full` branch.

- [ ] **Step 2: Verify RED**

```bash
python -m unittest tests.test_gate_b_hard_profile_cli -v
```

- [ ] **Step 3: Implement minimal evaluation driver**

Required arguments:

```text
--mode {dev,full}
--gate-a-checkpoint PATH
--config journal/configs/gate_a_plain_spanpair.json
--training-config journal/configs/gate_a_training_threshold_refine.json
--inventory journal/configs/gate_a_label_inventory.json
--canonicalization journal/configs/stix/relation_canonicalization_v1.json
--profile journal/configs/stix/task_relationship_profile_v1.json
--manifest experiments/cv_manifest/run_partitions_seed_42.json
--dataset PATH
--fold 1
--seed 42
--device {auto,cuda,cpu}
--output-dir PATH
```

The driver must:

```text
1. set strict deterministic runtime before model/device creation;
2. load Gate A configs/inventory/profile data;
3. recompute dataset and combined Gate A config SHA256;
4. load checkpoint and require checkpoint dataset/config hashes to match;
5. derive width cap from the training partition exactly as Gate A did;
6. instantiate the identical Gate A model and load checkpoint weights;
7. infer validation without gold injection;
8. for every threshold in the unchanged 0.85–0.99 grid, build Hard Profile validation records and score them;
9. select best threshold by relation F1, then primary entity F1, then lower threshold exactly as Gate A selection does;
10. write validation predictions, scored-pair artifact, metrics, profile diagnostics, threshold grid, environment, hashes, checkpoint metadata, and run summary;
11. return without touching test in dev mode;
12. in full mode only, infer test once with the frozen validation-selected threshold and write separate test predictions/artifacts/metrics/diagnostics.
```

- [ ] **Step 4: Add RED/GREEN provenance tests**

Use a tiny temporary fake checkpoint metadata dict/helper test to require mismatch errors for dataset hash, config hash, fold, and seed before any test evaluation.

- [ ] **Step 5: Verify CLI tests and full SCSP regression suite**

```bash
python -m unittest tests.test_gate_b_hard_profile_cli -v
python -m unittest discover -s tests -p "test_scsp*.py" -v
python -m unittest discover -s tests -p "test_*gate_b*.py" -v
```

- [ ] **Step 6: Commit Task 7**

```bash
git add journal/scripts/evaluate_gate_b_hard_profile.py tests/test_gate_b_hard_profile_cli.py
git commit -m "feat: add validation-guarded Gate B evaluator"
```

---

### Task 8: Verification gate before any Gate B experiment

**Files:**
- No production changes unless a failing test reveals a defect.

**Interfaces:**
- Produces the evidence needed to authorize validation-only Gate B execution.

- [ ] **Step 1: Run targeted Gate B tests**

```bash
python -m unittest tests.test_scsp_schema -v
python -m unittest tests.test_scsp_gate_b_decoder -v
python -m unittest tests.test_scsp_gate_b_artifacts -v
python -m unittest tests.test_gate_b_hard_profile_cli -v
```

Expected: all `OK`.

- [ ] **Step 2: Run full SCSP regression suite**

```bash
python -m unittest discover -s tests -p "test_scsp*.py" -v
```

Expected: all existing Gate A/SCSP tests remain green.

- [ ] **Step 3: Confirm branch diff does not touch Gate A training losses/model architecture**

```bash
git diff b4033edbaf2150605c286a36e4b0564d75b0ac91..HEAD -- \
  journal/scsp/losses.py \
  journal/scsp/model.py \
  journal/scsp/runtime_model.py \
  journal/scsp/training.py
```

Expected: empty diff for all four files.

- [ ] **Step 4: Confirm profile-data completeness**

Run repository profile test and print counts. Expected:

```text
relation canonicalization rules = 13
resolved entity types = 12
unresolved entity types = 3
allowed canonical triples = 55
```

- [ ] **Step 5: Stop before experiment execution**

Do **not** run Gate B `full` yet. First run only `dev` on Fold 1 / seed 42 using the frozen Gate A checkpoint. Inspect validation metrics, compatibility diagnostics, emitted-label counts, and artifact hashes. Freeze Hard Profile v1 before any `full` command.
