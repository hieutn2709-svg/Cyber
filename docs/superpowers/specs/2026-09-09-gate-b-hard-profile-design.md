# Gate B Hard-Profile Design

**Date:** 2026-09-09  
**Branch:** `journal/scsp-q2-gate-b-hard-profile`  
**Parent Gate A commit:** `b4033edbaf2150605c286a36e4b0564d75b0ac91`

## 1. Purpose

Gate B tests whether a versioned STIX-oriented task relationship profile improves relation decoding while preserving the Gate A model, data, training objective, candidate budget, and evaluation protocol.

The first Gate B variant is **Hard Profile**. It is intentionally decoding-only so the causal comparison with Gate A remains interpretable: no retraining change, no rescue rule, no calibration change, and no test-driven profile edits.

## 2. Evidence carried forward from Gate A

Gate A is frozen before this work begins. Fold 1 / seed 42 uses:

- `max_span_candidates = 128`
- `max_relation_token_distance = 96`
- `max_epochs = 12`
- validation threshold grid `0.85–0.99`
- strict deterministic runtime
- `schema_mode = none`
- no deterministic rescue

The frozen Fold 1 run selected epoch 10 and relation threshold 0.96 from validation. Test evaluation has already been opened for Gate A and must not be used to revise Gate B profile semantics.

The relation label inventory remains the existing 13 project labels:

`attributed-to`, `authored-by`, `deliver`, `drops`, `have`, `indicates`, `located-at`, `originates-from`, `targeted-by`, `targets`, `used-by`, `used-in`, `uses`.

The entity inventory remains 10 primary plus 5 auxiliary trainable types.

## 3. Terminology

The implementation must keep two concepts separate:

1. **OASIS STIX 2.1 conformance** — whether constructed STIX objects/bundles satisfy the specification and validator requirements.
2. **Task-profile semantic compatibility** — whether a `(source_type, relation_type, target_type)` triple is allowed by the versioned project profile used during extraction.

A relationship outside this task profile is not automatically called “invalid STIX.” The paper and code comments must use wording such as **specification-defined relationship triple**, **task-profile compatible**, or **task-profile incompatible**.

## 4. Canonicalization layer

Training labels and the dataset are not rewritten. Canonicalization is applied only for compatibility checks and reporting.

The versioned canonicalization rules are:

| Project label | Canonical relationship | Endpoint action | Status |
| --- | --- | --- | --- |
| `attributed-to` | `attributed-to` | keep | direct |
| `authored-by` | `authored-by` | keep | direct |
| `deliver` | `delivers` | keep | deterministic alias |
| `drops` | `drops` | keep | direct |
| `have` | `has` | keep | deterministic alias, compatibility still endpoint-conditioned |
| `indicates` | `indicates` | keep | direct |
| `located-at` | `located-at` | keep | direct |
| `originates-from` | `originates-from` | keep | direct |
| `targeted-by` | `targets` | swap source/target | deterministic inverse |
| `targets` | `targets` | keep | direct |
| `used-by` | `uses` | swap source/target | deterministic inverse |
| `used-in` | none | none | unresolved/pass-through |
| `uses` | `uses` | keep | direct |

### `used-in`

`used-in` remains unresolved in Hard Profile. It is not silently mapped to `uses` and is not hard-blocked solely because no canonical rule is defined. It passes through the relation-type mask unchanged and is reported separately in profile diagnostics.

This is deliberate: a single training-side instance is insufficient evidence for a safe global inverse/canonical mapping, and the mapping must not be inferred from validation or test labels.

## 5. Task relationship profile

`task_relationship_profile_v1.json` stores the allowed canonical triples over the entity types supported by this project.

The initial profile is the intersection of:

- relationship triples defined by the STIX 2.1 specification; and
- entity types present in the project inventory.

The profile is versioned data, not executable logic. Every entry must contain a short rationale/source note. Project-specific or unresolved relationships are not added merely to improve validation or test scores.

Examples of allowed canonical triples include:

- `campaign --attributed-to--> intrusion-set`
- `intrusion-set --attributed-to--> threat-actor`
- `malware --authored-by--> threat-actor`
- `attack-pattern --delivers--> malware`
- `tool --delivers--> malware`
- `malware --drops--> malware`
- `malware --drops--> tool`
- `tool --has--> vulnerability`
- `indicator --indicates--> malware`
- `identity --located-at--> location`
- `intrusion-set --originates-from--> location`
- `intrusion-set --targets--> identity`
- `intrusion-set --targets--> location`
- `intrusion-set --targets--> vulnerability`
- `intrusion-set --uses--> attack-pattern`
- `intrusion-set --uses--> malware`
- `intrusion-set --uses--> tool`

The complete file must be explicit rather than generated from informal prose.

## 6. Hard Profile behavior

Hard Profile operates on the relation-type decision after Gate A has produced retained spans and ordered relation candidates.

For each ordered candidate pair:

1. take the top-1 predicted entity type for each endpoint;
2. consider each project relation label in the relation-type logits;
3. canonicalize that label and, if required, canonicalize endpoint direction for the compatibility lookup;
4. if the canonical triple is allowed by `task_relationship_profile_v1`, keep the relation logit unchanged;
5. if the canonical triple is resolved and incompatible, mask that relation class before relation-type argmax;
6. if the relation label is explicitly unresolved (`used-in` in v1), leave it unmasked;
7. if every resolved class is masked and no unresolved class remains available, return a deterministic “no compatible relation type” outcome rather than inventing a class.

The relation-existence score is not altered by Hard Profile. The entity predictions are not altered. Candidate generation is not altered.

## 7. Separation from training

Hard Profile must not modify:

- encoder weights or architecture;
- entity loss;
- relation-existence loss;
- relation-type loss;
- negative sampling;
- candidate span budget;
- relation distance cutoff;
- checkpoint selection;
- validation threshold selection procedure.

The no-schema and hard-profile variants therefore use identical learned model capacity and differ only at relation-type decoding.

## 8. Files and responsibilities

Create:

- `journal/configs/stix/stix_2_1_normative_notes.json` — compact provenance notes and terminology guardrails.
- `journal/configs/stix/relation_canonicalization_v1.json` — project-label to canonical-label/direction mapping.
- `journal/configs/stix/task_relationship_profile_v1.json` — allowed canonical triples.
- `journal/configs/stix/README.md` — explains normative-vs-task-profile distinction and versioning policy.
- `journal/scsp/schema.py` — pure loading, validation, canonicalization, compatibility, and hard-mask utilities.
- `tests/test_scsp_schema.py` — unit tests for all Gate B schema behavior.

The first implementation must avoid modifying `journal/scsp/training.py` or Gate A losses. Driver integration comes only after the standalone schema unit passes its TDD cycle.

## 9. Proposed pure interfaces

The implementation plan should preserve small, testable interfaces equivalent to:

```python
@dataclass(frozen=True, slots=True)
class CanonicalRelation:
    label: str | None
    swap_endpoints: bool
    status: str


def load_relation_canonicalization(path: str | Path) -> ...:
    ...


def load_task_relationship_profile(path: str | Path) -> ...:
    ...


def canonicalize_relation(label: str, canonicalization) -> CanonicalRelation:
    ...


def is_profile_compatible(
    source_type: str,
    relation_label: str,
    target_type: str,
    *,
    canonicalization,
    profile,
) -> bool | None:
    """True=allowed, False=resolved but blocked, None=unresolved/pass-through."""
    ...


def hard_profile_mask(
    relation_logits,
    *,
    source_type: str,
    target_type: str,
    relation_types: tuple[str, ...],
    canonicalization,
    profile,
):
    ...
```

The exact container types may be refined in the implementation plan, but the semantics above are fixed.

## 10. Diagnostics and artifacts

Every Hard Profile evaluation must record enough information to reproduce the mask decisions without reading test gold labels.

At minimum report:

- candidate relation count;
- resolved compatible class opportunities;
- resolved blocked class opportunities;
- unresolved/pass-through opportunities;
- emitted relation count;
- emitted task-profile-compatible count;
- emitted unresolved count;
- emitted blocked count, which must be zero by construction;
- compatibility rate among resolved emitted relations;
- counts by project relation label;
- counts by canonical relation label;
- counts by source/target entity type.

Validation and test artifacts must be separate. Test diagnostics may summarize prediction-side compatibility decisions but must never feed profile edits.

## 11. Leakage and freeze rules

The following are hard constraints:

- Profile semantics may use OASIS STIX 2.1 and Fold 1 training-side audit evidence.
- Validation labels may be used only for the already-prespecified model-selection/evaluation protocol, not to invent new allowed triples.
- Test labels must not be used to add, remove, reverse, or rename profile relations.
- `used-in` remains unresolved for v1 regardless of Gate B validation/test behavior.
- No rescue rules are introduced in Gate B.
- No probabilistic compatibility term is introduced until the Hard Profile variant is frozen and evaluated.

## 12. TDD acceptance criteria

Before driver integration, unit tests must prove:

1. canonicalization maps `deliver -> delivers` without swapping endpoints;
2. canonicalization maps `targeted-by -> targets` and swaps endpoints;
3. canonicalization maps `used-by -> uses` and swaps endpoints;
4. `used-in` returns unresolved/pass-through status;
5. allowed triples return `True`;
6. resolved incompatible triples return `False`;
7. unresolved triples return `None`;
8. hard masking preserves logits for compatible labels;
9. hard masking removes resolved incompatible labels from argmax consideration;
10. unresolved `used-in` remains available;
11. profile/config loaders reject duplicate or malformed entries;
12. profile logic is deterministic;
13. no Gate A schema-disabled behavior changes when schema utilities are unused.

After standalone unit tests pass, integration tests must verify that no-schema and hard-profile evaluation use the same model outputs before masking and that only relation-type decoding changes.

## 13. Gate B Hard Profile acceptance gate

Hard Profile is ready for evaluation only when:

- canonicalization and task-profile files are versioned and validated;
- all schema unit tests pass;
- no-schema behavior remains unchanged;
- profile decisions can be reproduced from saved prediction/logit artifacts;
- validation/test separation is preserved;
- no profile rule was introduced from test-label inspection.

Only after this gate is complete may the project proceed to the probabilistic profile variant.