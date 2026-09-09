# Gate C Probabilistic Profile Design

**Date:** 2026-09-10  
**Branch:** `journal/scsp-q2-gate-c-prob-profile`  
**Parent frozen Hard Profile commit:** `2e74e98231c3c5bb1b0db4d826602b61b71ba07d`  
**Frozen Gate A checkpoint lineage:** `b4033edbaf2150605c286a36e4b0564d75b0ac91`

## 1. Purpose

Gate C evaluates a **Probabilistic Profile** decoder for low-resource STIX-oriented relation extraction while keeping the learned Gate A model and the frozen Gate B task profile unchanged.

The experiment isolates one question:

> Does propagating uncertainty over endpoint entity types into relation-type decoding improve strict end-to-end relation extraction compared with no profile and hard top-1 profile constraints?

The intended journal ablation chain is:

```text
Gate A: No profile
    -> Gate B: Hard top-1 profile
    -> Gate C: Probabilistic profile
```

All three variants use the same frozen Gate A checkpoint, entity candidate set, relation candidates, relation-existence logits, relation-type logits, and validation/test protocol. Gate C changes only relation-type decoding.

## 2. Frozen evidence and non-negotiable constraints

Gate C inherits the frozen Gate A/Gate B settings:

- `max_span_candidates = 128`
- `max_relation_token_distance = 96`
- `max_epochs = 12` in the source Gate A training run
- validation relation-existence threshold grid `0.85, 0.86, ..., 0.99`
- Fold 1 / seed 42 initial development track
- strict deterministic runtime
- no gold-span injection at inference
- no deterministic rescue
- no test-label-derived schema rules

Hard Profile v1 is frozen at parent commit `2e74e98231c3c5bb1b0db4d826602b61b71ba07d` with:

- relation canonicalization SHA-256: `a43616134a2b7b4c5a2cb4b6a7a8ab2da88551c651ad2cf9c9822506ebd8608f`
- task profile SHA-256: `44b17b92182130aed5372a20c067799dde18d060e653cfa71724d2e2bdac2281`
- validation-selected relation threshold `0.96`
- validation relation F1 `0.3247863247863248`
- Gate B test split still unopened

Gate C MUST NOT modify:

- encoder architecture or weights;
- entity head weights;
- relation-existence head weights;
- relation-type head weights;
- training losses;
- negative sampling;
- candidate span enumeration;
- span pruning;
- relation candidate generation;
- relation-existence logits;
- raw relation-type logits;
- Gate A checkpoint selection;
- the frozen Gate B canonicalization/profile semantics.

The following Gate A files are protected and must remain unchanged:

```text
journal/scsp/losses.py
journal/scsp/model.py
journal/scsp/runtime_model.py
journal/scsp/training.py
```

## 3. Why conditional non-NONE entity posteriors are the main method

Gate A already separates two concepts:

1. **span/entity existence confidence**, used for candidate retention, with pruning score
   \[
   P(\text{entity}) = 1 - P(\text{NONE});
   \]
2. **entity type choice**, currently represented downstream only by the best non-NONE class.

Probabilistic compatibility should model uncertainty among entity **types**, not re-penalize whether a retained span is an entity at all. Therefore Gate C conditions the entity posterior on the span being an entity.

For a retained span `i` and trainable entity type `a`, define:

\[
q_i(a)
=
P(a\mid \text{entity})
=
\frac{P_i(a)}{1-P_i(\text{NONE})}.
\]

The denominator is the same non-NONE mass already used by Gate A for span pruning. The `q_i` vector therefore sums to one over the 15 trainable entity types.

The raw posterior `P_i(a)` is not the main Gate C method because it would mix span-existence uncertainty with type-compatibility uncertainty a second time. It may be considered later only as a separately declared sensitivity/ablation experiment, never as a post-hoc replacement chosen from validation scores.

## 4. Recovering full entity posteriors without changing Gate A

Current `SpanCandidate` stores only:

- top-1 predicted non-NONE entity label;
- `entity_score = 1 - P(NONE)`.

It does not store the full class posterior. Gate C therefore adds a new inference-side posterior extraction layer without modifying `journal/scsp/training.py`.

For each window:

1. call the existing frozen Gate A `infer_window()` to obtain the authoritative `WindowInference`, including retained spans, scored relation pairs, existence logits, and raw relation-type logits;
2. re-encode the same window with the same frozen model in `eval()` / `no_grad()` mode;
3. pool representations only for the already-retained `WindowInference.predicted_spans`;
4. run the unchanged entity head on those retained span representations;
5. apply softmax over `NONE + 15 entity types`;
6. derive the conditional non-NONE posterior `q_i(a)`;
7. assert parity with the authoritative Gate A span record:
   - best non-NONE class equals `SpanCandidate.label`;
   - `1 - P(NONE)` equals `SpanCandidate.entity_score` within a strict numeric tolerance.

This second pass is intentionally redundant computationally but methodologically safe: the candidate set and relation scores come directly from the existing Gate A inference implementation, while Gate C only reads additional posterior information for the already-selected spans.

No posterior extraction step may add, remove, relabel, or reprune a span.

## 5. Compatibility tensor over project relation labels

Let the 15 trainable entity types be indexed by `a` and `b`, and the 13 project relation labels by `r`.

Gate C constructs a deterministic compatibility tensor:

\[
M_{arb} \in \{0,1\}.
\]

The tensor is derived solely from the frozen Gate B canonicalization and task relationship profile. It is not learned and is not revised using validation or test labels.

### 5.1 Resolved relation label and resolved endpoint types

For a resolved project relation `r` and resolved endpoint types `a,b`:

- canonicalize `r` using `relation_canonicalization_v1.json`;
- if the rule is inverse, swap `a,b` only for compatibility lookup;
- set `M_{arb}=1` if the canonical triple exists in the frozen task profile;
- otherwise set `M_{arb}=0`.

The emitted project relation label and original source/target direction are never rewritten by this lookup.

### 5.2 Unresolved endpoint types are neutral, not incompatible

The frozen profile marks these endpoint types unresolved/pass-through:

```text
file-paths
sha256s
tactic
```

For any relation `r`, if either hypothetical endpoint type `a` or `b` is one of those unresolved types, define:

\[
M_{arb}=1.
\]

This is a **neutral compatibility contribution**, not a claim that the corresponding STIX triple is specification-defined. It prevents epistemic uncertainty (`unknown`) from being converted into hard incompatibility (`false`).

Diagnostics must report how much posterior mass flows through unresolved endpoint types.

### 5.3 Unresolved relation label `used-in`

`used-in` remains unresolved/pass-through exactly as in Hard Profile v1. For every endpoint-type combination:

\[
M_{a,\text{used-in},b}=1.
\]

Therefore its compatibility score is always one and Gate C does not reward or penalize its raw relation logit.

This rule is frozen before validation evaluation and must not be changed to improve development F1.

## 6. Probabilistic compatibility score

For ordered relation candidate `(i,j)` and project relation label `r`, define:

\[
C_{ijr}
=
\sum_{a,b}
q_i(a) q_j(b) M_{arb}.
\]

Properties:

- `0 <= C_ijr <= 1`;
- unresolved endpoint posterior mass contributes neutrally through `M=1`;
- `C_ij,used-in = 1` by construction;
- exact one-hot resolved endpoint posteriors reduce to the corresponding hard compatibility lookup;
- the score uses no gold labels.

## 7. Schema-adjusted relation-type logits

Let `z_ijr` be the frozen Gate A raw relation-type logit. Gate C computes:

\[
z'_{ijr}
=
z_{ijr}
+
\beta \log(\max(C_{ijr}, \epsilon)).
\]

Constants are fixed before validation:

```text
epsilon = 1e-8
beta grid = [0.0, 0.25, 0.5, 1.0, 2.0]
```

`epsilon` is a numerical floor only and is not tuned.

The adjusted relation type is the deterministic argmax over `z'`, with the lowest relation inventory index winning exact ties.

The relation-existence logit/probability is never adjusted by `C` or `beta`.

### 7.1 Required baseline identity

At `beta = 0`:

\[
z'_{ijr}=z_{ijr}
\]

for every pair and relation label. Therefore Gate C with `beta=0` MUST reproduce Gate A relation-type decisions and final prediction records exactly at every relation-existence threshold. This is a hard implementation invariant, not merely an expected metric similarity.

## 8. Validation model selection

Gate C introduces one new validation-only hyperparameter, `beta`.

The prespecified search space is the Cartesian grid:

```text
beta      = [0.0, 0.25, 0.5, 1.0, 2.0]
threshold = [0.85, 0.86, ..., 0.99]
```

This gives 75 validation decoding combinations and requires no retraining or repeated encoder inference once validation inference/posteriors are cached in memory.

Choose the single frozen `(beta, threshold)` pair using this lexicographic objective:

1. higher strict all-relation F1;
2. higher primary entity F1, preserving the Gate A selection protocol even though entity predictions are unchanged;
3. lower `beta`;
4. lower relation-existence threshold.

The lower-beta tie-break explicitly prefers the less interventionist decoder when validation performance is indistinguishable.

`beta=0` is included as an internal no-profile control.

No test metric may participate in beta or threshold selection.

## 9. Evaluation modes and split guard

Gate C uses the same explicit split discipline as Gate B:

- `dev`: validation inference/evaluation only; `test_evaluated=false`;
- `full`: repeat frozen validation selection, then perform exactly one test inference/evaluation pass using the already-selected `(beta, threshold)`.

The first Gate C execution MUST be `dev` on Fold 1 / seed 42.

The test split remains inaccessible until:

1. implementation tests pass;
2. `beta=0` parity is demonstrated;
3. validation artifacts are inspected;
4. the Probabilistic Profile formula, beta grid, epsilon, compatibility tensor semantics, and selection rule are frozen.

## 10. Proposed components

Gate C should be implemented as new files/modules wherever practical so Gate A and frozen Gate B remain auditable.

### `journal/scsp/gate_c.py`

Pure/inference-side structures and functions, expected to include concepts such as:

```python
@dataclass(frozen=True, slots=True)
class EntityTypePosterior:
    span_key: tuple[object, ...]
    entity_types: tuple[str, ...]
    conditional_probabilities: tuple[float, ...]
    none_probability: float
    entity_probability: float

@dataclass(frozen=True, slots=True)
class ProbabilisticPairDecision:
    compatibility_scores: tuple[float, ...]
    adjusted_type_logits: tuple[float, ...]
    selected_index: int
```

Responsibilities:

- validate/normalize conditional non-NONE posteriors;
- construct or query deterministic `M_{arb}` from frozen profile data;
- compute `C_ijr`;
- compute schema-adjusted logits;
- deterministic argmax;
- build document-level prediction records while preserving Gate A aggregation and relation-existence behavior.

### `journal/scsp/gate_c_inference.py`

Responsibilities:

- call frozen Gate A `infer_window()` for authoritative candidates and relation scores;
- extract full entity posteriors for the already-retained spans using a second frozen-model pass;
- verify top-1/entity-score parity;
- return a Gate C inference bundle without mutating `WindowInference`.

### `journal/scsp/gate_c_artifacts.py`

Write split-specific artifacts sufficient to reproduce probabilistic decisions without gold labels.

### `journal/scripts/evaluate_gate_c_probabilistic_profile.py`

Evaluation-only driver using the frozen Gate A checkpoint, frozen Gate B profile files, prespecified beta grid, and unchanged threshold grid.

### Tests

At minimum:

```text
tests/test_scsp_gate_c_math.py
tests/test_scsp_gate_c_inference.py
tests/test_scsp_gate_c_decoder.py
tests/test_scsp_gate_c_artifacts.py
tests/test_gate_c_probabilistic_profile_cli.py
```

File names may be consolidated if that improves clarity, but Gate A protected modules must not be modified.

## 11. Artifact requirements

Every Gate C validation/test run must persist enough prediction-side information to independently reproduce `C`, adjusted logits, and the selected relation type.

At minimum, for each retained span save:

```text
document_id
window_index
start
end
top1_entity_type
entity_probability
none_probability
entity_types
conditional_non_none_posterior
```

For each scored relation pair save:

```text
document_id
window_index
source/target coordinates
source/target top-1 labels
existence_logit
raw_relation_type_logits
relation_types
compatibility_scores C_ijr
beta
adjusted_relation_type_logits
selected_index
selected_project_label
```

Validation run-level artifacts must additionally include:

```text
full beta x threshold score grid
selected beta
selected threshold
profile/canonicalization hashes
checkpoint hash and metadata
dataset/config hashes
environment and deterministic runtime metadata
run summary with test_evaluated=false
```

Test artifacts must use separate filenames and may only be created by explicit `full` mode.

## 12. Diagnostics for the journal analysis

Gate C should report at least:

- number of retained spans with posterior vectors;
- numerical posterior normalization errors/max deviation;
- top-1 parity mismatch count, required to be zero;
- entity-score parity max absolute difference;
- mean/median/min/max `C` by project relation label;
- fraction of pair/relation opportunities with `C` near 0, intermediate, and near 1;
- mean unresolved endpoint posterior mass at source and target;
- count/rate of relation-type argmax changes versus Gate A for each beta;
- count/rate of changes among pairs above each selected existence threshold;
- transition counts `raw_label -> adjusted_label`;
- emitted relation counts by project and canonical label;
- strict validation metrics for every beta/threshold pair;
- exact `beta=0` prediction parity result.

These diagnostics distinguish three outcomes important for the paper:

1. probabilistic compatibility changes many low-confidence decisions but not emitted relations;
2. it changes emitted relations without metric benefit;
3. it improves or degrades strict relation extraction.

All three are scientifically reportable; no rule is revised post hoc to force outcome 3.

## 13. TDD acceptance criteria

Production behavior must be introduced test-first.

Required acceptance behaviors include:

1. conditional non-NONE posterior sums to one;
2. posterior extraction reproduces Gate A top-1 label;
3. posterior extraction reproduces Gate A `entity_score` within tolerance;
4. compatibility score is in `[0,1]`;
5. one-hot resolved posteriors reproduce hard-profile compatibility;
6. unresolved endpoint mass is neutral rather than blocked;
7. `used-in` compatibility score is exactly one;
8. inverse relation rules swap endpoint types for compatibility lookup only;
9. `beta=0` returns raw relation logits exactly;
10. positive beta never increases a logit because `log(C)<=0`;
11. exact ties choose the lowest inventory index;
12. relation-existence logits and scores remain unchanged;
13. predicted entity spans remain unchanged;
14. candidate relation pairs remain unchanged;
15. `beta=0` document predictions equal Gate A exactly;
16. beta/threshold selection follows the declared lexicographic objective;
17. `dev` never requests or writes test artifacts;
18. checkpoint/dataset/config/profile hash mismatches fail before evaluation;
19. artifacts are deterministic and sufficient to recompute adjusted logits;
20. Gate A and frozen Gate B protected behavior remains unchanged when Gate C is unused.

## 14. Verification gate before the first experiment

Before any Gate C validation run:

1. run all targeted Gate C tests;
2. run the full `test_scsp*.py` regression suite;
3. verify protected Gate A files have no diff from the frozen parent lineage;
4. verify frozen canonicalization and task-profile file hashes are unchanged;
5. verify the configured beta grid is exactly `[0.0, 0.25, 0.5, 1.0, 2.0]`;
6. verify epsilon is exactly `1e-8`;
7. verify threshold grid remains `0.85-0.99`;
8. verify `beta=0` parity on synthetic tests and then on Fold-1 validation artifacts;
9. stop before `full` mode.

The first authorized experiment is only:

```text
Gate C dev
Fold 1
seed 42
frozen Gate A checkpoint
frozen Gate B profile v1
validation only
```

## 15. Interpretation policy

Gate C is not required to outperform Gate A or Gate B to be considered a valid experiment. The implementation and paper must preserve negative results.

In particular:

- do not alter `M`, unresolved handling, epsilon, or beta grid after viewing validation/test outcomes unless a demonstrated implementation defect exists;
- do not use test labels to choose beta, threshold, posterior definition, or compatibility semantics;
- do not silently reinterpret unresolved labels as incompatible;
- do not claim STIX invalidity for triples that are merely outside the project task profile;
- report `beta=0`, Hard Profile, and selected Probabilistic Profile results together so the effect of decoding constraints remains auditable.

This preserves the causal interpretation required for the journal contribution: any change in relation predictions comes from schema-aware decoding over a frozen extractor, not from retraining or test-driven rule engineering.
