# SCSP-CTI design specification

## Working title

**Schema-Constrained Span-Pair Modeling with Calibrated Decoding for
Low-Resource STIX Cyber Threat Intelligence Extraction**

## 1. Scientific problem

The current controlled study establishes a reproducible development baseline on
52 CTI documents but also exposes a specific end-to-end relation bottleneck.
The conservative V13 hybrid obtains approximately 0.710 entity F1 and 0.222
relation F1. When gold entity spans are supplied, relation F1 rises to roughly
0.40, while relation-type classification over gold candidate pairs is much
stronger. This pattern motivates the journal question:

> Can explicit span and entity-pair modeling, combined with probabilistic
> STIX-oriented compatibility and calibrated decoding, reduce endpoint and
> candidate-generation error propagation in low-resource CTI extraction?

The new method is named **SCSP-CTI: Schema-Constrained Span-Pair CTI
Extraction**.

The project does not optimize only for a higher score on the existing 52
reports. The journal contribution must also improve reproducibility,
calibration, structured-output validity, and external generalization.

## 2. Scope and non-goals

### Development scope

Primary development uses the corrected 52-document manifest already checked
into `experiments/cv_manifest/`. The controlled inventory contains ten core
entity types:

- attack-pattern
- campaign
- domain-name
- identity
- indicator
- intrusion-set
- location
- malware
- tool
- vulnerability

The relation ontology is versioned separately and derived from the supported
annotation/evaluation inventory. Custom or inverse labels must not be silently
claimed to be normative STIX relationship names.

### Non-goals

SCSP-CTI is not intended to:

- reproduce the complete original STIXnet implementation;
- claim complete semantic coverage of STIX 2.1;
- tune on the future frozen external test set;
- reconstruct missing V10/V13 training code and present it as original source;
- use deterministic rules as post-test manual patches;
- make a state-of-the-art claim without a directly comparable benchmark.

## 3. Architecture

The journal implementation is separated from legacy code:

```text
journal/
├── legacy/
│   └── provenance references only
├── scsp/
│   ├── encoder.py
│   ├── spans.py
│   ├── pairs.py
│   ├── model.py
│   ├── losses.py
│   ├── schema.py
│   ├── calibration.py
│   └── rescue.py
├── data/
├── evaluation/
├── configs/
├── scripts/
└── tests/
```

The primary data flow is:

```text
CTI text
  ↓
subword encoder (RoBERTa first; SecureBERT comparison later)
  ↓
contextual token states
  ↓
span proposal + entity typing
  ↓
span pruning
  ↓
ordered entity-pair generation
  ↓
relation-existence head
  ↓
relation-type head
  ↓
probabilistic task-profile compatibility
  ↓
validation-only calibration and thresholding
  ↓
optional deterministic rescue (final ablation only)
  ↓
STIX object construction
  ↓
OASIS STIX validator + task-profile report
```

The plain SpanPair model without schema is implemented and evaluated before any
schema or deterministic-rescue mechanism. This is required to identify the
source of any gain.

## 4. Span proposal

Given contextual token states

\[
H=(h_1,\ldots,h_T),
\]

a candidate span `s_i=(b_i,e_i)` has representation

\[
g_i = [h_{b_i};h_{e_i};\operatorname{AttnPool}(h_{b_i:e_i});E_w(w_i)],
\]

where `w_i=e_i-b_i+1` and `E_w` is a learned span-width embedding.

Entity probabilities are

\[
p_i(t)=\operatorname{softmax}(W_e g_i+b_e),
\]

with the ten supported entity types plus `NONE`.

### Span-width policy

A fixed arbitrary maximum span width is not prespecified. For each development
outer fold, the training partition determines the smallest width covering at
least 99.5% of training gold spans. The final external configuration freezes a
single width policy from development evidence before external gold is opened.

High-precision deterministic IOC candidates may optionally be injected even
when they exceed the learned span-width cap. They must carry explicit
`proposal_source` metadata and are evaluated in a separate rescue ablation.

### Span pruning

Span pruning uses entity non-`NONE` probability and a validation-selected
budget. The implementation must record, before relation scoring:

- gold span proposal recall;
- gold span recall after pruning;
- candidate count per document/window;
- recall by entity type and span-width bucket.

This diagnostic is mandatory because the journal hypothesis concerns endpoint
propagation.

## 5. Ordered entity-pair representation

For retained spans `s_i` and `s_j`, generate ordered pairs `(i,j)` because CTI
relations are directional.

The pair representation is

\[
q_{ij}=[g_i;g_j;g_i\odot g_j;|g_i-g_j|;c_{ij};E_d(d_{ij})],
\]

where `c_ij` is an attention-pooled representation of the context between the
spans and `E_d` is a learned distance embedding.

Candidate-pair generation must report:

- gold pair recall before distance filtering;
- gold pair recall after distance filtering;
- positive/negative candidate ratio;
- recall by relation type;
- recall by token-distance bucket.

A maximum relation distance may start from the legacy 96-token value, but any
journal value must be selected from development data only.

## 6. Relation decomposition

Relation prediction is split into two supervised decisions.

### Existence

\[
p_{ij}^{exist}=\sigma(W_xq_{ij}+b_x).
\]

The existence objective is designed for severe class imbalance. Initial
experiments use weighted/focal binary cross-entropy and development-only
negative sampling.

### Relation type

For a positive pair:

\[
z_{ij}^{r}=W_rq_{ij}+b_r,
\]

\[
p(r\mid i,j,exist)=\operatorname{softmax}(z_{ij}^{r}).
\]

The journal evaluator must distinguish:

1. end-to-end relation F1;
2. relation-existence performance;
3. relation-type performance conditioned on gold pairs;
4. gold-span relation performance.

This decomposition directly tests whether the proposed model fixes the
identified bottleneck rather than only changing the final aggregate score.

## 7. Probabilistic STIX-oriented compatibility

Two different notions are kept separate:

1. **OASIS STIX 2.1 conformance**: checked by an official/non-project validator;
2. **task-profile semantic compatibility**: a versioned project profile used by
   the extraction model.

The paper must never use these terms interchangeably.

Let

\[
M_{arb}\in\{0,1\}
\]

be the versioned task-profile tensor for source entity type `a`, relation type
`r`, and target entity type `b`.

Instead of applying a hard mask to top-1 entity predictions, SCSP uses the full
entity-type distributions:

\[
C_{ijr}=\sum_{a,b}p_i(a)p_j(b)M_{arb}.
\]

`C_ijr` is the expected compatibility of relation `r` under uncertain endpoint
types.

Schema-adjusted relation logits are

\[
\tilde z_{ijr}=z_{ijr}+\beta\log(C_{ijr}+\epsilon).
\]

The development study must compare three prespecified variants:

- `no_schema`: `z'=z`;
- `hard_profile`: incompatible relations masked using top-1 endpoint types;
- `probabilistic_profile`: expected compatibility adjustment above.

The profile is stored under:

```text
journal/configs/stix/
├── stix_2_1_normative_notes.json
├── task_relationship_profile_v1.json
├── relation_canonicalization_v1.json
└── README.md
```

Only semantically justified inverse/canonical mappings are allowed. The profile
must document the source and rationale for every allowed `(source, relation,
target)` triple.

## 8. Training objective

Primary losses are:

\[
L_{span}=CE(y_i^{span},p_i),
\]

\[
L_{exist}=FocalBCE(y_{ij}^{exist},p_{ij}^{exist}),
\]

\[
L_{type}=-\sum_{(i,j)\in\mathcal R^+}\log p(y_{ij}^{r}|i,j),
\]

and an optional schema regularizer

\[
L_{schema}=-\sum_{(i,j)}\log\left(\sum_{r\in A_{ij}}p(r|i,j)+\epsilon\right),
\]

where `A_ij` denotes relations allowed by the task profile under the relevant
endpoint-type distribution/assignment.

The complete objective is

\[
L=\lambda_sL_{span}+\lambda_xL_{exist}+\lambda_rL_{type}+\lambda_mL_{schema}.
\]

Legacy loss weights are not copied automatically into SCSP. They may be used as
initial search priors only. All journal hyperparameters are selected on
training/validation partitions and recorded in immutable run configuration.

## 9. Calibration and selective prediction

The phrase `confidence-aware` is used in the journal only when confidence is
measured/calibrated.

Scalar temperature scaling is the primary low-data calibration method:

\[
p_x^{cal}=\sigma(z_x/T_x),
\]

\[
p_r^{cal}=\operatorname{softmax}(z_r/T_r).
\]

`T_x` and `T_r` are fitted only to validation logits. No test labels are used.

The final relation score is initially defined as

\[
S_{ijr}=p_{ij}^{exist,cal}\,p_{ijr}^{type,cal}\,C_{ijr}^{\beta}.
\]

A relation is emitted only when `S_ijr >= tau`; otherwise the system abstains.
`tau` is validation-selected and saved with each fold/run.

Mandatory reliability outputs include NLL, Brier score, ECE, coverage, risk at
coverage, and an aggregate risk-coverage statistic.

## 10. Deterministic rescue

Deterministic rescue is implemented only after the no-schema SpanPair and SCSP
schema variants are complete.

Entity rescue may add high-precision structured IOCs (for example CVE/hash/URL
patterns) when the candidate is missing and does not conflict with a retained
high-confidence neural span.

Relation rescue is permitted only when:

1. both endpoint entities exist;
2. a prespecified lexical/template rule supports the relation;
3. the task-profile compatibility check passes;
4. a validation-selected rescue threshold passes.

Every rescue record must include rule ID, source, neural score, rescue score,
compatibility score, and decision. No rule is added after inspecting frozen
external errors.

## 11. STIX output and validation

STIX construction is separated from extraction scoring. The journal pipeline
produces:

1. extraction predictions;
2. task-profile compatibility report;
3. STIX object/bundle construction result;
4. OASIS validator output;
5. provenance metadata linking output to git commit, config, fold, seed, model
   revision, and calibration artifact.

A validator pass does not mean a relation is semantically correct, so
conformance and extraction metrics are reported separately.

## 12. Evaluation protocol

### Development corpus

Use the fixed current 52-document five-fold manifest. No resplitting is used for
the primary development comparison because comparability with V10/V13 is
required.

Development stages:

1. one-seed (`42`) engineering/debugging;
2. three-seed main CV (`42,123,2024`) for finalists;
3. optional five-seed stability analysis for the final model and strongest
   baselines.

### External corpus

A separately sourced CTI corpus is created later. It is partitioned into a
pilot portion for annotation-guideline development and a frozen confirmatory
portion. Model architecture, ontology/profile, thresholds, and calibration
procedure are frozen before final external gold is evaluated.

Current Bosch files already present in the repository are not eligible to serve
as an untouched confirmatory test set.

## 13. Metrics

Primary:

- strict exact entity micro precision/recall/F1;
- strict typed-endpoint relation micro precision/recall/F1.

Bottleneck diagnostics:

- candidate span recall;
- post-pruning span recall;
- candidate-pair recall;
- relation-existence AUPRC/F1;
- gold-span relation F1;
- conditional relation-type accuracy/F1 on gold pairs.

Reliability:

- NLL;
- Brier score;
- ECE;
- risk-coverage/AURC or an equivalent prespecified selective-prediction metric.

Structured output:

- OASIS validator bundle pass rate;
- task-profile-valid triple rate;
- dangling-reference rate;
- STIX object-construction failure rate;
- duplicate/conflict rate;
- mapping coverage.

Per-class and macro metrics are secondary because several labels are sparse.

## 14. Statistical analysis

The natural resampling unit is the document, not individual mentions or the
five folds.

Primary system comparisons use paired document bootstrap confidence intervals
for delta F1. Approximate randomization at document level may be added for
predeclared comparisons. Multiple primary comparisons use Holm correction.

The external evaluation is confirmatory only after model freeze. A weak external
result is reported rather than used to start a new tuning cycle.

## 15. Baselines and ablations

Minimum comparison matrix:

| ID | System | Purpose |
| --- | --- | --- |
| B0 | archived Rule/KB | deterministic historical baseline |
| B1 | archived V10 | contextual historical baseline |
| B2 | archived V13 | strongest legacy same-split baseline |
| B3 | plain SpanPair-RoBERTa | isolate new span/pair architecture |
| B4 | SpanPair-SecureBERT | domain-pretraining comparison |
| P0 | SCSP no schema | implementation control |
| P1 | SCSP hard profile | hard constraint ablation |
| P2 | SCSP probabilistic profile | main schema contribution |
| P3 | P2 + calibration | reliability contribution |
| P4 | P3 + deterministic rescue | final full system |

A strong published span-based relation extractor and one fixed open-LLM
structured-extraction baseline are added if implementation/compute permits,
without reducing the mandatory matrix above.

## 16. Reproducibility requirements

Every new journal run must record:

- git commit and dirty-tree flag;
- complete run config;
- data manifest/hash;
- encoder repository ID and immutable revision;
- fold and seed;
- Python/PyTorch/Transformers/CUDA environment;
- hardware identifier;
- best epoch and selection metric;
- validation logits;
- calibration artifact;
- raw test predictions;
- per-document TP/FP/FN;
- STIX validator report;
- runtime and peak VRAM.

New source must have unit tests for span construction, pair generation, schema
compatibility, loss behavior, calibration leakage guards, exact-match scoring,
and serialization.

## 17. Implementation acceptance gates

### Gate A: plain SpanPair foundation

- one fold/seed trains end-to-end;
- saved predictions can be rescored independently;
- candidate span and pair recall are logged;
- train/validation/test documents are disjoint;
- no schema logic or deterministic rescue is active.

### Gate B: schema contribution

- task profile is versioned and unit tested;
- no-schema, hard-profile, and probabilistic-profile variants use identical
  data/model capacity outside the schema mechanism;
- schema cannot read test labels;
- compatibility metrics are reproducible from saved predictions.

### Gate C: reliability

- temperature and thresholds fit validation only;
- calibration files are saved and reusable;
- ECE/Brier/risk-coverage are generated from raw probabilities.

### Gate D: external readiness

- architecture, ontology/profile, hyperparameters, seeds, and evaluation script
  are frozen in a git tag before confirmatory evaluation;
- external manifest and annotation protocol are frozen;
- external test is run once for the confirmatory claim.

## 18. Immediate implementation order

```text
legacy provenance freeze
→ plain SpanPair data structures/tests
→ plain SpanPair-RoBERTa model
→ endpoint/pair diagnostic harness
→ task-profile ontology audit
→ probabilistic schema layer
→ SecureBERT comparison
→ calibration/selective prediction
→ deterministic rescue
→ multi-seed experiment matrix
→ frozen external evaluation
→ statistical analysis
→ journal manuscript and artifact release
```

This ordering is mandatory for causal interpretability of ablations: the project
must be able to show whether gains originate from span/pair modeling, schema
knowledge, calibration, or deterministic rescue.
