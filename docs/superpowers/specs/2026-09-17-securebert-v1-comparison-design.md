# SpanPair-SecureBERT v1 Comparison Design

**Status:** Approved model choice; design freeze pending review

**Design branch:** `journal/scsp-q2-securebert-v1-design`

**Parent commit:** `65464bafb999790b5072ba7b66b83f8b161e4c83`

**Baseline system:** B3 Plain SpanPair-RoBERTa

**Comparison system:** B4 SpanPair-SecureBERT v1

## 1. Purpose

B4 tests whether a cybersecurity-domain encoder package improves the frozen
SpanPair extraction architecture on the controlled CTI corpus. The comparison
must preserve the downstream model, losses, candidate construction, split
discipline, optimization settings, validation selection, and evaluation scope.

The approved SecureBERT package is:

- model and tokenizer: `ehsanaghaei/SecureBERT`;
- immutable Hugging Face revision:
  `3a47918dd874e5c769efd152b51c5756a953fb67`;
- architecture family: RoBERTa;
- upstream base lineage: `FacebookAI/roberta-base`;
- license recorded from the model card: `bigscience-openrail-m`.

SecureBERT 2.0 is excluded from B4. It uses ModernBERT, a different parameter
budget, and a longer context architecture, so it would confound domain
adaptation with architectural capacity.

## 2. Scientific claim boundary

SecureBERT v1 includes both domain-adapted weights and a customized tokenizer.
Consequently, B4 is a comparison of a **domain-specific encoder package**, not
a weights-only estimate of the causal effect of continued pretraining.

The manuscript may state that B4 evaluates domain-specific language-model
adaptation in the deployed SecureBERT package. It must not claim that any
observed delta is caused solely by pretraining data. Mixing SecureBERT weights
with RoBERTa token IDs, or RoBERTa weights with SecureBERT token IDs, is invalid
because embedding rows are defined by the paired vocabulary.

No additional tokenizer-only or weights-only control is part of this gate. Such
controls require a separate preregistered experiment and may not be introduced
after inspecting B4 validation results.

## 3. Frozen comparison contract

The following are identical between B3 and B4:

- 52-document controlled corpus and document identities;
- Fold 1 document partition and split seed `11800`;
- initial training seed `42`;
- 15 trainable entity labels plus `NONE`;
- ten-label primary entity evaluation scope;
- 13 relation labels and 564-relation all-evaluable scope;
- SpanPooler, between-span context pooler, entity head, relation-existence
  head, and relation-type head;
- span-width coverage `0.995` derived from Fold 1 training only;
- maximum retained spans `128`;
- minimum entity score `0.05`;
- maximum relation token distance `96`;
- entity and relation negative sampling ratios;
- loss functions and loss weights;
- optimizer family, learning rates, weight decay, gradient accumulation,
  gradient clipping, fp16 policy, and epoch budget;
- validation threshold grid and lexicographic selection rule;
- exact-match aggregation and reporting scopes;
- `dev`-before-`full` split guard.

B4 changes only the paired encoder and tokenizer package. Schema constraints,
Gate C probabilistic decoding, calibration, deterministic rescue, and any test-
driven rule are disabled.

## 4. Tokenization and data feasibility gate

The existing controlled clean-window file stores RoBERTa `input_ids`. Those
IDs cannot be passed to SecureBERT. Before any B4 training run, the project must
establish a tokenizer-neutral reconstruction path.

### 4.1 Required source representation

The reconstruction source must provide, directly or through a deterministic
provenance chain:

- stable document ID and original document order;
- canonical text or canonical token sequence with offsets;
- entity annotations in character or tokenizer-neutral coordinates;
- relation endpoints linked to stable entity IDs;
- enough information to reproduce window membership and global coordinates.

Raw CTI text and private annotations remain external to the public repository.
Only code, hashes, aggregate audits, and synthetic fixtures may be committed.

### 4.2 Preferred parity path

A new deterministic builder first regenerates the RoBERTa controlled windows
from the tokenizer-neutral source using:

- `FacebookAI/roberta-base`;
- immutable revision `c2a5e573587885ce23744cf330ee7c402f0df16f`;
- the original special-token, prefix-space, window, and overlap policy.

The regenerated RoBERTa dataset must match the frozen controlled dataset on:

- 52 document IDs and their order;
- 67 logical windows and stable window identities;
- all entity spans, labels, and stable entity IDs;
- all relation endpoints and labels;
- split-level document membership;
- aggregate primary/auxiliary entity counts;
- all 564 relations and the 561 core-to-core subset.

Byte-identical dataset SHA-256 is the preferred proof. If serialization-only
differences prevent byte identity, a canonical semantic digest must prove exact
equality after removing only declared non-semantic formatting fields.

### 4.3 SecureBERT reconstruction

After RoBERTa parity passes, the same source documents and annotations are
encoded with the pinned SecureBERT tokenizer. Entity character spans map to all
overlapping non-special subword offsets. Empty, discontinuous, truncated, or
ambiguous mappings fail closed and identify the document/entity without
silently dropping it.

Window policy must be tokenizer-neutral wherever the source permits. If a
logical window exceeds the encoder limit under SecureBERT tokenization, the
builder applies one predeclared deterministic subwindow rule to both encoders.
No rule may be selected from validation or test metrics.

### 4.4 Feasibility stop condition

If the original 52-document annotations or tokenizer-neutral coordinates
cannot be reconstructed, B4 stops before training. Reusing RoBERTa IDs is
forbidden. A matched re-baseline on a newly reconstructed corpus would be a
protocol amendment requiring a separate design approval; it is not an implicit
fallback.

## 5. Components

Implementation should add new B4-specific files wherever practical:

### `journal/scsp/encoder_dataset.py`

Pure data structures and functions for tokenizer-neutral annotations,
offset-based entity alignment, deterministic logical-window construction,
canonical semantic hashing, and audit aggregation. It must not discover data
splits or read test metrics.

### `journal/scripts/build_encoder_comparison_dataset.py`

CLI that loads the external tokenizer-neutral source, pins the tokenizer by
40-character revision, builds RoBERTa or SecureBERT windows, validates the
inventory, and writes a dataset plus provenance manifest. It records source
SHA-256, tokenizer identity/revision, tokenizer file hashes when available,
builder Git commit, window policy, and semantic digest.

The initial authorized execution is the RoBERTa parity audit. SecureBERT output
is generated only after that audit passes.

### `journal/configs/b4_securebert_v1.json`

A Gate A-compatible config containing the frozen downstream settings and the
pinned SecureBERT model revision. Its experiment name must identify B4 and must
not imply schema use.

### `journal/scripts/train_securebert_v1.py`

A thin guarded entrypoint over the existing plain SpanPair training path. It
validates the B4 config, paired dataset manifest, tokenizer/model identity, and
split guard before delegating to the existing training implementation. It adds
B4 provenance artifacts without changing Gate A behavior.

### Tests

New tests cover the builder, provenance guard, paired model/tokenizer contract,
config freeze, mode guard, artifact serialization, and CLI entrypoint. Existing
Gate A, Gate B, and Gate C tests remain unchanged and green.

## 6. Runtime compatibility preflight

Before any training, a no-gradient preflight must prove:

1. model and tokenizer load from the same pinned model ID and revision;
2. `trust_remote_code` is not required;
3. encoder `model_type` is RoBERTa-compatible;
4. hidden size is compatible with the existing dynamically sized SpanPair
   heads;
5. tokenizer vocabulary size equals the encoder input-embedding vocabulary;
6. special token IDs are valid and distinct as required;
7. maximum position capacity supports the frozen input length;
8. one synthetic batch produces `[batch, tokens, hidden]` last hidden states;
9. all model parameters and both optimizer parameter groups are non-empty;
10. deterministic CUDA settings remain enabled.

Any mismatch fails before the dataset or checkpoint is written.

## 7. Tokenization diagnostics

For both encoders, save prediction-independent diagnostics:

- tokenizer class, model ID, revision, and vocabulary size;
- special-token IDs and model maximum length;
- total subword count and subwords per non-whitespace character;
- per-document and aggregate window counts;
- entity span subword-length distribution;
- count of entities crossing a logical-window boundary;
- unknown-token count/rate;
- truncation and overflow counts;
- entity and relation preservation counts;
- source, dataset, manifest, and semantic SHA-256 values.

Diagnostics may expose aggregate counts and stable document IDs but must not
commit raw CTI text.

## 8. Training and selection protocol

The first B4 experiment is strictly:

```text
SpanPair-SecureBERT v1 dev
Fold 1
seed 42
12-epoch maximum
validation-only checkpoint and threshold selection
no schema, calibration, or rescue
no test evaluation
```

Checkpoint selection maximizes validation relation F1, then primary entity F1,
using the existing deterministic tie behavior. Relation-existence thresholds
come from the same frozen grid as B3. B4 does not inherit Gate A weights; its
SpanPair heads are newly initialized under the same seed and its encoder is
fine-tuned with the same optimizer policy.

The first run sequence is:

1. tokenizer/model compatibility preflight;
2. RoBERTa reconstruction parity audit;
3. SecureBERT dataset build and tokenization audit;
4. one-window overfit diagnostic;
5. smoke run on training plus validation;
6. fresh full regression verification;
7. Fold 1 / seed 42 `dev` run;
8. artifact inspection and interpretation;
9. stop before `full`.

## 9. Artifacts

Each B4 run writes the existing Gate A artifacts plus:

- `encoder_package.json`;
- `tokenizer_audit.json`;
- `encoder_dataset_manifest.json`;
- `runtime_compatibility.json`;
- parameter counts for encoder and downstream heads;
- peak VRAM and elapsed runtime;
- comparison summary against the frozen B3 validation result.

`encoder_package.json` records model/tokenizer IDs, immutable revisions,
resolved classes, config fields needed for reconstruction, license identifier,
and upstream source URL. Checkpoints repeat these identifiers so a checkpoint
cannot be loaded with another tokenizer.

## 10. Evaluation and interpretation

Primary B4 comparisons on Fold 1 validation are:

- strict all-evaluable relation precision/recall/F1;
- core-to-core relation precision/recall/F1;
- primary entity precision/recall/F1;
- candidate span and pair recall;
- gold-span relation-existence F1;
- relation-type accuracy on gold pairs;
- selected epoch and threshold;
- runtime, peak VRAM, and parameter counts.

Tokenization diagnostics are explanatory, not model-selection objectives. B4 is
not required to outperform B3. A tie, degradation, or gain is retained without
changing the tokenizer, window policy, hyperparameters, or label inventory.

No `full` run is authorized by this design. Test evaluation requires a later
freeze review after validation artifacts are inspected. SecureBERT 2.0 cannot
replace v1 based on the B4 result.

## 11. TDD acceptance criteria

Production behavior is introduced test-first. Tests must prove:

1. mutable model or tokenizer revisions are rejected before loading;
2. model and tokenizer IDs/revisions must match the declared package;
3. RoBERTa reconstruction parity failures identify the first mismatch;
4. canonical semantic digest is deterministic and order-sensitive;
5. special tokens never become entity span endpoints;
6. character spans map deterministically to overlapping subwords;
7. empty, discontinuous, ambiguous, and truncated mappings fail closed;
8. no entity or relation is silently dropped;
9. document-level split membership is unchanged;
10. B4 downstream hyperparameters equal the frozen B3 contract;
11. schema mode remains `none`;
12. `dev` cannot request, score, or write test artifacts;
13. runtime compatibility fails on vocabulary/embedding mismatch;
14. checkpoint loading rejects another tokenizer package;
15. all provenance hashes are serialized and reusable;
16. Gate A, Gate B, and Gate C behavior is unchanged when B4 is unused.

## 12. Verification gate before the first B4 experiment

Before the RoBERTa parity audit or any model run:

- compile every new Python file;
- run all targeted B4 tests;
- run the full `test_scsp*.py` regression suite;
- verify protected Gate A/Gate B/Gate C files have no unintended diff;
- verify the parent lineage is
  `65464bafb999790b5072ba7b66b83f8b161e4c83`;
- verify the SecureBERT revision is exactly
  `3a47918dd874e5c769efd152b51c5756a953fb67`;
- verify all frozen B3 training values byte-for-byte or field-for-field;
- verify mode guards with a synthetic test-split sentinel;
- stop before any `dev` experiment until reconstruction parity is reviewed.

## 13. Out of scope

- SecureBERT 2.0 or another domain encoder;
- hyperparameter tuning specific to SecureBERT;
- changes to label inventory or relation scope;
- schema-aware decoding, calibration, or deterministic rescue;
- multi-seed or multi-fold execution;
- external evaluation;
- test-set inspection;
- claims that isolate tokenizer effects from pretraining effects.

## 14. Decision summary

B4 uses the original RoBERTa-based SecureBERT v1 package at an immutable
revision. The comparison reuses the frozen SpanPair system and changes only the
valid paired encoder/tokenizer package. Because existing windows contain
RoBERTa token IDs, tokenizer-neutral reconstruction and RoBERTa parity are hard
preconditions. Failure of that gate stops the experiment rather than weakening
the comparison or silently changing the corpus.
