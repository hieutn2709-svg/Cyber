# SecureBERT v1 dataset gate

## SecureBERT dev approved, 2026-10-08

The user approved the 12-epoch SecureBERT dev step after reviewing the completed
RoBERTa baseline. See `docs/superpowers/specs/2026-10-08-securebert-dev-approval.md`.
Both packages now support guarded dev; full/test remains unavailable. Earlier
references to a pending SecureBERT dev checkpoint below are historical.

The approved comparison changes the paired encoder/tokenizer package only.
The dataset gate is independent of training, predictions, splits and metrics.

## Epoch recovery (matched runs)

The interrupted first RoBERTa dev attempt reached epoch 11/12 but had no optimizer
or RNG state. It is retained as incomplete; the final `roberta_dev_v3` run uses the
same pinned corpus, fold, seed, hyperparameters and selection rule.

Matched smoke/dev runs now atomically save `recovery.pt` after each completed
epoch, including optimizer, scaler, Python/NumPy/Torch RNG, history and the selected
best checkpoint. Add `--resume` to the original matched training command to resume
an incomplete run in the same output directory. Identity checks require unchanged
code commit, package, configuration, dataset, mode and runtime profile. A completed
run or weights-only checkpoint cannot be resumed. An OS advisory lock rejects
concurrent writers. The intermediate v2 recovery diagnostic is excluded from final
metrics; v3 was resumed after epoch 1. Original Gate A defaults remain
unchanged. Full/test and SecureBERT dev authorization gates remain in force.

## Approved matched-70 continuation, 2026-10-07

The user approved the 70-window amendment and RoBERTa re-baseline. Both real
corpora were generated, with exact annotation preservation: 52 documents,
70 windows, 1,353 primary and 28 auxiliary entity occurrences, 564 relations
including 561 core-to-core. The original 67-window artifacts remain unchanged.

- RoBERTa dataset SHA: `16e7d45fd39e1fb774567d8374ffc0929633d973b1d27de8d7ba2f14f6ba9348`.
- SecureBERT dataset SHA: `ff60eba8a472f9a090bcb14be000ab18cb30fb9df68768a08a74de55c5d1b33d`.
- Shared character manifest SHA: `fb33d6639d0a7def4b5f66142817184463f9e4c451a5faa8040791e3e05c8961`.

Native token coordinates concatenate independently tokenized window content
per document, excluding specials. They are package-local and are NOT token
indices from full-document tokenization. Source character coordinates, original
window IDs and per-token offset mappings remain the cross-package mapping.
Only after-whitespace cuts from the approved audit are used.

The versioned `journal/configs/matched70_corpus_lock.json` binds the distributed
bundle. The builder reconstructs the complete original ancestor before deriving
the matched corpus. Both split files retain every original partition field except
the dataset hash. A rebuild from another code commit changes manifest provenance;
do not silently regenerate the release lock or substitute a different bundle.

Runtime preflight and guarded training are implemented for both packages.
Both no-gradient preflights passed with length 512 and hidden dimension 768.
Both 20-epoch overfit diagnostics and two-epoch smoke runs completed;
RoBERTa dev completed all 12 epochs, selecting epoch 12 at threshold 0.90:
validation relation F1 0.2538071066 and primary entity F1 0.3872180451.
The relation-first selection rule is unchanged; do not combine epoch-11 entity
F1 with epoch-12 relation F1. Test was not evaluated. The final run resumed after
epoch 1 and exactly reproduced all numerical history from the prior epochs 1–11.
See `matched70_execution_report.json` for the verified execution record.
The repository suite passed 244 tests. A fresh independent review found one
runtime compatibility defect (private Transformers revision metadata), fixed with
snapshot-path and model-file SHA verification and a RED-to-GREEN regression.
A resumed cache subsequently contained a truncated weight file; its hash failed
against the successful preflight. The wrapper now verifies model-file hashes
before training; the pinned file was restored without changing model identity.

### Matched execution commands

Use the delivered bundle outside the repository. Set `CTI70` to the directory
containing `common_window_manifest.json`, `roberta/` and `securebert/`.
Set `CTI_RUNS` to a new external run directory. Keep both paths quoted.

```bash
python -m journal.scripts.preflight_matched_encoders --package roberta \
  --bundle "$CTI70" --output-dir "$CTI_RUNS/preflight_roberta" --device cuda
python -m journal.scripts.train_matched_encoder --package roberta --mode overfit \
  --bundle "$CTI70" --preflight "$CTI_RUNS/preflight_roberta" \
  --output-dir "$CTI_RUNS/roberta_overfit" --device cuda
python -m journal.scripts.train_matched_encoder --package roberta --mode smoke \
  --bundle "$CTI70" --preflight "$CTI_RUNS/preflight_roberta" \
  --overfit-run "$CTI_RUNS/roberta_overfit" \
  --output-dir "$CTI_RUNS/roberta_smoke" --device cuda
python -m journal.scripts.train_matched_encoder --package roberta --mode dev \
  --bundle "$CTI70" --preflight "$CTI_RUNS/preflight_roberta" \
  --overfit-run "$CTI_RUNS/roberta_overfit" --smoke-run "$CTI_RUNS/roberta_smoke" \
  --output-dir "$CTI_RUNS/roberta_dev" --device cuda
```

Use `--device cpu` on CPU-only runtimes. Runtime preflight must run in the same
cache as training. Both model and tokenizer revisions are pinned and remote code
is disabled. The wrapper enforces Fold 1, split seed 11800, run seed 42, unchanged
heads/hyperparameters, decreasing overfit loss and successful smoke before dev.
SecureBERT supports preflight, overfit and smoke; its dev needs the later checkpoint.
Full/test is unavailable. Checkpoints are package-bound before state loading.
No test metric may be used to select splits, cuts, thresholds or checkpoints.

The sections below are historical evidence for the original corpus; their
unapproved/unrun statements apply to those earlier checkpoints only.

## Historical 67-window checkpoint, 2026-10-07

**Real RoBERTa parity PASSED. SecureBERT is blocked by window capacity.**
The verified bundle exactly matches the frozen semantic digest
`dbcc1d83bed30f1a7b286bb84ae95b4ee415ec6b70e6acdd6bdc8797422df631`:
52 documents, 67 windows, 1,353 primary and 28 auxiliary entity occurrences,
564 relations, including 561 core-to-core relations. Its dataset SHA-256 is
`751f692ab74e30096965cca1328a0b673a34177d717984bf1da0d83d370580cf`;
serialization differs from the original but canonical semantic equality is exact.
The clean local builder commit is `8cbdf7909c09751c9e873f40d38afe8a980d6970`.
Its exact code tree is published as `dc5500235539e5728c32074c8673a3e8d55c4176`;
both trees are `e2064f860c6d8b3841abe60dc9d796db39ee1c61`. Publication used the
GitHub connector, which assigns a different commit identity. The original
execution manifest is preserved unchanged.

The tokenizer-loading failure was traced to missing SOCKS support in HTTPX.
Install `httpx[socks]` in proxy-based runtimes; retain the configured proxy.
This execution used Transformers 5.18.0 and Tokenizers 0.23.2. The latest
regression run used PyTorch 2.14.1+cpu and passed all 214 tests. CUDA/model
compatibility has not been established and no model weights were loaded.

The reconstruction fixes preserve the original annotation coordinates while
allowing proven boundary whitespace; explicitly restore exclusion of all 20
reference-token overlap conflicts; and slice reference tokens after full-document
encoding to preserve contextual BPE IDs. The overlap rule is determined by the
reference tokenizer, never selected by SecureBERT. Default library behavior
still rejects ambiguous overlapping spans. Exact parity validates the actual
tokenizer behavior; an archived preprocessing script's prefix-space option
is not assumed to describe the frozen artifact's runtime.

SecureBERT loaded successfully and the real builder stopped with
`protocol_amendment_required` at document `366`, window index `4` (513 tokens).
A separate prediction-independent capacity audit checked all 67 windows and
all 1,381 retained entity occurrences: no target alignment issue was found,
but window indices 4, 7 and 8 of document 366 require 513, 516 and 520 tokens.
No SecureBERT dataset, checkpoint or prediction artifact was written.

The concrete 70-window matched proposal is in
`docs/superpowers/specs/2026-10-07-securebert-matched-window-amendment.md`.
It is **proposed, not approved or applied**. The frozen design and dataset-gate
plan require separate approval before changing window boundaries for either
encoder. Runtime integration, overfit, smoke, dev and full/test remain unrun.
See `securebert_v1_parity_audit.json` and `securebert_v1_capacity_audit.json`
for aggregate evidence; private reconstructed data remain outside Git.

## Recovery status, 2026-10-01

The following section is historical; the current status above supersedes its
tokenizer-loading and unverified-parity blockers.

GitHub's implementation branch was at `ddad941`; it contained the approved
plans but none of the reported local implementation commits. The unavailable
commits were not treated as verified or silently claimed as recovered. The
implementation was rebuilt from the approved plan with test-first changes on
`journal/scsp-q2-securebert-v1-resume`.

**Real reconstruction is BLOCKED at loading the pinned RoBERTa tokenizer.**
The initial source check found five `labels` records in document `23` without
`value.labels`: `cLkh2E389L`, `IEzv6HuyUK`, `PPePXMwUyp`, `rA4TOUUb2i`,
`Z6cRBq0G-z`. Each is absent from frozen entity spans and has no source relation
reference. An explicit, source-SHA-bound policy in
`journal/configs/securebert_source_exclusions_v1.json` now records these as
unlabeled/unreferenced regions. The loader remains strict by default and rejects
labeled, referenced, unlisted or unused exclusions. No label was inferred.

Using this policy, source validation passed; the next execution stopped when
Transformers could not load the pinned `FacebookAI/roberta-base` configuration.
No dataset was written. Actual RoBERTa parity and SecureBERT feasibility remain
unverified. A runtime able to load the exact pinned packages is needed next.

Two source relations carry `direction=left`; their frozen dataset endpoints
still follow `from_id` → `to_id`. The builder preserves that frozen convention
instead of changing annotation semantics during an encoder comparison.

## Frozen inputs

- RoBERTa: `FacebookAI/roberta-base` @ `c2a5e573587885ce23744cf330ee7c402f0df16f`.
- SecureBERT: `ehsanaghaei/SecureBERT` @ `3a47918dd874e5c769efd152b51c5756a953fb67`.
- Frozen dataset SHA-256: `190d3136edba33d89ee58f533e2d12cc6cac2842323e3168f6e3e0e71af72c48`.
- Required parity: 52 documents, 67 windows, 1,353 primary entities,
  28 auxiliary entities, 564 relations, 561 core-to-core relations.

Logical window character boundaries always come from the pinned RoBERTa
reference tokenizer. RoBERTa token IDs are sliced from the full document;
SecureBERT tokenizes those same character slices.
Character coordinates remain document-relative; token spans are local to the
encoded window, including its special-token positions. Entity surface text is
excluded from canonical parity; every other field and sequence order is checked.

## Verification

Run from the repository root in an environment with PyTorch and Transformers:

```bash
python -m unittest tests.test_scsp_encoder_dataset tests.test_scsp_encoder_dataset_artifacts tests.test_build_encoder_comparison_dataset_cli -v
python -m unittest discover -s tests -v
git diff --check
```

Tests use synthetic annotations; no model download or test-split evaluation is
needed. The initial restoration passed 205 repository tests before review.
See the recovery audit for final verification counts after the exclusion policy.

## Controlled execution

Set external paths to your authoritative source and frozen input. The output
must be outside this Git repository and must not already exist.

```bash
export CTI_SOURCE=/absolute/path/to/authoritative_annotations.json
export CTI_FROZEN=/absolute/path/to/train_multitask_v4_clean_windows.json
export B4_OUTPUT_ROOT=/content/drive/MyDrive/SCSP_Q2_runs/b4_securebert_v1
python -m journal.scripts.build_encoder_comparison_dataset \
  --mode roberta-parity --source "$CTI_SOURCE" \
  --source-exclusions journal/configs/securebert_source_exclusions_v1.json \
  --frozen-dataset "$CTI_FROZEN" --output-dir "$B4_OUTPUT_ROOT/data/roberta"
```

Inspect the passed RoBERTa bundle, then compute the SHA-256 of its actual
`encoder_dataset_manifest.json`. Only then run the second command:

```bash
export CTI_PARITY_SHA=REPLACE_WITH_ACTUAL_MANIFEST_SHA256
python -m journal.scripts.build_encoder_comparison_dataset \
  --mode securebert --source "$CTI_SOURCE" \
  --source-exclusions journal/configs/securebert_source_exclusions_v1.json \
  --frozen-dataset "$CTI_FROZEN" --output-dir "$B4_OUTPUT_ROOT/data/securebert" \
  --roberta-parity-manifest "$B4_OUTPUT_ROOT/data/roberta/encoder_dataset_manifest.json" \
  --roberta-parity-sha256 "$CTI_PARITY_SHA"
```

Each successful bundle contains `dataset.json`, `encoder_dataset_manifest.json`,
`tokenizer_audit.json`, and `alignment_exclusions.json`. All hashes and parity
checks pass before an atomic directory rename publishes the bundle. Existing
bundles are never intentionally overwritten. Retain these artifacts externally;
only code, synthetic fixtures, and aggregate/ID-only audits belong in Git.

An overflow reports proposed deterministic whitespace subwindows with
`protocol_amendment_required` and writes no dataset. Applying these subwindows
to both encoders requires separate design approval.

This implementation step does not authorize training, --mode dev, --mode full,
or test evaluation. Runtime/training integration remains blocked until the
real dataset gate passes and its artifacts are reviewed.

## Review and remaining verification

Independent review of `ddad941..eb6649b` found no Critical/Important issues.
The subsequent explicit-exclusion policy was verified with additional
RED→GREEN tests; it was not included in that review range.

Deferred minor findings: entity boundary exclusions currently omit character
coordinates and labels; malformed nested JSON may produce a traceback instead
of a structured blocked diagnostic. Both fail closed.

Before runtime integration, verify prefix-space behavior against real frozen
parity and adapt the reference-token global bounds for target-token candidate
conversion. No runtime readiness claim is made by this dataset-only gate.

Real frozen parity now verifies the reference behavior. Adapting global bounds
for native target-token candidate conversion remains pending, together with the
explicit matched-window amendment.
