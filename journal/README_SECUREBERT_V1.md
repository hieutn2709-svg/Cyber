# SecureBERT v1 dataset gate

The approved comparison changes the paired encoder/tokenizer package only.
The dataset gate is independent of training, predictions, splits and metrics.

## Recovery status, 2026-10-01

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
reference tokenizer. SecureBERT tokenizes those same character slices.
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
