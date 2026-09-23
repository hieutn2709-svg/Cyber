# SecureBERT v1 Runtime and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed B4 runtime and guarded training entrypoint that changes only the paired encoder/tokenizer package while preserving the frozen B3 SpanPair protocol.

**Architecture:** A B4 contract module compares the SecureBERT config and dataset manifest against frozen B3 settings before any model load. A separate runtime preflight validates the paired tokenizer/model package and writes immutable provenance; a thin training wrapper verifies those artifacts and delegates to the existing Gate A training path without changing Gate A/B/C behavior.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, existing `journal.scsp` SpanPair stack, JSON artifacts, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-17-securebert-v1-comparison-design.md`

**Dependency:** `docs/superpowers/plans/2026-09-23-securebert-v1-dataset-gate.md` must be implemented, and its real RoBERTa parity audit must have passed before Task 5 may execute a smoke run.

## Global Constraints

- B4 model and tokenizer are both `ehsanaghaei/SecureBERT` at immutable revision `3a47918dd874e5c769efd152b51c5756a953fb67`.
- Baseline encoder is `FacebookAI/roberta-base` at revision `c2a5e573587885ce23744cf330ee7c402f0df16f`.
- Parent design lineage is `281c9ade82b0b9376dc2c22ec319ca62377ecf29` or a descendant containing that exact spec.
- Fold is `1`, split seed is `11800`, and initial run seed is `42`.
- Schema mode is exactly `none`.
- Gate A training settings remain field-for-field equal to `journal/configs/gate_a_training.json`; maximum epochs are `12`.
- Span width coverage is `0.995`, retained span cap is `128`, minimum entity score is `0.05`, relation distance is `96`, and relation negative ratio is `6`.
- The B4 clean-window dataset must carry a passed RoBERTa parity ancestor and matching SecureBERT tokenizer identity.
- `trust_remote_code=False` is mandatory.
- `overfit`, `smoke`, and `dev` never score or write test artifacts; `full` remains unavailable in the B4 wrapper until a later explicit freeze approval.
- Gate C probabilistic decoding, Gate B schema logic, calibration, and rescue are not imported by B4.
- No real training run is authorized until the dataset gate, runtime preflight, and full regression suite pass.

## Review Focus

- A syntactically valid manifest may belong to another frozen dataset: Task 1 tests the SHA/provenance chain, not only field presence.
- Model and tokenizer vocabulary sizes can diverge despite matching names: Task 3 tests embedding/tokenizer equality.
- A wrapper may appear dev-only while delegated arguments still allow `full`: Task 4 tests the final delegated namespace.
- A checkpoint may be loaded with another tokenizer package: Task 4 tests checkpoint package metadata before state loading.
- Existing Gate A can regress through shared runtime changes: Task 5 requires the complete repository suite and a protected-file diff audit.

---

### Task 1: Frozen B4 experiment and dataset contracts

**Files:**
- Create: `journal/configs/b4_securebert_v1.json`
- Create: `journal/scsp/securebert_v1.py`
- Create: `tests/test_scsp_securebert_v1_contract.py`

**Interfaces:**
- Consumes: `GateAConfig`, B3 config JSON, Gate A training JSON, B4 dataset manifest, and RoBERTa parity manifest.
- Produces: `SecureBertV1Contract`, `load_securebert_contract(...)`, `validate_frozen_b3_settings(...)`, and `validate_encoder_dataset_chain(...)`.

- [ ] **Step 1: Write failing contract tests**

```python
class SecureBertV1ContractTests(unittest.TestCase):
    def test_approved_config_changes_only_encoder_package_and_output_identity(self):
        baseline = GateAConfig.from_json(BASELINE_CONFIG)
        candidate = GateAConfig.from_json(B4_CONFIG)
        validate_frozen_b3_settings(baseline, candidate)
        self.assertEqual(candidate.encoder_model, "ehsanaghaei/SecureBERT")
        self.assertEqual(
            candidate.encoder_revision,
            "3a47918dd874e5c769efd152b51c5756a953fb67",
        )

    def test_changed_downstream_setting_is_rejected_by_field_name(self):
        baseline = config_payload()
        candidate = {**baseline, "encoder_model": "ehsanaghaei/SecureBERT",
                     "encoder_revision": "3a47918dd874e5c769efd152b51c5756a953fb67",
                     "max_span_candidates": 256}
        with self.assertRaisesRegex(ValueError, "max_span_candidates"):
            validate_frozen_b3_settings(baseline, candidate)

    def test_dataset_chain_rejects_wrong_frozen_sha_or_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "frozen_dataset_sha256"):
            validate_encoder_dataset_chain(dataset_manifest(wrong_frozen_sha=True))
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            validate_encoder_dataset_chain(dataset_manifest(model_id="other/model"))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_securebert_v1_contract -v`

Expected: missing B4 module/config.

- [ ] **Step 3: Add the B4 config and immutable validation**

```json
{
  "experiment_name": "b4_plain_spanpair_securebert_v1",
  "schema_mode": "none",
  "encoder_model": "ehsanaghaei/SecureBERT",
  "encoder_revision": "3a47918dd874e5c769efd152b51c5756a953fb67",
  "split_seed": 11800,
  "seed": 42,
  "fold": 1,
  "span_width_coverage": 0.995,
  "max_span_candidates": 128,
  "min_entity_score": 0.05,
  "max_relation_token_distance": 96,
  "relation_negative_ratio": 6,
  "output_dir": "journal/runs/b4_plain_spanpair_securebert_v1"
}
```

Permit differences only in `experiment_name`, `encoder_model`, `encoder_revision`, and `output_dir`; verify every other base-config field. Validate the dataset chain's source/frozen hashes, passed parity status, parent parity-manifest SHA, model/tokenizer IDs and revisions, counts, and semantic digest presence.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_securebert_v1_contract -v`

Expected: all contract tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add journal/configs/b4_securebert_v1.json journal/scsp/securebert_v1.py tests/test_scsp_securebert_v1_contract.py
git commit -m "feat: freeze SecureBERT v1 comparison contract"
```

### Task 2: Encoder-package provenance and deterministic artifact schema

**Files:**
- Create: `journal/scsp/encoder_package.py`
- Create: `tests/test_scsp_encoder_package.py`

**Interfaces:**
- Consumes: tokenizer/model metadata, dataset manifest, config hashes, Git commit, device/runtime metadata.
- Produces: `EncoderPackageIdentity`, `RuntimeCompatibilityReport`, `build_encoder_package_artifact(...)`, and `validate_checkpoint_encoder_package(...)`.

- [ ] **Step 1: Write failing provenance tests**

```python
class EncoderPackageTests(unittest.TestCase):
    def test_artifact_records_paired_package_and_hash_chain(self):
        artifact = build_encoder_package_artifact(
            identity=approved_identity(),
            tokenizer_class="RobertaTokenizerFast",
            model_class="RobertaModel",
            vocab_size=50265,
            embedding_vocab_size=50265,
            hidden_size=768,
            max_positions=514,
            dataset_manifest_sha256="d" * 64,
            combined_config_sha256="c" * 64,
            git_commit="a" * 40,
        )
        self.assertEqual(artifact["model"], artifact["tokenizer"])
        self.assertEqual(artifact["trust_remote_code"], False)

    def test_checkpoint_rejects_another_tokenizer_revision(self):
        checkpoint = {"encoder_package": approved_artifact()}
        current = approved_artifact()
        current["tokenizer"]["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "tokenizer revision"):
            validate_checkpoint_encoder_package(checkpoint, current)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_package -v`

Expected: encoder-package module import failure.

- [ ] **Step 3: Implement immutable package records and checkpoint validation**

```python
@dataclass(frozen=True, slots=True)
class EncoderPackageIdentity:
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    license_id: str

    def validate(self) -> None:
        if self.model_id != self.tokenizer_id:
            raise ValueError("model and tokenizer IDs must match")
        if self.model_revision != self.tokenizer_revision:
            raise ValueError("model and tokenizer revisions must match")
```

Use strict expected fields and exact revision matching. The artifact records resolved classes, vocabulary sizes, hidden size, max positions, special-token IDs, dataset/config hashes, Git commit, parameter counts, license, and upstream URL.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_encoder_package -v`

Expected: all provenance tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add journal/scsp/encoder_package.py tests/test_scsp_encoder_package.py
git commit -m "feat: record paired encoder package provenance"
```

### Task 3: No-gradient SecureBERT runtime compatibility preflight

**Files:**
- Create: `journal/scripts/preflight_securebert_v1.py`
- Create: `tests/test_preflight_securebert_v1.py`

**Interfaces:**
- Consumes: approved B4 config, dataset manifest, inventory, training config, output directory, and device choice.
- Produces: `runtime_compatibility.json` and `encoder_package.json`; no checkpoint or prediction file.

- [ ] **Step 1: Write failing pure-validator and CLI tests**

```python
class SecureBertRuntimePreflightTests(unittest.TestCase):
    def test_runtime_validator_requires_roberta_shape_and_vocab_match(self):
        report = validate_runtime_components(
            model_config=SimpleNamespace(
                model_type="roberta", hidden_size=768,
                vocab_size=50265, max_position_embeddings=514,
            ),
            embedding_vocab_size=50265,
            tokenizer_metadata=tokenizer_metadata(vocab_size=50265),
            last_hidden_state_shape=(1, 5, 768),
            encoder_parameter_count=124_000_000,
            head_parameter_count=1_000_000,
            required_input_length=512,
        )
        self.assertEqual(report["status"], "passed")
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            validate_runtime_components(
                model_config=SimpleNamespace(model_type="roberta", hidden_size=768,
                    vocab_size=50265, max_position_embeddings=514),
                embedding_vocab_size=50264,
                tokenizer_metadata=tokenizer_metadata(vocab_size=50265),
                last_hidden_state_shape=(1, 5, 768),
                encoder_parameter_count=1, head_parameter_count=1,
                required_input_length=512,
            )

    def test_help_does_not_import_torch_or_transformers(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], cwd=ROOT,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_preflight_securebert_v1 -v`

Expected: preflight module/file is missing.

- [ ] **Step 3: Implement lazy runtime loading and validation**

Load `AutoTokenizer` and `AutoModel` with the same approved ID/revision, `use_fast=True`, and `trust_remote_code=False`. Build `GateASpanPairModel` with the loaded encoder, run one synthetic no-gradient batch, assert `[batch, tokens, hidden]`, non-empty disjoint optimizer parameter groups, deterministic CUDA flags, valid distinct special-token IDs, vocabulary equality, and sufficient position capacity. Write artifacts only after every check passes.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_preflight_securebert_v1 -v`

Expected: pure validation and help tests pass without a model download.

- [ ] **Step 5: Compile the runtime-preflight files**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m py_compile journal/scsp/encoder_package.py journal/scripts/preflight_securebert_v1.py tests/test_preflight_securebert_v1.py`

Expected: exit code 0 and no output.

- [ ] **Step 6: Commit Task 3**

```bash
git add journal/scripts/preflight_securebert_v1.py tests/test_preflight_securebert_v1.py
git commit -m "feat: add SecureBERT runtime preflight"
```

### Task 4: Dev-only training wrapper and checkpoint package guard

**Files:**
- Create: `journal/scripts/train_securebert_v1.py`
- Create: `tests/test_train_securebert_v1_cli.py`
- Modify: `journal/scripts/train_gate_a.py`
- Modify: `tests/test_train_gate_a_cli.py`

**Interfaces:**
- Consumes: Gate A training arguments plus `--encoder-dataset-manifest`, `--runtime-compatibility`, and `--encoder-package`.
- Produces: delegated Gate A `overfit`/`smoke`/`dev` execution, B4 provenance in `training_run_config.json`, and checkpoint `encoder_package`; rejects `full` before dataset/model loading.

- [ ] **Step 1: Write failing wrapper and checkpoint tests**

```python
class TrainSecureBertV1CliTests(unittest.TestCase):
    def test_parser_does_not_offer_full_mode(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], cwd=ROOT,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("full", result.stdout)

    def test_delegated_namespace_cannot_evaluate_test(self):
        args = build_delegated_args(wrapper_args(mode="dev"))
        self.assertEqual(args.mode, "dev")
        self.assertFalse(gate_a.mode_evaluates_test(args.mode))
        with self.assertRaisesRegex(ValueError, "full.*not authorized"):
            build_delegated_args(wrapper_args(mode="full"))

    def test_checkpoint_package_mismatch_fails_before_state_load(self):
        checkpoint = checkpoint_fixture(tokenizer_revision="b" * 40)
        with self.assertRaisesRegex(ValueError, "tokenizer revision"):
            validate_b4_checkpoint(checkpoint, approved_artifact())

    def test_run_summary_records_runtime_and_parameter_counts(self):
        summary = build_b4_run_summary(
            gate_a_summary={"mode": "smoke", "test_evaluated": False},
            elapsed_seconds=12.5,
            peak_vram_bytes=1024,
            encoder_parameter_count=100,
            head_parameter_count=20,
        )
        self.assertEqual(summary["elapsed_seconds"], 12.5)
        self.assertEqual(summary["peak_vram_bytes"], 1024)
        self.assertEqual(summary["parameter_counts"], {
            "encoder": 100, "downstream_heads": 20, "total": 120,
        })
        self.assertFalse(summary["test_evaluated"])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_train_securebert_v1_cli -v`

Expected: wrapper module/file is missing.

- [ ] **Step 3: Add narrow Gate A provenance hooks without changing default behavior**

Extend `train_gate_a.run(args, *, run_metadata=None, checkpoint_metadata=None)` with default `None`. Merge validated metadata into `training_run_config.json` and the saved checkpoint; preserve byte-for-byte behavior of all computations and existing callers when both are `None`.

```python
payload = {
    "model_state_dict": model.state_dict(),
    "epoch": epoch,
    "threshold": best_threshold,
    "validation": best_selection,
    "git_commit": commit,
    "dataset_sha256": dataset_sha256,
    "config_sha256": config_sha256,
    "width_cap": width_cap,
}
if checkpoint_metadata is not None:
    payload.update(checkpoint_metadata)
torch.save(payload, checkpoint_path)
```

- [ ] **Step 4: Implement the B4 wrapper**

Validate all three artifact hashes and statuses before importing torch or
delegating. Restrict choices to `overfit`, `smoke`, and `dev`; hardcode the B4
config path and existing Gate A training/inventory/partition paths; confirm
fold/seed `1/42`; pass encoder-package provenance into both metadata hooks.
Measure elapsed wall time and CUDA peak allocated memory around the delegated
run, then write `b4_comparison_summary.json` with encoder/head/total parameter
counts, runtime, peak VRAM, validation metrics, and `test_evaluated: false`.

- [ ] **Step 5: Run focused B4 and Gate A regression tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_train_securebert_v1_cli tests.test_train_gate_a_cli -v`

Expected: all wrapper tests pass and Gate A CLI tests remain green.

- [ ] **Step 6: Commit Task 4**

```bash
git add journal/scripts/train_securebert_v1.py journal/scripts/train_gate_a.py tests/test_train_securebert_v1_cli.py tests/test_train_gate_a_cli.py
git commit -m "feat: add dev-only SecureBERT training wrapper"
```

### Task 5: Pre-experiment verification and controlled execution handoff

**Files:**
- Modify: `journal/README_SECUREBERT_V1.md`
- Modify only for verified defects: files created or changed in Tasks 1–4

**Interfaces:**
- Consumes: passed real dataset-gate artifacts, pinned runtime environment, and all B4 code/tests.
- Produces: verified commands for compatibility preflight, one-window overfit, smoke, and later Fold 1/seed 42 `dev`; this task stops before real `dev` unless separately authorized.

- [ ] **Step 1: Document the guarded run sequence**

Record exact commands and output directories for compatibility preflight, overfit, smoke, regression verification, and `dev`. Mark `dev` as a later explicit execution checkpoint and state that `full`/test evaluation is unavailable from the wrapper.

- [ ] **Step 2: Run all targeted B4 tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_scsp_securebert_v1_contract tests.test_scsp_encoder_package tests.test_preflight_securebert_v1 tests.test_train_securebert_v1_cli -v`

Expected: 0 failures and 0 errors.

- [ ] **Step 3: Run the complete repository suite in the provisioned PyTorch environment**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`

Expected: 0 failures and 0 errors. Absence of PyTorch/Transformers is an environment blocker and must not be reported as a passing suite.

- [ ] **Step 4: Run the real no-gradient runtime preflight**

Run:

```bash
export B4_OUTPUT_ROOT=/content/drive/MyDrive/SCSP_Q2_runs/b4_securebert_v1
python -m journal.scripts.preflight_securebert_v1 \
  --config journal/configs/b4_securebert_v1.json \
  --training-config journal/configs/gate_a_training.json \
  --inventory journal/configs/gate_a_label_inventory.json \
  --encoder-dataset-manifest "$B4_OUTPUT_ROOT/data/securebert/encoder_dataset_manifest.json" \
  --device cuda \
  --output-dir "$B4_OUTPUT_ROOT/runtime_preflight"
```

Expected: exit code 0; `runtime_compatibility.json` says `passed`; model and tokenizer revisions match the approved SecureBERT commit; no checkpoint or prediction artifact exists.

- [ ] **Step 5: Run one-window overfit only after Step 4 passes**

Run the documented wrapper command with `--mode overfit`. Expected: decreasing diagnostic loss, `test_evaluated: false`, and no `test_*` artifact.

- [ ] **Step 6: Run smoke only after the overfit diagnostic is reviewed**

Run the documented wrapper command with `--mode smoke`. Expected: configured two-epoch train/validation execution, `test_evaluated: false`, and no `test_*` artifact.

- [ ] **Step 7: Stop before `dev` and inspect artifacts**

Verify package hashes, dataset provenance, selected validation threshold, entity/relation diagnostics, runtime, VRAM, and absence of test artifacts. Do not execute `dev` or `full` in this implementation task.

- [ ] **Step 8: Verify diff scope and commit documentation**

Run: `git diff --check origin/journal/scsp-q2-securebert-v1-design...HEAD && git diff --name-only origin/journal/scsp-q2-securebert-v1-design...HEAD`

Expected: no Gate B/Gate C production file changes; Gate A change is limited to optional provenance hooks covered by its existing and new tests.

```bash
git add journal/README_SECUREBERT_V1.md docs/superpowers/plans/2026-09-23-securebert-v1-runtime-training.md
git commit -m "docs: document SecureBERT runtime gates"
```

## Completion boundary

Implementation completion means the real dataset parity gate, no-gradient runtime preflight, overfit diagnostic, smoke run, and full regression suite have passed with no test artifact. Fold 1/seed 42 `dev` remains a separate validation experiment authorization. `full` remains unavailable until a later frozen-selection review.
