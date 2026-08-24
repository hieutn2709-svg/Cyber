# Reproducibility Artifact Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public repository describe and verify the exact V10--V13 run partitions, architecture, configuration, and saved evaluation artifacts.

**Architecture:** Treat the archived V10 `fold_manifest_v4.json` and the saved seed-42 run partitions as immutable provenance inputs. Derive the human-readable document manifest, outer split files, and fold summaries from those inputs and the checked-in annotations; verify result counts against the saved V13 per-fold table. Keep the 61-tag entity head and four-class role head separate in every active configuration and document.

**Tech Stack:** Python standard library, JSON, CSV, YAML documentation, `unittest`, Git/GitHub.

**Spec:** User audit dated 2026-08-24 in this conversation; source evidence in Google Drive `models_checkpoints/reviewer_v10` through `reviewer_v13_conservative`.

## Global Constraints

- The run manifest uses split seed `11800`; the evaluation/model run seed is `42`.
- The five run test folds contain `11, 11, 10, 10, 10` complete documents.
- The architecture is `61` BIEOS/type sequence tags plus a separate role head with `O`, `ROLE_1`, `ROLE_2`, and `ROLE_BOTH`.
- Task-loss weights are entity `0.80`, role `0.05`, and relation `0.15`.
- V13 fold gold-relation counts are `128, 106, 137, 110, 83` after preprocessing; corresponding raw annotation counts for the same document sets are `134, 106, 138, 111, 85`.
- Do not change the repository license without an explicit owner decision.
- Do not synthesize missing experimental metrics; use only saved results or values recomputed from saved predictions.

---

### Task 1: Lock the run provenance with failing tests

**Files:**
- Modify: `tests/test_reproducibility.py`
- Create: `experiments/cv_manifest/fold_manifest_v4.json`
- Create: `experiments/cv_manifest/run_partitions_seed_42.json`

**Interfaces:**
- Consumes: checked-in `data/Annotations.json`, saved V10 manifest, saved V10 run result partitions, and V13 per-fold counts.
- Produces: assertions for exact document IDs, raw/evaluable relation counts, train/validation/test disjointness, and dataset hash provenance.

- [ ] **Step 1: Add tests for the exact five run folds and the 137-count explanation**

Add assertions that the stable document-ID folds are:

```python
EXPECTED_RUN_FOLDS = [
    ["405", "387", "413", "417", "402", "23", "414", "415", "425", "401", "410"],
    ["420", "423", "406", "407", "390", "388", "424", "376", "371", "396", "378"],
    ["370", "386", "372", "419", "421", "399", "395", "412", "403", "389"],
    ["377", "374", "381", "422", "409", "397", "398", "400", "418", "416"],
    ["426", "373", "393", "392", "411", "408", "375", "391", "366", "365"],
]
```

Assert raw relation counts `[134, 106, 138, 111, 85]` and V13 evaluable counts `[128, 106, 137, 110, 83]`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_reproducibility.ReproducibilityPreflight.test_official_document_manifest -v`

Expected: FAIL because the current repository manifest contains different document sets.

- [ ] **Step 3: Add the two immutable provenance manifests**

Store the exact V10 numeric fold manifest verbatim as `fold_manifest_v4.json`. Store each seed-42 train/validation/test partition, its stable document IDs, dataset SHA-256 `190d3136edba33d89ee58f533e2d12cc6cac2842323e3168f6e3e0e71af72c48`, and V13 gold counts in `run_partitions_seed_42.json`.

- [ ] **Step 4: Run the focused test**

Run: `python -m unittest tests.test_reproducibility.ReproducibilityPreflight.test_official_document_manifest -v`

Expected: PASS after Task 2 derives the public manifest from these provenance files.

### Task 2: Make `--check` derive the official manifest from the run manifest

**Files:**
- Modify: `experiments/build_document_cv.py`
- Modify: `experiments/cv_manifest/document_folds.json`
- Modify: `experiments/cv_manifest/document_folds.csv`
- Modify: `experiments/cv_manifest/fold_summary.csv`
- Modify: `experiments/cv_manifest/outer_splits/outer_fold_0.json`
- Modify: `experiments/cv_manifest/outer_splits/outer_fold_1.json`
- Modify: `experiments/cv_manifest/outer_splits/outer_fold_2.json`
- Modify: `experiments/cv_manifest/outer_splits/outer_fold_3.json`
- Modify: `experiments/cv_manifest/outer_splits/outer_fold_4.json`

**Interfaces:**
- Consumes: `fold_manifest_v4.json`, `run_partitions_seed_42.json`, and annotations in their checked-in order.
- Produces: stable document-ID manifests and `python experiments/build_document_cv.py --check` validation.

- [ ] **Step 1: Add a failing check-mode test**

Extend the existing check-mode test to require the output phrase `matches archived run manifest`.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_reproducibility.ReproducibilityPreflight.test_manifest_builder_check_mode_matches_checked_in_files -v`

Expected: FAIL because the builder still generates a new balanced split.

- [ ] **Step 3: Replace split generation with archived-manifest mapping**

Implement validation that numeric indices cover `0..51` exactly, map each index to the corresponding annotation task ID, and reject any mismatch between saved numeric and stable-ID partitions. Generate exact saved validation partitions instead of selecting new validation documents.

- [ ] **Step 4: Generate the derived manifest files**

Run: `python experiments/build_document_cv.py`

Expected: the five test folds match `EXPECTED_RUN_FOLDS`; fold 3 reports 138 raw relations and 137 evaluable relations.

- [ ] **Step 5: Verify GREEN**

Run: `python experiments/build_document_cv.py --check`

Expected: exit `0` and output `Fresh generation matches archived run manifest and checked-in derived files`.

### Task 3: Align canonical model configuration

**Files:**
- Modify: `Configs/reviewer_experiment.json`
- Modify: `Configs/model_config.yaml`
- Modify: `tests/test_reproducibility.py`

**Interfaces:**
- Consumes: saved `preflight_report.json` and `resolved_training_spec.json`.
- Produces: one canonical config mirrored consistently in YAML.

- [ ] **Step 1: Add exact config assertions**

Assert `max_seq_length=512`, physical batch `2`, gradient accumulation `2`, emission CE weight `1.5`, O-class weight cap `0.25`, max relation distance `96`, run seed `42`, and task-loss weights `{entity: 0.80, role: 0.05, relation: 0.15}`.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest tests.test_reproducibility.ReproducibilityPreflight.test_reviewer_configuration_is_canonical -v`

Expected: FAIL on the first missing field.

- [ ] **Step 3: Update JSON and YAML minimally**

Add the missing canonical fields and replace the obsolete YAML `ner_loss_weight: 0.8 / rel_loss_weight: 0.2` pair with the three task weights.

- [ ] **Step 4: Verify GREEN**

Run the focused configuration test; expect PASS.

### Task 4: Publish saved fold-level evidence

**Files:**
- Create: `experiments/results/rule_kb_baseline_per_fold.csv`
- Create: `experiments/results/contextual_neural_per_fold.csv`
- Create: `experiments/results/naive_hybrid_per_fold.csv`
- Create: `experiments/results/per_type_calibrated_hybrid_per_fold.csv`
- Create: `experiments/results/five_fold_results_conservative_hybrid.csv`
- Create: `experiments/results/five_fold_summary.csv`
- Create: `experiments/results/gold_span_diagnostic_per_fold.csv`
- Create: `experiments/results/gold_pair_type_accuracy_per_fold.csv`
- Create: `experiments/results/per_class/entity_per_class_by_fold.csv`
- Create: `experiments/results/per_class/relation_per_class_by_fold.csv`
- Create: `experiments/results/predictions/conservative_hybrid/fold_1_seed_42/test_predictions.jsonl`
- Create: equivalent prediction files for folds 2--5
- Create: `corpus_label_stats.csv`
- Modify: `experiments/results/README.md`
- Modify: `tests/test_reproducibility.py`

**Interfaces:**
- Consumes: saved Drive CSVs and predictions recovered in `repo_update_package.zip`.
- Produces: repository evidence that recomputes the V13 per-fold and per-class claims.

- [ ] **Step 1: Add failing artifact-integrity tests**

Assert every required file exists; for each V13 fold, assert `tp + fn` equals the run manifest gold count; sum per-class counts and assert they equal the fold-level totals.

- [ ] **Step 2: Verify RED**

Run the focused artifact test; expect FAIL because the fold-level files are absent.

- [ ] **Step 3: Copy only recovered evidence files**

Copy the listed CSV and JSONL files from the recovered package without altering numeric values.

- [ ] **Step 4: Verify GREEN**

Run the focused artifact test; expect PASS.

### Task 5: Document Drive artifacts and resolved limitations

**Files:**
- Modify: `README.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `experiments/PART1_STATUS.md`
- Modify: `experiments/results/README.md`
- Modify: `tests/test_reproducibility.py`

**Interfaces:**
- Consumes: verified Anyone-with-link Viewer folder URLs for `Cyber`, `models_checkpoints`, baseline, and V10--V13.
- Produces: reviewer-facing links, folder descriptions, and an explicit raw-versus-evaluable count explanation.

- [ ] **Step 1: Add failing documentation assertions**

Require a `Large artifacts` section, the exact `models_checkpoints` URL, the phrase `Anyone with the link — Viewer`, the two fold-count sequences, and the 61-plus-role-head architecture statement.

- [ ] **Step 2: Verify RED**

Run the documentation test; expect FAIL because the section is absent.

- [ ] **Step 3: Update active documentation**

Explain that the former repository manifest was generated by a different assignment algorithm; the run used the archived NumPy-permutation manifest. Explain that fold 3 has 138 raw annotations and 137 evaluable relations after preprocessing. Replace the old `artifacts missing` statements with links and precise remaining limitations.

- [ ] **Step 4: Verify GREEN**

Run the documentation test; expect PASS.

### Task 6: Full verification and branch publication

**Files:**
- Verify all modified and created files.

**Interfaces:**
- Consumes: completed Tasks 1--5.
- Produces: a pushed branch and reviewable pull request or branch URL.

- [ ] **Step 1: Run all static checks**

Run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python experiments/build_document_cv.py --check
```

Expected: all tests PASS and manifest check exits `0`.

- [ ] **Step 2: Inspect the complete diff**

Run: `git diff --check && git status --short && git diff --stat`

Expected: no whitespace errors; only planned files are changed.

- [ ] **Step 3: Commit and push the isolated branch**

```bash
git add README.md REPRODUCIBILITY.md Configs experiments corpus_label_stats.csv tests docs
git commit -m "Align run manifest and publish fold evidence"
git push -u origin fix/run-manifest-and-artifacts
```

- [ ] **Step 4: Verify the pushed branch**

Fetch the branch files through GitHub and confirm the commit SHA and updated manifest/config/result files are visible.
