# SCSP-CTI Gate A: Plain SpanPair foundation

This directory documents the first journal implementation gate for SCSP-CTI.
Gate A is intentionally a **no-schema** baseline. It exists to determine whether
explicit span and ordered entity-pair modeling can reduce endpoint/candidate
error propagation before STIX-oriented constraints, calibration, or deterministic
rescue are introduced.

## Isolation

Gate A development lives on branch:

```text
journal/scsp-q2-gate-a-spanpair
```

The legacy paper code and `master` branch are not modified by this work.
Archived V10/V13 results remain historical comparators; this implementation does
not reconstruct missing legacy source and present it as original code.

## Fixed development evidence

The primary development comparison uses:

- manifest: `experiments/cv_manifest/run_partitions_seed_42.json`
- split seed: `11800`
- development seed: `42`
- five document-level folds
- expected controlled-dataset SHA-256:
  `190d3136edba33d89ee58f533e2d12cc6cac2842323e3168f6e3e0e71af72c48`

The controlled dataset itself is not committed to the public repository. Supply
it explicitly at execution time.

The controlled Gate A training inventory contains 15 trainable entity labels
plus `NONE`: 10 primary/core labels used for the main Entity F1 report and 5
auxiliary endpoint labels retained so annotated relations are not made
structurally impossible. See `DATA_SCOPE_DECISION.md`.

## What is implemented

The Gate A foundation contains:

- immutable no-schema experiment configuration;
- fixed-manifest loading and train/validation/test leakage guards;
- verified clean-window dataset adapter and aggregate audit;
- training-derived span-width policy;
- content-only exhaustive span proposal and deterministic score-based pruning;
- training-only gold-span injection for relation supervision;
- strict typed span recall diagnostics, including entity-type and width buckets;
- directional ordered entity-pair generation;
- training pair sampling that preserves all gold positives while applying the
  inference distance cutoff only to negatives;
- pair recall, class-balance, relation-type, and token-distance diagnostics;
- pinned RoBERTa runtime encoder;
- vectorized span attention pooling and between-span context pooling;
- PyTorch entity typing, pair representation, relation-existence, and
  relation-type heads;
- imbalance-aware relation losses;
- validation-only checkpoint and relation-threshold selection;
- multi-window local-to-document-global artifact aggregation;
- JSONL prediction artifacts with run/data/config provenance;
- independent strict entity/relation rescoring from saved predictions;
- primary/core scoped reporting in addition to all-evaluable metrics;
- preflight checks for immutable encoder revision, manifest consistency,
  dataset hash, and validation-only model selection.

No schema compatibility layer, deterministic rescue, calibration, or test-label
threshold tuning is active in Gate A.

## Dry-run preflight

From the repository root:

```bash
python journal/scripts/run_gate_a.py \
  --config journal/configs/gate_a_plain_spanpair.json \
  --manifest experiments/cv_manifest/run_partitions_seed_42.json \
  --fold 1 \
  --dry-run
```

A dry run does not require the private controlled dataset. It validates the
configuration and fixed document partition and prints the resolved provenance
record.

## Dataset-verified run preparation

When the controlled dataset is mounted locally:

```bash
python journal/scripts/run_gate_a.py \
  --config journal/configs/gate_a_plain_spanpair.json \
  --manifest experiments/cv_manifest/run_partitions_seed_42.json \
  --fold 1 \
  --dataset /path/to/train_multitask_v4_clean_windows.json \
  --output-dir journal/runs/gate_a_plain_spanpair/fold_1_seed_42
```

The command refuses a file whose SHA-256 differs from the manifest and writes
`resolved_config.json` plus `provenance.json` before model execution.

## Training execution modes

The journal training driver is `journal/scripts/train_gate_a.py`. Execution is
staged deliberately so test labels cannot influence debugging or selection.

### E1: one-window overfit diagnostic

```bash
python journal/scripts/train_gate_a.py \
  --mode overfit \
  --dataset /path/to/train_multitask_v4_clean_windows.json \
  --fold 1 \
  --seed 42 \
  --output-dir /path/to/runs/fold_1_seed_42_overfit
```

`overfit` selects one positive training window and never evaluates validation or
test. Its purpose is to detect broken gradients, labels, pair construction, or
loss wiring before a fold run.

### E2: two-epoch smoke run

```bash
python journal/scripts/train_gate_a.py \
  --mode smoke \
  --dataset /path/to/train_multitask_v4_clean_windows.json \
  --fold 1 \
  --seed 42 \
  --output-dir /path/to/runs/fold_1_seed_42_smoke
```

`smoke` trains for `smoke_epochs` from `gate_a_training.json`, evaluates only the
fixed validation partition, and writes validation logits/predictions,
checkpointing metadata, and candidate diagnostics. It does **not** evaluate the
test partition.

### E3: frozen full Fold-1 run

Run this only after E1 and E2 have been inspected:

```bash
python journal/scripts/train_gate_a.py \
  --mode full \
  --dataset /path/to/train_multitask_v4_clean_windows.json \
  --fold 1 \
  --seed 42 \
  --output-dir /path/to/runs/fold_1_seed_42_full
```

`full` selects the best checkpoint and relation threshold from validation only.
The fixed test partition is evaluated once after checkpoint and threshold are
frozen.

## Artifact contract for real runs

A completed Gate A training run preserves, as applicable:

- `resolved_config.json` and `provenance.json`;
- `environment.json` and `training_run_config.json`;
- `training_history.json` and `best_model.pt`;
- validation logits and validation prediction JSONL;
- validation-only threshold selection;
- span proposal and post-pruning diagnostics;
- pair recall before and after distance filtering;
- per-document strict TP/FP/FN-derived metrics;
- raw test prediction JSONL for a full run;
- all-evaluable relation metrics and core-to-core relation metrics;
- primary/core Entity F1 and all-trainable-label diagnostics;
- model/config/data hashes and exact Git commit.

The saved JSONL must be sufficient for an independent evaluator to reproduce the
reported strict entity and typed-endpoint relation counts without importing the
training loop.

Relation-existence metrics, relation-type metrics conditioned on gold pairs, and
gold-span relation diagnostics remain mandatory Gate A analysis outputs before
a journal result is treated as scientifically complete.

## Gate A acceptance rule

The code foundation is not equivalent to a completed Gate A experiment. Gate A
is scientifically complete only after at least one real fold/seed trains
end-to-end on the hash-verified controlled dataset, the saved prediction
artifact can be independently rescored, and the mandatory bottleneck
diagnostics have been reviewed. Until then, no new F1 result should be quoted in
the journal manuscript.
