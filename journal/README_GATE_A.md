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

## What is implemented

The Gate A foundation contains:

- immutable no-schema experiment configuration;
- fixed-manifest loading and train/validation/test leakage guards;
- training-derived span-width policy;
- exhaustive span proposal and deterministic score-based pruning;
- strict typed span recall diagnostics, including entity-type and width buckets;
- directional ordered entity-pair generation;
- pair recall, class-balance, relation-type, and token-distance diagnostics;
- PyTorch span pooling, entity typing, pair representation, relation-existence,
  and relation-type heads;
- imbalance-aware relation losses;
- JSONL prediction artifacts with run/data/config provenance;
- independent strict entity/relation rescoring from saved predictions;
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

The command refuses a file whose SHA-256 differs from the manifest. Before any
training is allowed, it writes:

```text
resolved_config.json
provenance.json
```

The current runner stops at `preflight_complete_training_not_started`. This is
intentional: the exact dataset adapter/training driver is added only after its
input schema has been verified against the controlled dataset. No placeholder
metric or synthetic journal result is emitted.

## Artifact contract for real runs

A completed Gate A training run must additionally preserve:

- raw test predictions in JSONL;
- validation logits used for any validation-only selection;
- span proposal recall before pruning;
- span recall after pruning;
- pair recall before and after distance filtering;
- relation-existence metrics;
- relation-type metrics conditioned on gold pairs;
- gold-span relation metrics;
- per-document strict TP/FP/FN;
- model/config/data hashes and the exact Git commit;
- runtime environment and hardware metadata.

The saved JSONL must be sufficient for an independent evaluator to reproduce the
reported strict entity and typed-endpoint relation counts without importing the
training loop.

## Gate A acceptance rule

The foundation code is not equivalent to a completed Gate A experiment. Gate A
is scientifically complete only after at least one real fold/seed trains
end-to-end on the hash-verified controlled dataset and the saved prediction
artifact can be independently rescored. Until then, no new F1 result should be
quoted in the journal manuscript.
