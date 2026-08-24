# STIXnet-CyberEntRel reproducibility repository

This repository contains the code, document-level fold manifest, configuration,
and provenance records used to audit a STIX-oriented entity and relation
extraction prototype. Current claims are deliberately limited to results with
identified experimental support. Historical uploads that cannot support an
active result claim are retained under `archive/deprecated/`.

## Current same-split results

The primary comparison uses the same fixed five-fold document manifest for all
systems. Values are macro means and sample standard deviations over the five
outer folds.

| Configuration | Variant | Entity F1 | Relation F1 |
| --- | --- | ---: | ---: |
| Rule/knowledge-based same-split reproduction | baseline | 0.289 ± 0.051 | 0.011 ± 0.012 |
| Contextual neural | V10 | 0.698 ± 0.051 | 0.219 ± 0.041 |
| Conservative hybrid | V13 | 0.710 ± 0.051 | 0.222 ± 0.040 |

The machine-readable primary table is
[`experiments/results/current_results.csv`](experiments/results/current_results.csv).
V11 and V12 are preserved only as sequential development-stage diagnostics in
[`experiments/results/variant_provenance.csv`](experiments/results/variant_provenance.csv);
they are not independent confirmatory ablations.

Saved fold-level counts, diagnostics, per-class metrics, and V13 test
predictions are listed in
[`experiments/results/README.md`](experiments/results/README.md).

### STIXnet literature result versus this reproduction

The original STIXnet paper reports entity F1 0.916 and relation F1 0.724 in its
own experimental setting. Those are **literature-reported** numbers, not results
produced by this repository. The 0.289/0.011 row above is this project's
**same-split reproduction** of the available rule/knowledge-based comparator.
It is STIXnet-inspired rather than a faithful execution of the full original
STIXnet system, so the two rows are not directly comparable. See Marchiori,
Conti, and Verde, *STIXnet: A Novel and Modular Solution for Extracting All
STIX Objects in CTI Reports*, ARES 2023,
[DOI 10.1145/3600160.3600182](https://doi.org/10.1145/3600160.3600182).

## Reviewed architecture and configuration

The contextual model uses a `roberta-base` encoder, a two-layer BiGRU with
hidden size 250, eight attention heads, and dropout 0.35. Entity boundary/type
prediction uses 61 sequence tags: `O` plus B/I/E/S for 15 model entity types.
Relation roles are not folded into that tag inventory. They are predicted by a
separate four-class token head:

- `O`
- `ROLE_1`
- `ROLE_2`
- `ROLE_BOTH`

The optimizer uses encoder learning rate `3e-5`, task-head learning rate
`3e-4`, and weight decay `0.01`. Joint training is capped at 30 epochs with
patience 5. Relation refinement is capped at 12 epochs with patience 4. The
entity/role/relation task-loss weights are `0.80/0.05/0.15`. The maximum
sequence length is 512, the physical batch size is 2 with gradient accumulation
2, the emission-level CE weight is 1.5, the O-class weight cap is 0.25, and the
maximum relation distance is 96 tokens. The run seed is 42; it is distinct from
the split seed. The
canonical record is [`Configs/reviewer_experiment.json`](Configs/reviewer_experiment.json),
with a human-readable mirror in [`Configs/model_config.yaml`](Configs/model_config.yaml).

## Document-level evaluation

The official run manifest uses `split_seed=11800` and 52 complete documents.
Fold sizes are 11, 11, 10, 10, and 10. The exact archived numeric-index file is
[`experiments/cv_manifest/fold_manifest_v4.json`](experiments/cv_manifest/fold_manifest_v4.json),
and its stable document-ID translation is
[`document_folds.json`](experiments/cv_manifest/document_folds.json).

The earlier public `document_folds.json` was produced by a separate
label-balancing algorithm that also received the integer 11800; it was not the
allocation used by V10--V13. The archived run instead used the NumPy permutation
recorded in `fold_manifest_v4.json`. Reusing a seed across different assignment
algorithms does not produce the same folds. The checked-in derived manifest now
maps that archived allocation to stable annotation task IDs.

The two allocations contain the same 52-document universe exactly once, but
they do **not** assign those documents to the same folds: no old fold has the
same document set as any run fold. The previous public file itself declared
`split_seed=11800`; the value 42 is the model/evaluation run seed recorded with
the saved partitions, not a second name for that old split. Intersections
between each V13 run fold (rows) and the former public folds 0--4 (columns) are:

| V13 run fold | old 0 | old 1 | old 2 | old 3 | old 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 1 | 5 | 1 | 2 |
| 2 | 2 | 3 | 1 | 3 | 2 |
| 3 | 1 | 4 | 0 | 2 | 3 |
| 4 | 4 | 1 | 2 | 3 | 0 |
| 5 | 2 | 2 | 2 | 1 | 3 |

Consequently there is no content-preserving renumbering from the old folds to
the run folds. In the corrected artifact, V13 fold `k` (1--5) is
`fold_manifest_v4.json["folds"][k-1]` and is written to
`outer_splits/outer_fold_{k-1}.json`. The `k-1` relation is only a 1-based to
0-based filename convention; the old outer-fold contents have been replaced
by the archived run allocation.

Raw annotation relation counts for the five actual run folds are
`134, 106, 138, 111, 85`. Relations retained in the clean windowed evaluation
dataset are `128, 106, 137, 110, 83`. Thus V13 fold 3 contains 138 raw
annotations, of which 137 are evaluable after preprocessing; its reported
23 true positives plus 114 false negatives correctly total 137. The later V10
sanitation pass removed zero additional relations because it received the
already-cleaned window dataset.

The controlled evaluation's core entity types are:

`attack-pattern`, `campaign`, `domain-name`, `identity`, `indicator`,
`intrusion-set`, `location`, `malware`, `tool`, and `vulnerability`.

For outer fold `k`, fold `k` is held out for testing. Validation documents are
selected only from the remaining outer-training documents. Model checkpoints,
decoding choices, reconciliation policies, and thresholds are selected on that
validation partition and frozen before the corresponding test fold is scored.
Test reports do not participate in model or threshold selection.

See [`experiments/README.md`](experiments/README.md) and
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for commands and provenance limits.

## Preflight

The audit requires only the Python standard library:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python experiments/build_document_cv.py --check
```

The first command rejects conflicting active metrics/configuration and validates
the tag spaces, result tables, and split topology. The second compares a fresh
translation of the archived run manifest against the checked-in derived files
without rewriting them.

## Repository status and limits

- The raw 52-document corpus and annotations remain in `data/`.
- The checked-in source includes prototype extraction and integration modules.
- No new experiment was run during this audit and no synthetic result was
  generated.
- Fold-level CSVs, gold-span/gold-pair diagnostics, per-class metrics, and saved
  V13 test predictions are checked in under `experiments/results/`.
- Run logs and checkpoints are linked below rather than duplicated in Git.
- Immutable dependency/base-model revision locks and complete checkpoint hashes
  remain unresolved provenance limitations.
- Unsupported historical metrics, the former fixed-run bundle, the legacy
  combined-role label export, and the old notebook are explicitly deprecated in
  [`archive/deprecated/README.md`](archive/deprecated/README.md).

## Large artifacts

Sharing was verified on 2026-08-24 as **Anyone with the link — Viewer** for the
[Cyber Drive folder](https://drive.google.com/drive/folders/16Vi2N4a1bxDjOjdt9APTOtxus6JT3SoT)
and its
[`models_checkpoints` folder](https://drive.google.com/drive/folders/1karEftjWrFmWq7gdjLwD0BhVnOL44caY).

- [`stixnet_baseline_repro`](https://drive.google.com/drive/folders/1_gfJDaMibmTECF6_-2s6MPevkY8nkb8z): same-split rule/knowledge-based baseline outputs.
- [`reviewer_v10`](https://drive.google.com/drive/folders/1Bmv6sIfThPxLz1nYlupA3uxM9HWlRV4k): archived split manifest, preflight, resolved training specification, fold logs, checkpoints, predictions, and contextual-neural results.
- [`reviewer_v11_hybrid`](https://drive.google.com/drive/folders/1VXYI94JYkSmK-XBS8JLp-Y74B9gjPhV5): naive-hybrid fold selections and outputs.
- [`reviewer_v12_calibrated`](https://drive.google.com/drive/folders/1r8pymvR6hR_J28stQKUKswBst4_9-1In): per-type calibrated-hybrid fold selections and outputs.
- [`reviewer_v13_conservative`](https://drive.google.com/drive/folders/1ES3xljtU-h41xdHznnGXKKJI6PxRYNzZ): conservative-hybrid fold selections, test metrics, and saved test predictions.

## License and third-party material

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). The repository currently
does not grant an open-source license. It is publicly inspectable, but it must
not be described as open-source or as released under MIT/CC BY. Dataset, model,
paper, and third-party software rights remain with their respective owners.
