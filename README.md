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
canonical record is [`Configs/reviewer_experiment.json`](Configs/reviewer_experiment.json),
with a human-readable mirror in [`Configs/model_config.yaml`](Configs/model_config.yaml).

## Document-level evaluation

The official manifest uses `split_seed=11800` and 52 complete documents. Fold
sizes are 11, 11, 10, 10, and 10. The controlled evaluation's core entity types
are:

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
in-memory manifest against the checked-in official files without rewriting
them.

## Repository status and limits

- The raw 52-document corpus and annotations remain in `data/`.
- The checked-in source includes prototype extraction and integration modules.
- No new experiment was run during this audit and no synthetic result was
  generated.
- Exact environment locks, independently verifiable run logs, checkpoints for
  V10--V13, and per-fold prediction artifacts are not present. These remain
  unresolved provenance limitations and are not implied by the summary tables.
- Unsupported historical metrics, the former fixed-run bundle, the legacy
  combined-role label export, and the old notebook are explicitly deprecated in
  [`archive/deprecated/README.md`](archive/deprecated/README.md).

## License and third-party material

See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). The repository currently
does not grant an open-source license. Dataset, model, paper, and third-party
software rights remain with their respective owners.
