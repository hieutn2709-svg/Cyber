# Paired development pilot of boundary replacement

We compared typed negative sampling with boundary replacement in two encoders. Each arm ran for 12 epochs from its original pinned pretrained weights, with seed 42, fold 1 and split seed 11800. Within each encoder, only the sampling policy differed. Architecture, gold labels, loss, optimizer, candidate generation and validation selection rules were held fixed.

| Encoder | Policy | Selected epoch | Threshold | Entity F1 (%) | Relation F1 (%) |
|---|---|---:|---:|---:|---:|
| roberta | typed | 12 | 0.90 | 38.722 | 25.381 |
| roberta | boundary_replace | 11 | 0.90 | 38.372 | 26.901 |
| securebert | typed | 8 | 0.90 | 31.447 | 20.438 |
| securebert | boundary_replace | 8 | 0.90 | 28.402 | 16.842 |

Differences are boundary replacement minus typed sampling in percentage points.

- roberta: relation F1 +1.520; primary entity F1 -0.350. Relation TP/FP/FN changed from 25/123/24 to 23/99/26.
- securebert: relation F1 -3.596; primary entity F1 -3.044. Relation TP/FP/FN changed from 14/74/35 to 16/125/33.

| Encoder and policy | Sampled conflicts / groups | Eligible conflicts / groups | Removed | Refilled | Shortfall |
|---|---:|---:|---:|---:|---:|
| roberta typed | 3 / 32711 | 813 / 4824356 | 0 | 0 | 0 |
| roberta boundary_replace | 0 / 32717 | 813 / 4866898 | 1 | 1 | 0 |
| securebert typed | 11 / 32866 | 837 / 5601364 | 0 | 0 | 0 |
| securebert boundary_replace | 0 / 32864 | 880 / 5627392 | 4 | 4 | 0 |

Counts sum 588 training-window records per arm over 12 epochs; repeated windows are not independent relations. Eligible counts describe candidates before intervention; sampled counts describe pairs used in the loss. After the first intervention, weights and candidates can diverge, so replacement-arm removals need not equal baseline conflicts.

All runs used frozen source 0089d1bbe4fbce847f8ed26d9605ac5de4748907 and matched recorded runtime versions (CPU, Torch 2.14.1+cpu, Transformers 5.18.0; four Torch threads and one interop thread). RoBERTa typed completed November 14 and replacement November 15. Both saved runs were restored November 16 without retraining. Their archives, selected tensors and predictions were freshly checked; recovery and optimizer checks are archived evidence from their completion dates, since recovery tensors are not included. Both SecureBERT arms ran November 16 and were verified against live recovery states. No same-physical-machine or speed claim is made.

For both pairs, sampled-batch hashes match before the first intervention and the first intervention counterfactual baseline hash matches the paired baseline. All 48 epochs and 2,352 training-window records were audited. Saved validation predictions were rescored with the official scorer. Selected checkpoints were saved separately; archives omit full optimizer recovery. Unfinished or unsaved November 13 attempts, the incomplete November 14 RoBERTa replacement attempt and the incomplete November 15 SecureBERT attempt are excluded.

This is a one-seed development pilot on five validation documents used to select both checkpoint and threshold. Descriptive differences do not establish significance, generalization, encoder ranking or speed. No full/test evaluation or merge was performed. Gold labels remain unchanged and independent human adjudication remains pending. A further prespecified multi-seed protocol and annotation adjudication are needed before choosing a default policy.
