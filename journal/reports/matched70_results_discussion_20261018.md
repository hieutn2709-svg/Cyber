# Results and Discussion

## Controlled development comparison

We compared RoBERTa and SecureBERT v1 using 70 shared character windows from 52 documents. Each encoder used its own pinned tokenizer and pretrained weights. The document split was fixed (fold 1; split seed 11800), with 49 training windows, five validation windows, and 16 held-out test windows. The test partition was not evaluated. The five validation documents contained 120 primary entity mentions and 49 gold relations. We trained the baseline models for 12 epochs using seeds 42, 43, and 44. Checkpoints were selected by validation relation F1, with primary entity F1 as the tie-breaker. The selected epochs were 12, 10, and 12 for RoBERTa and 8, 11, and 10 for SecureBERT. All baseline-selected relation thresholds were 0.90.

We assessed candidate caps of 128 and 96 using the same selected checkpoint and threshold within each encoder–seed pair. The smaller cap was neither retrained nor assigned a separately optimized threshold. Table 1 reports means and sample standard deviations over three seeds; these describe optimization variability on one fixed validation set.

| Encoder | Candidate cap | Primary entity F1 (%) | Relation F1 (%) |
|---|---:|---:|---:|
| RoBERTa | 128 | 40.276 ± 1.812 | 29.916 ± 3.973 |
| RoBERTa | 96 | 44.328 ± 1.210 | 30.136 ± 3.597 |
| SecureBERT | 128 | 31.298 ± 1.142 | 24.271 ± 4.253 |
| SecureBERT | 96 | 36.116 ± 1.228 | 24.529 ± 3.914 |

RoBERTa achieved higher mean scores under this development protocol. This observation does not establish a general ranking of the encoders or isolate the effect of domain pretraining: the comparison concerns complete encoder–tokenizer packages. The original prototype's five-fold results use a different experimental design and are not direct paired estimates of improvement over the present validation results.

## Persistent errors across seeds

We assigned each gold relation to one of five mutually exclusive outcomes: correct prediction, loss of at least one correctly typed endpoint, retained endpoints without a scored pair, existence score below the selected threshold, or an incorrect relation type above that threshold. Endpoint loss was measured after candidate pruning and therefore does not identify a unique upstream cause. Gold identities were aligned across tokenizers using document IDs, entity IDs, and character spans.

At cap 128, RoBERTa recovered 25, 15, and 20 gold relations for seeds 42, 43, and 44. The corresponding endpoint-loss counts were 12, 17, and 14; threshold-rejection counts were 10, 15, and 13; and wrong-type counts were two in each run. SecureBERT recovered 14, 14, and 15 relations, with endpoint-loss counts of 19, 15, and 18, threshold-rejection counts of 14, 18, and 14, and two wrong-type outcomes per run. No additional gold relation was lost solely because a pair of retained endpoints was not scored.

RoBERTa failed to recover 22 of 49 gold relations in all three seeds and recovered 12 in every seed. SecureBERT never recovered 30 and consistently recovered 11. Across all six baseline runs, 22 relations were never recovered by either encoder. Sixteen of these occurred in documents 365 and 389, accounting for 72.73% of this persistently missed subset. The missed labels comprised eight uses, five targets, four targeted-by, two indicates, and one each of deliver, located-at, and originates-from. These patterns motivate document-level inspection of boundary, type, direction, and annotation ambiguity; they do not justify automatically modifying the gold annotations.

Endpoint errors also dominated false positives. At cap 128, at least one predicted endpoint failed to match a gold typed span in 137 of 207 RoBERTa false-positive occurrences (66.18%) and 114 of 170 SecureBERT occurrences (67.06%), pooled over seeds. Thirteen exact RoBERTa false positives and seven exact SecureBERT false positives recurred in all three seeds. Pooled occurrences should not be treated as independent errors.

## Candidate pruning and encoder complementarity

Reducing the cap yielded mean paired relation-F1 changes of +0.220 percentage points for RoBERTa and +0.258 for SecureBERT. The improvement occurred only at seed 42, where each encoder removed five false positives involving non-gold typed endpoints. The true-positive set remained unchanged in all six encoder–seed comparisons. At seeds 43 and 44, neither encoder changed its final relation TP, FP, or FN counts.

Nevertheless, unchanged F1 concealed reduced candidate coverage: four SecureBERT gold relations at seed 43 and one at seed 44 changed from threshold rejection to endpoint loss under cap 96. Thus, pruning reduced one error burden while eliminating some opportunities for a future relation classifier to succeed. We therefore retain cap 96 as a development ablation rather than claiming uniformly improved recall or efficiency. The available timings are not a repeated performance benchmark.

Paired true-positive overlap also varied by seed. At seed 42, both encoders recovered 14 relations, RoBERTa alone recovered 11, and SecureBERT alone recovered none. At seed 43 these counts were 11, four, and three; at seed 44 they were 14, six, and one. Consequently, the seed-42 observation of no SecureBERT-only true positives does not hold for every paired run. Across the union of all three seeds, however, SecureBERT added no true-positive identity outside the RoBERTa union. This gold-informed overlap analysis is not an ensemble evaluation; no ensemble precision or F1 is inferred.

## Reproducibility and limitations

All 12 serialized prediction sets were independently rescored using the project's strict scorer. Saved logits reproduced prediction identities and relation probabilities. One provenance exception must be disclosed: the archived fresh-forward gzip for SecureBERT seed 42 at cap 128 was truncated, retaining 31,527 complete records out of 34,136 expected. We used the SHA-256-verified original selected-baseline development logits for that case. Every readable fresh-forward record matched its original counterpart exactly by pair key, and decoding the complete original logits reproduced the saved predictions and metrics. The unavailable fresh-forward tail was not recovered and no additional forward pass was performed. The other 11 gzip streams were read in full.

The analysis contains 588 gold-relation outcomes, representing 49 relations repeatedly observed across 12 configurations. It does not contain 588 independent gold relations. With only five validation documents and three seeds, the results do not support statistical-significance or generalization claims. Validation was used for checkpoint and threshold selection and candidate-cap exploration, so the estimates remain exploratory. Further work should prioritize controlled endpoint-preservation and negative-sampling ablations, followed by a frozen evaluation design on independent documents. Schema constraints, calibration, and STIX serialization require separate evidence before their benefits can be claimed.
