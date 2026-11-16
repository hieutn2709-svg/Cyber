# Independent final review — E2 paired pilot, 2026-11-16

**Decision:** ready for descriptive development-pilot reporting and a draft-PR update. Keep the PR draft; this review does not authorize full/test evaluation, merge, a default-policy change or generalization claims. No Critical, Important or Minor correctness finding was identified within the verified scope below.

This was one fresh, read-only review of the completed four-arm experiment at source `0089d1bbe4fbce847f8ed26d9605ac5de4748907`, following `final_review_brief.md`. The reviewer wrote only this file. No training, neural forward, new inference, full/test evaluation, source edit, branch/index change or subagent execution occurred. Existing full-suite results were not rerun.

## Strengths and observed findings

- The intervention preserves the matched controls and reports the unfavorable outcomes. Relation F1 changes by +1.520 percentage points for RoBERTa and -3.596 for SecureBERT; primary entity F1 declines by 0.350 and 3.044 points respectively. Candidate gold-pair coverage also falls, from 37 to 33 and from 30 to 27 of 49. These are development-validation observations, not evidence for a uniformly better policy.
- Eligible candidate conflicts are kept distinct from conflicts in actual sampled loss batches. Replacement yields zero sampled conflict groups, with removals/refills of 1/1 and 4/4 and no shortfall. All denominators are repeated window/epoch observations, not independent relations.
- Historical versus fresh checkpoint evidence is accurately separated in `CTI_Q2_E2_Doi_chung_20261116.md:67`, `CTI_Q2_E2_Doi_chung_20261116.md:71`, the English insertion, restoration records and checkpoint delivery map. The older RoBERTa optimizer evidence is not represented as a newly loaded recovery checkpoint.
- Persistence handling is fail-closed. `persistence_gate.py:14` rejects an altered archive; `persistence_gate.py:22` blocks an unresolved prior write; `persistence_gate.py:29` halts on a failed helper; `persistence_gate.py:31` requires a confirmed successful file receipt. The final HTTP401 was an explicit failure, not a claimed successful or unknown write. The retained failure log, successful retry log/receipt and `delivery_recovery_20261116.json:4` consistently show a fresh-process retry of the same SHA-verified archive after the original driver exited. No retraining is needed or indicated. The final receipt at `securebert_boundary_replace_delivery_receipt.json:4` matches the delivery map and local archive SHA.

**Critical findings:** none found. **Important findings:** none found. **Minor findings:** none found. There is consequently no requested scientific-data or policy-code fix from this review.

## Exact independent checks performed

### Source, inputs, controls and provenance

- Read the frozen protocol and boundary-consistency design, aggregation/results/checkpoint/logit/restoration scripts, persistence gate and orchestrator, packaging scripts, Vietnamese and English Markdown reports, JSON results, delivery map and proposed PR update.
- Confirmed execution worktree HEAD equals the frozen SHA and `git status --porcelain` was empty at inspection. Independently rehashed all 13 corpus-lock files; all matched. The raw input files therefore match the pinned data without relying only on a success flag.
- Used the native cached-model validator, in local-files-only mode, to independently hash each original pretrained config/weight file and verify its pinned snapshot revision for both encoders. No model was instantiated or forwarded by this check. Run package pretrained hashes match the input verification records.
- Compared the complete run-config dictionaries within each encoder pair: removing only the replacement policy field makes them exactly equal. Verified dev mode, 12 epochs, seed 42, fold 1, split seed 11800, diagnostics enabled, frozen source and no test evaluation. All four saved execution-environment dictionaries exactly match the November 16 input environment. Historical preflight-package runtime metadata remains historical metadata; it is not substituted for the execution environment.
- Preserved the stated completion dates: RoBERTa typed November 14, replacement November 15, both restored November 16; both SecureBERT arms November 16. Earlier lost/incomplete attempts are excluded in the reports and ledger. No same-physical-machine or speed inference was made.

### All histories and training-window records

- Independently traversed all 48 epoch histories and all 2,352 train-window records, 588 per arm. Reconstructed each epoch's shuffled order using seed `42 + epoch * 1009`, checked all 49 windows against the 36-document training membership, and checked epoch/step/sample seed `42 + epoch * 100000 + step`, window index and coordinate offset.
- Reconstructed each training window's source relation count, omitted self-edge count, distinct selectable typed positive count and long-positive count from the locked raw dataset. These agree with every diagnostic record across both policies and both encoders. Each epoch contains 387 source relations, 386 selectable positives, one self-edge and one positive beyond distance 96. Each arm contains 96 zero-positive window observations.
- Recomputed every eligible/sampled count total and affected-window/document total from the histories and compared them with the 48 trajectory rows and `paired_results.json`. Checked nonnegative integer counts, collision subset constraints, sampled versus eligible bounds, positive preservation, per-epoch positive/pair totals, finite losses and validation values, and history SHA equality.
- For every replacement record checked safe eligible count, baseline negative budget, removals = refills + shortfall, refill capacity arithmetic and actual sampled negatives = baseline negatives - shortfall. All three sampled collision counts are zero. Unchanged windows retain their baseline digest. Typed records omit policy metadata.
- Independently compared paired batch-digest prefixes and first-intervention counterfactual hashes: RoBERTa has 367 identical windows before epoch 8/step 25/document 370/window 0; SecureBERT has 24 before epoch 1/step 25/document 421/window 0. Both first-intervention baseline hashes match the paired typed batch. This checks saved digest consistency; it does not reconstruct every historical sampled endpoint list.

| Arm | Sampled conflicts / groups | Eligible conflicts / groups | Removed / refilled / shortfall |
|---|---:|---:|---:|
| RoBERTa typed | 3 / 32,711 | 813 / 4,824,356 | 0 / 0 / 0 |
| RoBERTa replacement | 0 / 32,717 | 813 / 4,866,898 | 1 / 1 / 0 |
| SecureBERT typed | 11 / 32,866 | 837 / 5,601,364 | 0 / 0 / 0 |
| SecureBERT replacement | 0 / 32,864 | 880 / 5,627,392 | 4 / 4 / 0 |

### Selection, predictions, metrics and logits

- Recomputed epoch selection using the official `_is_better` rule, retaining the first epoch in an exact tie. Independently selected the best saved threshold-grid entry by relation F1, then primary entity F1, then lower threshold. Verified all 17 original thresholds from 0.10 through 0.90 and consistency among grid, history selection, summary, result JSON and selected checkpoint metadata. Selected epochs are 12, 11, 8 and 8; all thresholds are 0.90.
- Loaded and officially rescored all 20 saved validation prediction records, five per arm. Their document sets exactly match the five validation documents. Complete scorer output equals both saved metrics and reported JSON. Independently checked primary-entity/relation TP/FP/FN-to-P/R/F1 arithmetic and both paired deltas.
- Checked all saved coverage numerator/denominator arithmetic and result equality. Pair coverage is 37/49, 33/49, 30/49 and 27/49. Coverage here is validated from saved diagnostics and logit candidate totals; a new neural candidate-generation pass was not performed.
- Read every validation logit row: 22,458, 21,142, 34,136 and 40,494 respectively. Every existence logit and all 13 relation-type logits per row are finite; endpoint/window keys are unique, validation document membership is exact, row counts equal post-distance candidate counts and file SHA values match the verification report. No test output files were present in any reviewed run folder.

| Arm | Relation TP / FP / FN | Primary entity TP / FP / FN | Relation F1 (%) | Primary entity F1 (%) |
|---|---:|---:|---:|---:|
| RoBERTa typed | 25 / 123 / 24 | 103 / 309 / 17 | 25.381 | 38.722 |
| RoBERTa replacement | 23 / 99 / 26 | 99 / 297 / 21 | 26.901 | 38.372 |
| SecureBERT typed | 14 / 74 / 35 | 100 / 416 / 20 | 20.438 | 31.447 |
| SecureBERT replacement | 16 / 125 / 33 | 96 / 460 / 24 | 16.842 | 28.402 |

### Independently loaded tensors and delivery checks

- Loaded all four selected checkpoints: 212 model-state tensors per checkpoint, 848 total. Independently confirmed every selected tensor is finite, each checkpoint SHA matches its verification record, and epoch, validation selection, policy, dataset/config hashes and encoder package match the run metadata.
- Loaded both current SecureBERT epoch-12 recovery files. Independently checked their SHA values, source/policy/thread identity, complete history and selected epoch. Compared every one of the 212 embedded selected tensors per recovery exactly with the separately delivered selected checkpoint, plus equality of all non-tensor selected metadata. Also checked all 212 current model-state tensors in each recovery and every tensor in its 210 optimizer states for finiteness. Both optimizer distributions are exactly 210 states at step 156; the independently counted 156 accumulation updates have no empty-pair groups. These are fresh tensor checks.
- Did **not** load a RoBERTa recovery: neither restored archive contains it. Its embedded-best equality, optimizer counts/steps and recovery SHA are explicitly reused archived completion-date evidence. Both RoBERTa selected checkpoints were freshly loaded as described above.
- Independently hashed all four local selected-checkpoint ZIP archives and checked ZIP CRC plus all 21 SHA-manifest entries per archive. Every archived run entry exactly matches the reviewed local run bytes. Each archive contains the selected checkpoint and excludes full optimizer recovery. Delivery-map archive/checkpoint hashes and IDs agree with restoration records for RoBERTa and successful helper receipts for SecureBERT.
- Inspected the original failed HTTP401 output, successful retry receipt/output, recovery explanation and event ledger. The saved archive hash is `e0f610dcca5ac6c65e4276f30eb2318167c38810484fdfe9dd140ca95658c4df` before and after retry. The typed arm was confirmed saved before the replacement arm began. No unconfirmed failed write was promoted to success by the gate.

### Previously reviewed implementation and test evidence

Read the sampler implementation at `boundary_execution/journal/scsp/candidates.py:119`, policy metadata/full guard, matched wrapper forwarding, recovery identity equality and the focused policy/integration test sources. Directed keys include document and ordered boundaries; replacements use safe previously unsampled negatives, retain safe original draws and restore pool order; exhausted pools produce an explicit shortfall; zero positives/zero budget and default typed behavior retain their defined semantics. Cross-policy recovery identity mismatch is rejected, and both wrapper paths have explicit focused coverage. No ambiguity requiring renewed engineering work was found.

Reused the recorded November 14 result of 290 tests plus 100 subtests and the November 16 three-test persistence success log after inspecting their relevant source coverage. These are existing results, not tests freshly executed by this reviewer. No unchanged full suite or fixture model forward was repeated.

## Declined-to-judge scope and reasons

- Statistical significance, generalization, causal encoder ranking or a default-policy recommendation: one seed and five validation documents were used for both checkpoint and threshold selection. All reported declines remain visible. Human adjudication is pending.
- Full/test performance, gold-label correctness or completion of human adjudication: outside the protocol; no such evaluation or adjudication was performed.
- Fresh RoBERTa optimizer/recovery consistency or resumability from delivered archives: those tensors are absent. Historical verification evidence is retained, while selected weights are freshly checked. The delivered archives alone are not complete training-resume packages.
- New forward equivalence, independent decoding of every saved logit or re-scoring every historical threshold from reconstructed inferences: forbidden or outside this read-only scope. Selected saved predictions were officially rescored; selection over the recorded grid/history was independently recomputed.
- Remote service storage internals or independent download of newly uploaded SecureBERT archives: this review verifies local bytes, complete archive manifests and recorded successful service receipts. It does not make an additional remote readback claim.
- Final Word rendering, the not-yet-built combined evidence ZIP, its subsequent upload and any later PR commit/push: the primary agent owns these delivery steps. This review covers the Markdown/JSON content, packaging code and four already saved checkpoint archives; it must not be described as an independent review of artifacts created afterward.
- Physical-machine identity, elapsed-time comparability, and unchanged historical preflight/overfit/smoke execution: only their documented provenance and applicable recorded controls are reused; no new execution or speed claim is inferred.

The verified scientific content is ready for reporting with these limitations. Subsequent packaging must preserve this review and the delivery/provenance distinctions, and the PR must remain draft.
