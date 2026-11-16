# E2 readiness audit — 2026-10-25

Read-only fold1 train audit: 36 documents and49 windows per encoder, disjoint train/validation/test document sets, authenticated dataset/split manifests. Both encoders contain949 gold-span instances and387 relation instances; their character/annotation identities match. No validation/test metrics or new forward/training were run.

Gold-only candidate proxy (not actual dynamic training batches):386 selectable positive pairs, one same-boundary/self relation omitted, one long positive preserved; negative pool12949 RoBERTa/13613 SecureBERT; sampled negatives2312 per seed in this proxy. The current sampler's negative budget is six times selectable positives, or at most six for zero-positive windows.

Source doc381 explicitly annotates APT38 entity Fqsez1Djn2 (chars416:421) targets itself. build_training_pairs skips same-boundary pairs. This is a source/protocol discrepancy requiring human review, not an authorized gold correction; source remains unchanged.

A minimal synthetic example using unchanged inject_gold_spans/build_training_pairs reproduces both existence targets0 and1 for the same directed boundary pair, where only endpoint type differs. Reproduced for seeds42/43/44. Relation features omit entity labels, so this establishes a possible supervision conflict. Actual frequency in historical dynamic training batches and impact on F1 are unmeasured; do not infer causation for the E1 result.

Independent review reproduced counts, source annotation, synthetic examples, input hashes and architecture reasoning. Next diagnostic proposal: measure these collisions in real train candidate batches before prioritizing wrong-type/same-boundary hard negatives. Separate any collision-policy change from the E2 sampler ablation; no such implementation or training is claimed. Human annotation review remains pending.

Full Vietnamese report, reproducible analytical script, per-window counts and pending source-review case are in the separate CTI_Q2_Kiem_tra_E2_20261025.zip handoff. The research PR remains draft.
