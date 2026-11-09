# E2 fixed-checkpoint train-mode snapshot — 2026-10-26

Both seed42 selected checkpoints (RoBERTa epoch12, SecureBERT epoch8) were restored
and SHA256-verified. Each was observed on the locked fold1 TRAIN partition only:
36 documents,49 windows,949 gold span instances,387 source relation instances.
This is a new train-mode/dropout snapshot at fixed weights under no_grad, not a
historical training replay or a new optimizer trajectory. No held-out forward,
validation/test scoring, backward, optimizer step, gold edit or sampler change.

| Window-summed counts | RoBERTa | SecureBERT |
|---|---:|---:|
| Eligible positive pairs |386|386|
| Eligible negative pairs |227739|405161|
| Eligible directed boundary groups |225589|393481|
| Eligible collision groups |18|71|
| Positive/negative instances within eligible collisions |18/18|71/75|
| Windows with eligible collisions |9/49|28/49|
| Documents with eligible collisions |9/36|27/36|
| Sampled positive pairs |386|386|
| Sampled negative pairs |2328|2358|
| Sampled directed boundary groups |2712|2743|
| Sampled collision groups |1|0|
| Positive/negative instances within sampled collisions |1/1|0/0|
| Windows/documents with sampled collisions |1/49;1/36|0/49;0/36|

A collision means the same directed endpoint boundaries appear with both binary
existence labels; endpoint types distinguish the typed pairs. Group totals are
summed over windows, not deduplicated corpus relations. Each model still omits one
source self-relation and retains one positive beyond distance96, as in the earlier
structural audit. Actual negative pools/budgets differ from the gold-only proxy.

All98 logging-off/on comparisons preserve sampled batches/order, loss tensors,
result metadata and post-forward Torch RNG exactly. Python/NumPy RNG and model
state tensors are unchanged; checkpoint loading was strict and tensor-verified.
A separate stdlib enumerator, importing no project sampler, reconstructed every
eligible count, sampled count, sampled-batch digest and aggregate for98 windows.
Saved sampler inputs enable independent reproduction.

Snapshot settings: unchanged width6/cap128/entity0.05/distance96/negative-ratio6;
original training loss config; initial RNG seed42, epoch1-style shuffle and
sampling seed42+100000+step. These choices do not recreate historical RNG states.
CPU4 threads, PyTorch2.14.1+cpu, NumPy2.3.5, Transformers5.18.0, SDPA.
Recorded paired-pass time ~52.69/52.79s and process peak RSS ~1556.6/1553.2MiB
include diagnostic work and checkpoint/model loading respectively. No backward or
optimizer-state allocation occurs; these are not full-training cost estimates or
randomized logging-overhead benchmarks.

Three committed regression tests now exercise both matched-wrapper forwarding
paths (preflight/model execution mocked) and diagnostics-bearing tiny-model
save/restore/next-epoch exact parity with changed observation identities rejected.
Full suite272/272 passed; production training/sampler code unchanged.

Interpretation: wrong-type variants can conflict in eligible pools, while random
sampling may rarely expose those conflicts in this particular snapshot. Zero
sampled SecureBERT collisions in one pass does not establish absence throughout
training; prioritizing these variants as hard negatives could change exposure.
No causal explanation for E1 F1 or encoder ranking is inferred. Next: pre-register
and run trajectory measurement under a separate run identity, then evaluate any
collision-policy correction separately from hard-negative mining. Human label
adjudication remains pending. Keep research PR draft.

Independent review found no Critical/Important/Minor issues and closed the previous
coverage minor. Reviewer reproduced272 tests and independently re-enumerated98
saved candidate pools with all counts/digests matching; verified checkpoint,
configuration, tensor digests, train partition and schedule. Encoder forward passes
were not independently rerun. Full review limitations are preserved in the evidence
bundle; no external trust-root or independent real-model forward replication claim.
