# E2 training-pair diagnostics — 2026-10-25

Implemented default-off `--train-pair-diagnostics` in Gate A and both matched
entrypoints. The sampler summarizes its existing eligible lists and actual
sampled batch; it does not reconstruct a different candidate pool or change
sampling. Per-window records are retained in training history, with document,
window, epoch, step, sampling seed and global token offset. Run config and recovery
identity record enabled observation; toggling it on resume is rejected by the
existing strict identity guard.

Counts distinguish eligible versus sampled directed boundary groups with both
existence labels. Denominators, collision-positive/negative pair counts, omitted
source self-relations, retained long positives and a sampled-batch SHA256 are
included. Examples are bounded to five groups and ten type combinations per label.
No source text is exported. Disabled history and selection behavior are unchanged.

Verification: baseline repository 259 tests pass; changed repository 269 tests
pass in the restored CPU environment. Ten new tests cover adversarial sampler
inputs and real differentiable tiny-model loss/gradient/RNG/AdamW parity, including
contradictory labels that reach the loss, two-window gradient accumulation, overfit
history and CLI opt-in. Exact equality is checked, not numerical tolerance.
PyTorch 2.14.1+cpu and NumPy 2.3.5; this is engineering evidence on synthetic fixtures,
not pretrained-model or CUDA parity evidence.

A separate synthetic exported example intentionally retains the fixture budget
of two negatives: one eligible collision group (one positive/three negative type
variants) but zero sampled collision groups for seeds42/43/44. This demonstrates
why eligible and sampled collision counts must be reported separately. These are
not CTI corpus measurements and must not be used as historical prevalence.

No pretrained training, CTI corpus forward, validation/test scoring, sampling-policy
change or gold edit was performed. Source self-edge review and human adjudication
remain pending. Next scientific step: collect diagnostics in a separately named,
protocol-fixed training run; analyze rates before deciding on a collision-policy
correction, then evaluate hard-negative sampling as a separate ablation. Existing
completed run outputs and recovery identities must not be overwritten or relabeled.

Independent review: no Critical/Important findings. Reproduced16 focused tests,
240 base-sampler comparisons, multi-type/reverse/truncation cases, enabled recovery
and exact resumed next-epoch parity, and rejection of changed observation identity.
Deferred minor: committed tests do not cover matched-wrapper forwarding or
observation-specific recovery metadata end-to-end; wiring was inspected and recovery
was independently probed. Real-package execution and production-scale overhead
remain unmeasured. Review covers this delta only.
