# CTI Q2 — E1 readiness, 2026-10-24

The completed D0 matrix is unchanged. A separate offline audit independently regenerated retained span selection, ordered pairs and interval distances from the saved snapshots, then reconciled every exported logit-row key after local-to-global conversion: 12 configurations, 60 window observations, 292,272 pairs, all exact. No new forward pass or training was run.

Architecture inspection found that Gate A relation features do not consume entity labels or type posteriors. A factorized top-2 MAP choice returns the original top-1 pair for all 292,272 saved pairs. Existing Gate C changes relation-type decoding but preserves the typed endpoints; it is not an endpoint-type correction.

For the selected 22 persistent-error cases at cap128, both gold typed endpoints are covered in 51/132 top-1 observations versus 81/132 retained-top-2 observations. This is gold-informed retrospective endpoint coverage, not 30 recovered true positives, deployable recall, or an F1 improvement. Repeated observations are not independent.

Two model-blinded reviewer packets are prepared (22 source-verified cases each, full document text, blank structured/printable forms), with a separate coordinator map. Human reviews completed: zero. Annotation changes: zero. No packets were sent. The sample is selected for persistent errors and cannot estimate corpus-wide annotation error rates.

A proposed E1-MULTI ablation would preserve cap128 boundaries/existence scores/relation logits and emit top-2 x top-2 typed alternatives, reporting all resulting FP, strict P/R/F1, output multiplicity and decode cost. It is not implemented or evaluated. Independent label review and final decoder design remain pending; new training, test/full evaluation and external evaluation have not occurred.

The 2026-10-24 user handoff contains the reproducible offline analysis, detailed Vietnamese report, proposed protocol and reviewer packets. This commit records readiness only; the research PR remains draft.
