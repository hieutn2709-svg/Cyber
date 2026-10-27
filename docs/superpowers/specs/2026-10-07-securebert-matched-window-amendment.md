# Approved matched-window amendment for SecureBERT v1

Status: **APPROVED for implementation and RoBERTa re-baseline**.

User approved the 70-window proposal with “ok” and reiterated continuation on
2026-10-07. Full/test remains prohibited.

## Reason and approval boundary

The approved 67-window RoBERTa corpus has now been reconstructed with exact
semantic parity. Native SecureBERT encoding exceeds the 512-token capacity in
three of those windows. Automatic truncation or increasing model capacity would
violate the frozen comparison. The approved comparison design, section 4.4, and
the dataset-gate plan, Task 4, require separate approval for a matched re-baseline.
The original corpus and its hashes remain immutable.

## Concrete shared character windows

All three cases belong to document `366`. Indices are zero-based; character
ranges are half-open Python coordinates. Token counts below include both specials.

| Original window | Original character range | SecureBERT tokens | Proposed ranges | RoBERTa lengths | SecureBERT lengths |
| --- | --- | ---: | --- | --- | --- |
| 4 | [7572, 9808) | 513 | [7572, 9792), [9792, 9808) | 511, 7 | 512, 6 |
| 7 | [14129, 15007) | 516 | [14129, 14966), [14966, 15007) | 489, 26 | 494, 25 |
| 8 | [15008, 15904) | 520 | [15008, 15863), [15863, 15904) | 487, 28 | 495, 28 |

The three proposed cuts are whitespace boundaries, split no retained entity and
separate no annotated relation endpoints. Capacity was checked with both pinned
tokenizers. The proposal yields 70 common windows for the same 52 documents.
The expected annotation totals remain 1,353 primary, 28 auxiliary and 564 relation
occurrences; these totals are acceptance targets, not a claim that the amended
datasets already exist.

Rule: for each over-capacity logical window, enumerate boundaries immediately after whitespace (as in the approved audit);
reject cuts inside a retained entity or between the endpoints of a retained
relation; require both resulting parts to fit both tokenizers; select the largest
valid character cut. If no such cut exists, fail closed and review the protocol.
The current three cases require only one cut each. No validation/test prediction
or metric was consulted to select these boundaries.

## Proposed implementation and acceptance

1. Freeze a new, separately named common character-window manifest linked to the
   verified 67-window parity ancestor. Preserve original document order and record
   original window ID plus subwindow ID. Do not overwrite original data or runs.
2. Encode the same 70 character slices independently with each pinned package,
   using identical special-token and no-truncation rules. Record native token
   bounds separately from common character bounds; never feed reference token
   indices to SecureBERT candidate conversion. Existing reference-token global
   bounds cannot be reused as native SecureBERT bounds.
3. Re-audit every window, exact document/character/entity/relation membership,
   hashes, exclusion policy, local/native coordinate conversion and all counts.
   The proposed slices were capacity-checked; complete amended artifact generation
   and semantic validation are required before training.
4. Re-run B3 RoBERTa under this new common-window protocol before interpreting
   B4 SecureBERT. Reusing an old 67-window B3 score as the matched baseline is
   disallowed. Both packages keep the frozen Fold 1, split seed 11800, seed 42,
   schema none, downstream architecture, 12-epoch budget and validation rules.
5. Follow the approved runtime preflight, one-window overfit and smoke gates.
   This amendment does not authorize full/test evaluation or waive later dev
   execution checkpoints. No metrics are claimed at this stage.

## Evidence and review decisions

- Verified local builder commit: `8cbdf7909c09751c9e873f40d38afe8a980d6970`.
- Equivalent published builder commit: `dc5500235539e5728c32074c8673a3e8d55c4176`
  (identical complete Git tree `e2064f860c6d8b3841abe60dc9d796db39ee1c61`).
- Frozen input SHA-256:
  `190d3136edba33d89ee58f533e2d12cc6cac2842323e3168f6e3e0e71af72c48`.
- Passed parent manifest SHA-256:
  `7244756f44e40b7ed10d1217652613b3f7c508b1b08b1d83540828561c329797`.
- Full capacity and alignment audit: `journal/securebert_v1_capacity_audit.json`.
- Regression verification: 214 tests passed; no training/test evaluation occurred.
- Independent code review of the parity fixes found no blocking findings.
  Its exclusions from review were independently reproduced corpus digest and clean
  provenance (checked by the executor), upcoming SecureBERT/runtime behavior
  (capacity audited here, runtime still blocked), training/test behavior (unrun),
  and pre-existing generic JSON diagnostics/exclusion-coordinate polish (deferred).

The executor restored only behavior proven by the frozen corpus, with explicit
audits and exact parity. The subsequent explicit user approval authorizes the 70-window protocol
and RoBERTa rerun, subject to the runtime and diagnostic gates above.
