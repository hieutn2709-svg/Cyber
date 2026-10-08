# SecureBERT matched-70 dev execution approval

Status: APPROVED by the user's “ok hay chay buoc tiep theo cho toi” after the
completed RoBERTa baseline report and explicit request to authorize SecureBERT
dev for 12 epochs. The user subsequently reiterated “Tiếp tục tác vụ này”.

Scope: SecureBERT dev only, Fold 1, run seed 42, split seed 11800, the existing
matched70 corpus lock, pinned SecureBERT model/tokenizer, frozen downstream
architecture/hyperparameters and 12-epoch budget. Start from pinned pretrained
weights, not from the smoke checkpoint. Preflight, overfit and smoke remain
mandatory prerequisites. Select checkpoint/threshold on validation only using
the existing relation-first rule. Full/test remains unavailable.

Implementation: remove the SecureBERT-specific dev prohibition, retain known
package and mode guards, run the full regression suite, execute dev with atomic
epoch recovery, then compare both selected validation checkpoints and report
limitations (one fold/seed; CPU only; no test). Save result artifacts and update
existing draft PR 5 without merging.

Recovery ruling: the workspace reverted to 7d45e1e and lost runtime artifacts.
Restore the exact published tree 728c3a3641dfe7690c481486e9250aff61c261f0 from
commit 270af0ee98e968f741b23e3d4864d8f904153ba7 and the version-1 archived results;
verify all internal SHA-256 entries. Do not rerun the completed RoBERTa baseline
or alter original run provenance. Restore only missing pinned model cache files.
