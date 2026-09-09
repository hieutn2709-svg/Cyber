# Gate B STIX-oriented task profile

These files support the deterministic **Gate B Hard Profile** decoder. They are versioned experiment data, not a replacement for the OASIS STIX validator and not a claim that every triple outside this project profile is invalid STIX.

## Files

- `stix_2_1_normative_notes.json` records the OASIS STIX 2.1 source, terminology, and experiment guardrails.
- `relation_canonicalization_v1.json` maps the 13 project relation labels to canonical relationship labels used only for compatibility lookup/reporting. Training labels and emitted project labels are unchanged.
- `task_relationship_profile_v1.json` lists the resolved/unresolved endpoint types and 55 explicit canonical triples allowed by Hard Profile v1.

## Resolution policy

`used-in` is relation-label unresolved/pass-through. `file-paths`, `sha256s`, and `tactic` are endpoint-unresolved/pass-through. No global mapping is invented for those labels.

For resolved endpoints, a project relation label is canonicalized and any required endpoint swap is applied only to the compatibility lookup. A compatible logit is preserved; a resolved incompatible logit is masked; an unresolved relation class remains available. Relation existence, entity predictions, candidate generation, checkpoint selection, and model weights are unchanged.

## Provenance and versioning

The profile semantics are based on the OASIS STIX 2.1 specification, especially Appendix B Relationship Summary, together with training-side audit evidence. Validation and test labels must never create or revise profile rules. Any future semantic change requires a new versioned file rather than editing v1 in place after observing evaluation outcomes.
