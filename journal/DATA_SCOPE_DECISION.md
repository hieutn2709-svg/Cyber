# Gate A data-scope decision: primary vs auxiliary entity labels

## Status

Accepted implementation correction for `journal/scsp-q2-gate-a-spanpair`.

## Why the original ten-type assumption was insufficient

The controlled clean-window dataset used by the fixed 52-document manifest has
15 annotated entity types. Ten are the journal **core entity types** used for the
primary Entity F1:

- attack-pattern
- campaign
- domain-name
- identity
- indicator
- intrusion-set
- location
- malware
- tool
- vulnerability

Five additional annotated types are present:

- file-paths
- sha256s
- tactic
- threat-actor
- url

The hash-verified clean-window corpus contains 564 relation instances. Three of
those relations have at least one endpoint outside the ten-type core inventory:

- two relations in the fold-1 test documents use `tactic` as an endpoint;
- one relation in the fold-5 test documents uses `threat-actor` as an endpoint.

A model restricted to the ten core entity types would therefore make those
relations structurally impossible to recover while the historical relation
scope still counts them. That would create an avoidable evaluation-scope change
and contaminate comparisons with the archived V10/V13 relation results.

## Decision

Gate A will use **15 trainable entity labels plus NONE** for span typing. The ten
core labels remain the only labels counted in the primary core Entity F1. The
five additional labels are designated **auxiliary endpoint labels**.

Auxiliary labels are retained for three reasons:

1. preserve annotated relation endpoint coverage;
2. avoid creating structural false negatives by construction;
3. keep the 564-relation clean-window evaluation scope aligned with the fixed
   manifest and archived comparator artifacts.

They must not be presented as evidence that the journal system has strong
entity extraction performance for those five sparse types. Per-type support and
metrics must remain visible.

## Relation reporting

The journal evaluation should report both:

- **all-evaluable relation F1** over the 564 clean-window relations for direct
  same-scope comparison; and
- **core-to-core relation F1** over the 561 relations whose endpoints are both
  among the ten primary core entity types.

The difference between these scopes must be documented rather than hidden.

## STIX interpretation

This decision is an extraction-inventory choice, not a claim of full STIX 2.1
coverage. The later schema/task profile remains separately versioned. Auxiliary
labels are not automatically promoted to the primary STIX output inventory and
must not be used to overstate standards compliance.
