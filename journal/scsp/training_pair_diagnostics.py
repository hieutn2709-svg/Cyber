"""Observational binary-label collisions in actual sampler lists; no RNG/tensors."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json


def _boundary_key(pair):
    return (pair.document_id, pair.source.start, pair.source.end,
            pair.target.start, pair.target.end)


def _groups(positives, negatives):
    groups = defaultdict(lambda: [[], []])
    for label, pairs in ((0, positives), (1, negatives)):
        for pair in pairs:
            groups[_boundary_key(pair)][label].append(pair)
    return groups


def _counts(groups):
    collisions = [rows for rows in groups.values() if rows[0] and rows[1]]
    return {
        'positive_pairs': sum(len(rows[0]) for rows in groups.values()),
        'negative_pairs': sum(len(rows[1]) for rows in groups.values()),
        'boundary_groups': len(groups),
        'collision_groups': len(collisions),
        'collision_positive_pairs': sum(len(rows[0]) for rows in collisions),
        'collision_negative_pairs': sum(len(rows[1]) for rows in collisions),
    }


def summarize_training_pairs(positives, eligible_negatives, batch, gold_relations,
                             *, max_token_distance):
    """JSON-safe window counts, not unique corpus relations or historical rates.

    Lists come directly from the sampler before sampling. Direction and document
    identity are preserved. The runner adds window identity to local coordinates.
    Examples are limited to five groups and ten type combinations per label.
    """
    eligible = _groups(positives, eligible_negatives)
    sampled = _groups(
        [p for p, positive in zip(batch.pairs, batch.positive_mask) if positive],
        [p for p, positive in zip(batch.pairs, batch.positive_mask) if not positive])
    collision_keys = sorted(key for key, rows in eligible.items() if all(rows))
    examples = []
    for key in collision_keys[:5]:
        positive, negative = eligible[key]
        sampled_rows = sampled.get(key, ((), ()))
        example = {
            'boundary_key': list(key),
            'eligible_positive_pairs': len(positive),
            'eligible_negative_pairs': len(negative),
            'sampled_positive_pairs': len(sampled_rows[0]),
            'sampled_negative_pairs': len(sampled_rows[1]),
        }
        for label, pairs in (('positive', positive), ('negative', negative)):
            types = sorted({(p.source.label, p.target.label) for p in pairs})
            example[f'eligible_{label}_type_pairs'] = [list(t) for t in types[:10]]
            example[f'eligible_{label}_type_pairs_omitted'] = max(0, len(types) - 10)
        examples.append(example)
    payload = [
        {'pair': asdict(pair), 'positive': positive, 'relation_label': label}
        for pair, positive, label in zip(batch.pairs, batch.positive_mask, batch.relation_labels)]
    return {
        'schema_version': 1,
        'scope': 'one_training_window_sampler_call',
        'coordinate_system': 'window_local_inclusive',
        'eligible': _counts(eligible), 'sampled': _counts(sampled),
        'gold_relation_instances': len(gold_relations),
        'gold_self_relation_instances': sum(
            (r.source.start, r.source.end) == (r.target.start, r.target.end) for r in gold_relations),
        'eligible_long_positive_pairs': sum(p.token_distance > max_token_distance for p in positives),
        'collision_examples': examples,
        'collision_examples_omitted': max(0, len(collision_keys) - 5),
        'sampled_batch_sha256': hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
    }
