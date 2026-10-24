"""Gold-free offline top-k typed alternatives over frozen D0 boundary pairs.

Scores are boundary existence scores, not calibrated typed-triple probabilities.
No model, annotation, threshold fitting or schema compatibility is used here.
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import product


def span_key(span):
    return (span['document_id'], span['start'], span['end'], span['label'])


def relation_key(relation):
    s, t = relation['source'], relation['target']
    return (*span_key(s), relation['label'], t['start'], t['end'], t['label'])


def decode_snapshots(snapshots, pair_rows, relation_types, *, k, threshold=0.9):
    """Return sorted prediction-only dictionaries; reject incomplete pair input.

    Snapshot boundaries are window-local. Exported pair rows are global.
    k is 1 (baseline replay) or 2 (multi-hypothesis ablation).
    """
    if k not in (1, 2) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid k or threshold')
    if not relation_types or len(set(relation_types)) != len(relation_types):
        raise ValueError('relation inventory must be nonempty and unique')
    spans_by_doc = defaultdict(dict)
    relations_by_doc = defaultdict(dict)
    expected = {}
    windows = set()
    inventory = None
    for snapshot in snapshots:
        doc = str(snapshot['document_id'])
        window = (doc, snapshot['window_index'])
        if window in windows:
            raise ValueError('duplicate snapshot window')
        windows.add(window)
        classes = snapshot['classes']
        if (classes[0] != 'NONE' or len(classes) < k + 1
                or len(set(classes)) != len(classes)):
            raise ValueError('invalid entity inventory')
        if inventory is not None and classes != inventory:
            raise ValueError('entity inventory changed across windows')
        inventory = classes
        spans_by_doc[doc]  # Preserve documents even with no retained spans.
        offset = snapshot['global_token_offset']
        kept = {}
        boundaries = set()
        for i, span in enumerate(snapshot['spans']):
            if span['selection_reason'] != 'retained':
                continue
            probs = span['probabilities']
            if (len(probs) != len(classes) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probs)
                    or abs(sum(probs) - 1) > 1e-6 or sum(probs[1:]) <= 0):
                raise ValueError('invalid entity posterior')
            ranked = sorted(range(1, len(classes)), key=lambda j: (-probs[j], j))
            if classes[ranked[0]] != span['label']:
                raise ValueError('posterior top1 differs from retained label')
            score = span['entity_score']
            if not math.isfinite(score) or not 0 <= score <= 1 or abs(score - (1-probs[0])) > 1e-6:
                raise ValueError('invalid retained entity score')
            start, end = span['start'] + offset, span['end'] + offset
            if start < 0 or end < start or (start, end) in boundaries:
                raise ValueError('invalid or duplicate retained boundary')
            boundaries.add((start, end))
            alternatives = [dict(document_id=doc,start=start,end=end,label=classes[j],
                                 entity_score=score,proposal_source='predicted') for j in ranked[:k]]
            kept[i] = alternatives
            for a in alternatives:
                key = span_key(a)
                old = spans_by_doc[doc].get(key)
                if old is None or score > old['entity_score']:
                    spans_by_doc[doc][key] = a
        for pair in snapshot['generated_pairs']:
            i, j = pair['source_index'], pair['target_index']
            if i not in kept or j not in kept or i == j:
                raise ValueError('pair endpoint missing or same boundary')
            a, b = kept[i][0], kept[j][0]
            distance = max(0, max(a['start'], b['start']) - min(a['end'], b['end']) - 1)
            if pair['token_distance'] != distance:
                raise ValueError('snapshot pair distance mismatch')
            key = (*window, a['start'], a['end'], a['label'], b['start'], b['end'], b['label'])
            if key in expected:
                raise ValueError('duplicate generated pair')
            expected[key] = (kept[i], kept[j], distance)
    for row in pair_rows:
        doc = str(row['document_id'])
        a, b = row['source'], row['target']
        key = (doc,row['window_index'],a['start'],a['end'],a['label'],b['start'],b['end'],b['label'])
        if key not in expected:
            raise ValueError('unexpected or duplicate exported pair')
        sources, targets, distance = expected.pop(key)
        if row['token_distance'] != distance:
            raise ValueError('exported pair distance mismatch')
        logits = row['relation_type_logits']
        value = row['existence_logit']
        if len(logits) != len(relation_types) or not all(math.isfinite(v) for v in [value, *logits]):
            raise ValueError('invalid relation logits')
        if value >= 0:
            probability = 1 / (1 + math.exp(-value))
        else:
            z = math.exp(value)
            probability = z / (1 + z)
        if probability < threshold:
            continue
        label = relation_types[max(range(len(logits)), key=lambda j: (logits[j], -j))]
        for source, target in product(sources, targets):
            relation = dict(source=source,target=target,label=label,relation_score=probability)
            rk = relation_key(relation)
            old = relations_by_doc[doc].get(rk)
            if old is None or probability > old['relation_score']:
                relations_by_doc[doc][rk] = relation
    if expected:
        raise ValueError('missing exported pairs')
    return [dict(document_id=doc,
                 predicted_spans=[v for _,v in sorted(spans_by_doc[doc].items())],
                 predicted_relations=[v for _,v in sorted(relations_by_doc[doc].items())])
            for doc in sorted(spans_by_doc)]
