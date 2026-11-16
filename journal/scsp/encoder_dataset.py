"""Tokenizer-neutral annotations and fail-closed encoder reconstruction.

This module does not load models, splits, checkpoints, or prediction metrics.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SourceEntity:
    entity_id: str
    char_start: int
    char_end: int
    label: str

    @property
    def key(self):
        return self.entity_id, self.char_start, self.char_end, self.label


@dataclass(frozen=True, slots=True)
class SourceRelation:
    source_id: str
    target_id: str
    label: str


@dataclass(frozen=True, slots=True)
class UnlabeledRegion:
    entity_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class SourceDocument:
    doc_seq_index: int
    doc_id: str
    text: str
    entities: tuple[SourceEntity, ...]
    relations: tuple[SourceRelation, ...]
    unlabeled_regions: tuple[UnlabeledRegion, ...] = ()


def normalize_source_text(text: str) -> str:
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _identifier(value, field):
    if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value).strip():
        raise ValueError(f'invalid {field}')
    return str(value)


def load_label_studio_documents(path, *, excluded_unlabeled_regions=frozenset()):
    consumed_exclusions = set()
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('source must be a non-empty document list')
    documents, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'document row {index} must be an object')
        doc_id = _identifier(row.get('id'), 'document id')
        if doc_id in seen:
            raise ValueError(f'duplicate document id {doc_id}')
        seen.add(doc_id)
        text = row.get('data', {}).get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'{doc_id}: non-empty source text required')
        normalized = normalize_source_text(text)
        accepted = [a for a in row.get('annotations', [])
                    if not a.get('was_cancelled', False) and a.get('result')]
        if len(accepted) != 1:
            raise ValueError(f'{doc_id}: exactly one accepted annotation required')
        entities, relations, ids, unlabeled = [], [], set(), []
        for record in accepted[0]['result']:
            kind = record.get('type')
            if kind == 'labels':
                eid = _identifier(record.get('id'), 'entity_id')
                if eid in ids:
                    raise ValueError(f'{doc_id}: duplicate entity_id {eid}')
                ids.add(eid)
                value = record.get('value', {})
                a, b = value.get('start'), value.get('end')
                if (type(a) is not int or type(b) is not int or
                        not 0 <= a < b <= len(text) or not text[a:b].strip()):
                    raise ValueError(f'{doc_id} {eid}: invalid coordinates')
                if (len(normalized) != len(text) or normalized[a:b] != text[a:b]
                        or value.get('text', text[a:b]) != text[a:b]):
                    raise ValueError(f'{doc_id} {eid}: normalization/surface invalidates coordinates')
                labels = value.get('labels')
                if (doc_id, eid) in excluded_unlabeled_regions:
                    if labels:
                        raise ValueError(f'{doc_id} {eid}: cannot exclude a labeled entity')
                    consumed_exclusions.add((doc_id,eid))
                    unlabeled.append(UnlabeledRegion(eid,a,b))
                    continue
                if not isinstance(labels, list) or len(labels) != 1:
                    raise ValueError(f'{doc_id} {eid}: ambiguous labels')
                label = _identifier(labels[0], 'entity label')
                entities.append(SourceEntity(eid, a, b, 'file-paths' if label == 'file_paths' else label))
            elif kind == 'relation':
                labels = record.get('labels')
                if not isinstance(labels, list) or len(labels) != 1:
                    raise ValueError(f'{doc_id}: ambiguous relation label')
                if record.get('direction', 'right') not in ('right', 'left'):
                    raise ValueError(f'{doc_id}: unsupported relation direction')
                relations.append(SourceRelation(_identifier(record.get('from_id'), 'source_id'),
                    _identifier(record.get('to_id'), 'target_id'), _identifier(labels[0], 'relation label')))
            else:
                raise ValueError(f'{doc_id}: unsupported annotation type')
        if normalized != text:
            # Even unannotated repairs can shift later tokenizer boundaries.
            raise ValueError(f'{doc_id}: normalization changes source coordinates')
        for relation in relations:
            if relation.source_id in {e.entity_id for e in unlabeled} or relation.target_id in {e.entity_id for e in unlabeled}:
                raise ValueError(f'{doc_id}: excluded region is a relation endpoint')
            if relation.source_id not in ids or relation.target_id not in ids:
                raise ValueError(f'{doc_id}: unknown relation endpoint')
        documents.append(SourceDocument(index, doc_id, text, tuple(entities), tuple(relations), tuple(unlabeled)))
    if consumed_exclusions != set(excluded_unlabeled_regions):
        raise ValueError('unused source exclusion entries')
    return tuple(documents)


@dataclass(frozen=True, slots=True)
class LogicalWindow:
    doc_seq_index: int
    doc_id: str
    window_index: int
    token_start_global: int
    token_end_global: int


def load_logical_window_contract(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('frozen windows must be a non-empty list')
    windows, seen, identities = [], set(), {}
    for row in rows:
        fields = ('doc_seq_index', 'window_index', 'token_start_global', 'token_end_global')
        if any(type(row.get(k)) is not int or row[k] < 0 for k in fields):
            raise ValueError('invalid logical window coordinates')
        doc_id = _identifier(row.get('doc_id'), 'document id')
        w = LogicalWindow(row['doc_seq_index'], doc_id, row['window_index'],
                          row['token_start_global'], row['token_end_global'])
        if w.token_start_global >= w.token_end_global:
            raise ValueError('empty logical window')
        key = (w.doc_seq_index, w.window_index)
        if key in seen:
            raise ValueError('duplicate logical window')
        seen.add(key)
        if identities.setdefault(w.doc_seq_index, doc_id) != doc_id:
            raise ValueError('inconsistent document identity')
        windows.append(w)
    return tuple(windows)


def canonical_window_payload(rows):
    # Mapping order is formatting; every sequence order is semantic.
    return [{**row, 'entity_spans': [
        {k: v for k, v in entity.items() if k != 'text'}
        for entity in row['entity_spans']]} for row in rows]


def semantic_sha256(rows):
    return hashlib.sha256(json.dumps(canonical_window_payload(rows), ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def compare_reconstruction(expected, actual):
    def compare(a, b, path):
        if type(a) is not type(b):
            raise ValueError(f'semantic parity mismatch at {path}: type')
        if isinstance(a, dict):
            if set(a) != set(b):
                raise ValueError(f'semantic parity mismatch at {path}: keys')
            for key in a:
                compare(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f'semantic parity mismatch at {path}: length {len(a)} != {len(b)}')
            for i, (left, right) in enumerate(zip(a, b)):
                compare(left, right, f'{path}[{i}]')
        elif a != b:
            # Values intentionally omitted: diagnostics must not leak source text.
            raise ValueError(f'semantic parity mismatch at {path}: value')
    left, right = canonical_window_payload(expected), canonical_window_payload(actual)
    if len(left) != len(right):
        raise ValueError('semantic parity mismatch: row count')
    for i, (a, b) in enumerate(zip(left, right)):
        compare(a, b, f'row {i}')


@dataclass(frozen=True, slots=True)
class AlignedEntity:
    entity_id: str
    label: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True, slots=True)
class AlignmentExclusion:
    doc_id: str
    window_index: int
    entity_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class WindowProjection:
    char_start: int
    char_end: int
    entities: tuple[SourceEntity, ...]
    exclusions: tuple[AlignmentExclusion, ...]


def _overlaps(a, b, c, d):
    return max(a, c) < min(b, d)


def _content_bounds(entity, document_text):
    """Trim only proven whitespace for coverage checks; never rewrite gold spans."""
    a, b = entity.char_start, entity.char_end
    if document_text is not None:
        if not 0 <= a < b <= len(document_text):
            raise ValueError('invalid source coordinates for alignment')
        while a < b and document_text[a].isspace():
            a += 1
        while b > a and document_text[b-1].isspace():
            b -= 1
        if a == b:
            raise ValueError('empty entity content')
    return a, b


def align_entity(*, doc_id, window_index, entity, offsets, special_tokens_mask,
                 window_char_start, window_char_end, offsets_are_local=False,
                 document_text=None):
    context = f'{doc_id} window {window_index} entity {entity.entity_id}'
    if len(offsets) != len(special_tokens_mask):
        raise ValueError(f'{context}: offset/mask length mismatch')
    content_start, content_end = _content_bounds(entity, document_text)
    if not window_char_start <= content_start < content_end <= window_char_end:
        raise ValueError(f'{context}: truncated entity')
    shift = window_char_start if offsets_are_local else 0
    absolute = [(a+shift, b+shift) for a, b in offsets]
    indices = [i for i, (a,b) in enumerate(absolute) if not special_tokens_mask[i]
               and _overlaps(a,b,entity.char_start,entity.char_end)]
    if not indices:
        raise ValueError(f'{context}: empty alignment')
    if indices != list(range(indices[0], indices[-1]+1)):
        raise ValueError(f'{context}: discontinuous alignment')
    selected = [absolute[i] for i in indices]
    if any(b <= a for a,b in selected) or any(selected[i][0] < selected[i-1][1] for i in range(1,len(selected))):
        raise ValueError(f'{context}: ambiguous offsets')
    if selected[0][0] > content_start or selected[-1][1] < content_end:
        raise ValueError(f'{context}: truncated alignment')
    return AlignedEntity(entity.entity_id, entity.label, entity.char_start, entity.char_end,
                         indices[0], indices[-1])


def project_logical_window(*, document, logical_window, document_offsets):
    w = logical_window
    if (document.doc_seq_index, document.doc_id) != (w.doc_seq_index, w.doc_id):
        raise ValueError('logical window document identity mismatch')
    if not 0 <= w.token_start_global < w.token_end_global <= len(document_offsets):
        raise ValueError(f'{w.doc_id} window {w.window_index}: invalid global token bounds')
    selected = document_offsets[w.token_start_global:w.token_end_global]
    a, b = selected[0][0], selected[-1][1]
    if not 0 <= a < b <= len(document.text):
        raise ValueError(f'{w.doc_id}: invalid window character bounds')
    entities, exclusions = [], []
    for entity in document.entities:
        content_start, content_end = _content_bounds(entity, document.text)
        if a <= content_start < content_end <= b:
            entities.append(entity)
        else:
            reason = ('crosses_window_boundary' if _overlaps(a,b,entity.char_start,entity.char_end)
                      else 'outside_window')
            exclusions.append(AlignmentExclusion(w.doc_id,w.window_index,entity.entity_id,reason))
    return WindowProjection(a,b,tuple(entities),tuple(exclusions))


@dataclass(frozen=True, slots=True)
class WindowEncoding:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class BuildResult:
    rows: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


class ProtocolAmendmentRequired(ValueError):
    def __init__(self, message, subwindows):
        super().__init__(message)
        self.subwindows = tuple(subwindows)


def deterministic_whitespace_subwindows(text, start, end, tokenizer, max_length):
    import re
    boundaries = [start] + [start+m.end() for m in re.finditer(r'\S+\s*',text[start:end])]
    if boundaries[-1] != end:
        boundaries.append(end)
    parts, left = [], start
    while left < end:
        fits = []
        for right in boundaries:
            if right <= left:
                continue
            encoding = tokenizer.encode_window(text[left:right], max_length=max_length)
            if not encoding.truncated and len(encoding.input_ids) <= max_length:
                fits.append(right)
        if not fits:
            raise ValueError('single whitespace-delimited token exceeds model capacity')
        right = max(fits); parts.append((left,right)); left = right
    return tuple(parts)


def materialize_clean_window_row(*, document, logical_window, projection, encoding, inventory):
    from dataclasses import asdict
    n = len(encoding.input_ids)
    if not n or any(len(v)!=n for v in (encoding.attention_mask, encoding.offset_mapping, encoding.special_tokens_mask)):
        raise ValueError('encoding vector length mismatch')
    if any(x not in (0,1) for x in (*encoding.attention_mask,*encoding.special_tokens_mask)):
        raise ValueError('invalid encoding mask')
    mask = [bool(a and not s) for a,s in zip(encoding.attention_mask,encoding.special_tokens_mask)]
    spans, occupied = [], set()
    labels, roles, core = ['O']*n, ['O']*n, list(mask)
    source_ids = {r.source_id for r in document.relations}
    target_ids = {r.target_id for r in document.relations}
    exclusions = [dict(asdict(e), kind='entity') for e in projection.exclusions]
    for entity in projection.entities:
        aligned = align_entity(doc_id=document.doc_id, window_index=logical_window.window_index,
            entity=entity, offsets=encoding.offset_mapping, special_tokens_mask=encoding.special_tokens_mask,
            window_char_start=projection.char_start, window_char_end=projection.char_end, offsets_are_local=True,
            document_text=document.text)
        indices = set(range(aligned.token_start,aligned.token_end+1))
        if occupied & indices:
            raise ValueError(f'{document.doc_id} {entity.entity_id}: ambiguous overlapping aligned spans')
        if any(not mask[i] for i in indices):
            raise ValueError('entity aligned to masked tokens')
        occupied.update(indices)
        role = ('ROLE_BOTH' if entity.entity_id in source_ids & target_ids else
                'ROLE_1' if entity.entity_id in source_ids else
                'ROLE_2' if entity.entity_id in target_ids else 'O')
        for i in sorted(indices):
            prefix = ('S' if len(indices)==1 else 'B' if i==aligned.token_start else
                      'E' if i==aligned.token_end else 'I')
            labels[i], roles[i] = f'{prefix}-{entity.label}', role
            if inventory.is_auxiliary(entity.label):
                core[i] = False
        spans.append({'entity_id':entity.entity_id,'type':entity.label,'role':role,
            'token_start':aligned.token_start,'token_end':aligned.token_end,
            'char_start':entity.char_start,'char_end':entity.char_end})
    present = {e['entity_id'] for e in spans}
    relations = []
    for relation in document.relations:
        if relation.source_id in present and relation.target_id in present:
            relations.append({'source_id':relation.source_id,'target_id':relation.target_id,'type':relation.label})
        else:
            exclusions.append({'kind':'relation','doc_id':document.doc_id,
                'window_index':logical_window.window_index,'source_id':relation.source_id,
                'target_id':relation.target_id,'type':relation.label,'reason':'endpoint_not_in_window'})
    return {'doc_seq_index':document.doc_seq_index,'doc_id':document.doc_id,
        'window_index':logical_window.window_index,'token_start_global':logical_window.token_start_global,
        'token_end_global':logical_window.token_end_global,'input_ids':list(encoding.input_ids),
        'attention_mask':list(encoding.attention_mask),'label_mask':mask,
        'bieos_labels':labels,'role_labels':roles,'core_label_mask':core,
        'entity_spans':spans,'relations':relations}, exclusions


def audit_encoder_build(rows, exclusions, inventory):
    from collections import Counter
    entity_counts = Counter(e['type'] for r in rows for e in r['entity_spans'])
    relation_counts = Counter(e['type'] for r in rows for e in r['relations'])
    core_relations = 0
    for row in rows:
        types = {e['entity_id']:e['type'] for e in row['entity_spans']}
        core_relations += sum(inventory.is_primary(types[r['source_id']]) and
            inventory.is_primary(types[r['target_id']]) for r in row['relations'])
    return {'document_count':len({r['doc_id'] for r in rows}), 'window_count':len(rows),
        'primary_entity_count':sum(entity_counts[t] for t in inventory.primary_entity_types),
        'auxiliary_entity_count':sum(entity_counts[t] for t in inventory.auxiliary_entity_types),
        'relation_count':sum(relation_counts.values()),'core_to_core_relation_count':core_relations,
        'entity_counts':dict(entity_counts),'relation_counts':dict(relation_counts),
        'entity_exclusions_by_reason':dict(Counter(e['reason'] for e in exclusions if e['kind']=='entity')),
        'relation_exclusions_by_reason':dict(Counter(e['reason'] for e in exclusions if e['kind']=='relation')),
        'boundary_crossing_count':sum(e['reason']=='crosses_window_boundary' for e in exclusions),
        'per_document_window_counts':dict(Counter(r['doc_id'] for r in rows)),
        'entity_subword_length_distribution':dict(Counter(str(e['token_end']-e['token_start']+1)
            for r in rows for e in r['entity_spans'])),
        'total_subword_count':sum(sum(r['label_mask']) for r in rows),
        'truncation_count':0,'overflow_count':0}


def build_encoder_windows(*, documents, logical_windows, tokenizer, inventory, encoder_id,
                          encoder_revision, max_length, reference_tokenizer=None,
                          restore_frozen_overlap_policy=False):
    from dataclasses import replace
    import re
    if not re.fullmatch(r'[0-9a-f]{40}',encoder_revision):
        raise ValueError('encoder requires immutable 40-character revision')
    reference = reference_tokenizer or tokenizer
    documents_by_key = {(d.doc_seq_index,d.doc_id):d for d in documents}
    if len(documents_by_key)!=len(documents):
        raise ValueError('duplicate source identity')
    for doc in documents:
        if any(e.label not in inventory.trainable_entity_types for e in doc.entities):
            raise ValueError(f'{doc.doc_id}: entity label absent from inventory')
        if any(r.label not in inventory.relation_types for r in doc.relations):
            raise ValueError(f'{doc.doc_id}: relation label absent from inventory')
    rows, exclusions, cache, seen = [], [], {}, set()
    reference_conflicts = {}
    for doc in documents:
        for region in doc.unlabeled_regions:
            exclusions.append({'kind':'unlabeled_region','doc_id':doc.doc_id,
                'entity_id':region.entity_id,'char_start':region.char_start,'char_end':region.char_end,
                'reason':'unlabeled_unreferenced_region'})
        if restore_frozen_overlap_policy:
            offsets = cache[(doc.doc_seq_index,doc.doc_id)] = reference.document_offsets(doc.text)
            spans = []
            for entity in doc.entities:
                indices = [i for i,(a,b) in enumerate(offsets)
                           if _overlaps(a,b,entity.char_start,entity.char_end)]
                if not indices:
                    raise ValueError(f'{doc.doc_id} {entity.entity_id}: empty reference alignment')
                spans.append((indices[0],indices[-1]))
            conflicts = set()
            for i,(a,b) in enumerate(spans):
                for j in range(i+1,len(spans)):
                    c,d = spans[j]
                    if max(a,c) <= min(b,d):
                        conflicts.update((i,j))
            ids = {doc.entities[i].entity_id for i in conflicts}
            reference_conflicts[(doc.doc_seq_index,doc.doc_id)] = ids
            for entity in doc.entities:
                if entity.entity_id in ids:
                    exclusions.append({'kind':'reference_entity','doc_id':doc.doc_id,
                        'entity_id':entity.entity_id,'char_start':entity.char_start,
                        'char_end':entity.char_end,'type':entity.label,
                        'reason':'frozen_reference_overlap'})
    chars = unknown = 0
    for w in logical_windows:
        key = (w.doc_seq_index,w.doc_id)
        if key not in documents_by_key:
            raise ValueError('logical window document identity mismatch')
        if (key,w.window_index) in seen:
            raise ValueError('duplicate window identity')
        seen.add((key,w.window_index))
        doc = documents_by_key[key]
        if restore_frozen_overlap_policy:
            # Restore reference-corpus supervision before target tokenization.
            # Keep original relations so retained endpoint roles do not change.
            doc = replace(doc, entities=tuple(e for e in doc.entities
                          if e.entity_id not in reference_conflicts[key]))
        if key not in cache:
            cache[key] = reference.document_offsets(doc.text)
        projection = project_logical_window(document=doc,logical_window=w,document_offsets=cache[key])
        text = doc.text[projection.char_start:projection.char_end]
        if tokenizer is reference and hasattr(tokenizer,'encode_reference_window'):
            encoding = tokenizer.encode_reference_window(doc.text,logical_window=w,
                projection=projection,max_length=max_length)
        else:
            encoding = tokenizer.encode_window(text,max_length=max_length)
        if encoding.truncated or len(encoding.input_ids)>max_length:
            subwindows = deterministic_whitespace_subwindows(doc.text,projection.char_start,
                projection.char_end,tokenizer,max_length)
            raise ProtocolAmendmentRequired(f'{doc.doc_id} window {w.window_index} requires a matched subwindow protocol amendment',subwindows)
        row, dropped = materialize_clean_window_row(document=doc,logical_window=w,
            projection=projection,encoding=encoding,inventory=inventory)
        rows.append(row); exclusions.extend(dropped)
        chars += sum(not c.isspace() for c in text)
        unknown += sum(t==getattr(tokenizer,'unk_token_id',None) and flag
                       for t,flag in zip(encoding.input_ids,row['label_mask']))
    audit = audit_encoder_build(rows,exclusions,inventory)
    audit.update(excluded_unlabeled_region_count=sum(len(d.unlabeled_regions) for d in documents),
        reference_overlap_excluded_count=sum(len(ids) for ids in reference_conflicts.values()),
        source_overlap_policy=('exclude_all_frozen_reference_conflicts_v1'
                               if restore_frozen_overlap_policy else 'reject'),
        source_entity_count=sum(len(d.entities) for d in documents),
        source_relation_count=sum(len(d.relations) for d in documents),
        emitted_entity_count=sum(len(r['entity_spans']) for r in rows),
        emitted_relation_count=sum(len(r['relations']) for r in rows),
        excluded_entity_memberships=sum(e['kind']=='entity' for e in exclusions),
        excluded_relation_memberships=sum(e['kind']=='relation' for e in exclusions),
        unknown_token_count=unknown,unknown_token_rate=unknown/max(1,audit['total_subword_count']),
        subwords_per_non_whitespace_character=audit['total_subword_count']/max(1,chars))
    return BuildResult(tuple(rows),tuple(exclusions),audit)
