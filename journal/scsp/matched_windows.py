"""Approved character-matched corpus; native token coordinates are package-local."""
from __future__ import annotations

from collections import Counter, defaultdict
import os
from pathlib import Path
import shutil
import tempfile
from . import encoder_dataset as ed
from .encoder_dataset_artifacts import json_bytes

COORDINATE_POLICY = 'concatenated_native_window_content_per_document_v1'
WINDOW_POLICY = 'matched_character_slices_v1'


def retained_entities(document, frozen_row):
    source = {e.entity_id: e for e in document.entities}
    result = []
    for raw in frozen_row['entity_spans']:
        entity = source.get(raw['entity_id'])
        if entity is None or entity.key != (raw['entity_id'], raw['char_start'], raw['char_end'], raw['type']):
            raise ValueError('frozen/source entity mismatch')
        result.append(entity)
    return tuple(result)


def choose_slices(text, start, end, entities, relations, tokenizers, max_length=512):
    if not 0 <= start < end <= len(text) or len(tokenizers) != 2:
        raise ValueError('invalid character interval or paired tokenizers')
    def fits(a, b):
        if not text[a:b].strip():
            return False
        for tokenizer in tokenizers:
            encoding = tokenizer.encode_window(text[a:b], max_length=max_length)
            if encoding.truncated or len(encoding.input_ids) > max_length:
                return False
        return True
    if fits(start, end):
        return ((start, end),)
    by_id = {e.entity_id:e for e in entities}
    for cut in range(end-1, start, -1):
        if not text[cut-1].isspace():
            continue
        if any(e.char_start < cut < e.char_end for e in entities):
            continue
        if any((by_id[r.source_id].char_end <= cut) !=
               (by_id[r.target_id].char_end <= cut) for r in relations):
            continue
        if fits(start, cut) and fits(cut, end):
            return ((start, cut), (cut, end))
    raise ValueError('no safe shared cut; protocol review required')


def annotation_membership(rows, *, split):
    entities, relations = Counter(), Counter()
    for r in rows:
        key = (r['doc_seq_index'],str(r['doc_id']),
               r['original_window_index'] if split else r['window_index'])
        for e in r['entity_spans']:
            entities[key + (e['entity_id'],e['type'],e['char_start'],e['char_end'],e['role'])] += 1
        for rel in r['relations']:
            relations[key + (rel['source_id'],rel['target_id'],rel['type'])] += 1
    return entities, relations


def validate_membership(frozen_rows, rows, *, split=True):
    if annotation_membership(frozen_rows,split=False) != annotation_membership(rows,split=split):
        raise ValueError('annotation membership changed')


def encode_common_windows(documents, frozen_rows, common, tokenizer, inventory, max_length=512):
    docs={(d.doc_seq_index,d.doc_id):d for d in documents}
    originals={(r['doc_seq_index'],str(r['doc_id']),r['window_index']):r for r in frozen_rows}
    cursors=defaultdict(int)
    rows,exclusions=[],[]
    for w in common:
        doc=docs[(w['doc_seq_index'],w['doc_id'])]
        original=originals[(doc.doc_seq_index,doc.doc_id,w['original_window_index'])]
        a,b=w['char_start'],w['char_end']
        retained=retained_entities(doc,original)
        # Whitespace outside original bounds is allowed, as in verified parity.
        entities=tuple(e for e in retained if a <= ed._content_bounds(e,doc.text)[0]
                       and ed._content_bounds(e,doc.text)[1] <= b)
        encoding=tokenizer.encode_window(doc.text[a:b],max_length=max_length)
        if encoding.truncated or len(encoding.input_ids)>max_length:
            raise ValueError('native encoding exceeds shared capacity')
        mask=encoding.special_tokens_mask
        if tuple(mask) != (1,*(0 for _ in range(len(mask)-2)),1) or len(mask)<3:
            raise ValueError('expected BOS, native content, EOS without padding')
        n=len(mask)-2
        start=cursors[doc.doc_id]
        logical=ed.LogicalWindow(doc.doc_seq_index,doc.doc_id,w['window_index'],start,start+n)
        row,dropped=ed.materialize_clean_window_row(document=doc,logical_window=logical,
            projection=ed.WindowProjection(a,b,entities,()),encoding=encoding,inventory=inventory)
        row.update(w)
        row['offset_mapping']=[list(pair) for pair in encoding.offset_mapping]
        row['coordinate_policy']=COORDINATE_POLICY
        cursors[doc.doc_id]+=n
        rows.append(row); exclusions.extend(dropped)
    validate_membership(frozen_rows,rows)
    return ed.BuildResult(tuple(rows),tuple(exclusions),ed.audit_encoder_build(rows,exclusions,inventory))


def write_atomic_bundle(output_dir, payloads):
    output=Path(output_dir)
    if output.exists():
        raise FileExistsError(str(output))
    data={}
    for name,value in payloads.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('unsafe bundle path')
        data[name]=json_bytes(value)
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        for name,content in data.items():
            dest=staging/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('wb') as handle:
                handle.write(content); handle.flush(); os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(str(output))
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
