"""Hash-bound reconstruction manifests and all-or-nothing artifact bundles."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from .encoder_dataset import audit_encoder_build, compare_reconstruction, semantic_sha256
from .data import LabelInventory

ROBERTA_ID = 'FacebookAI/roberta-base'
ROBERTA_REVISION = 'c2a5e573587885ce23744cf330ee7c402f0df16f'
SECUREBERT_ID = 'ehsanaghaei/SecureBERT'
SECUREBERT_REVISION = '3a47918dd874e5c769efd152b51c5756a953fb67'
FROZEN_SHA256 = '190d3136edba33d89ee58f533e2d12cc6cac2842323e3168f6e3e0e71af72c48'
FROZEN_COUNTS = dict(document_count=52,window_count=67,primary_entity_count=1353,
                     auxiliary_entity_count=28,relation_count=564,core_to_core_relation_count=561)


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(value, name, length=64):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}',value):
        raise ValueError(f'{name} must be a {length}-character hexadecimal hash')


def build_encoder_dataset_manifest(*, source_sha256, frozen_dataset_sha256, encoder_id,
        encoder_revision, tokenizer_class, tokenizer_files, build_result, dataset_sha256,
        semantic_sha256_value, builder_git_commit):
    for name,value in [('source_sha256',source_sha256),('frozen_dataset_sha256',frozen_dataset_sha256),
                       ('dataset_sha256',dataset_sha256),('semantic_sha256',semantic_sha256_value)]:
        _hash(value,name)
    _hash(encoder_revision,'encoder revision',40); _hash(builder_git_commit,'builder_git_commit',40)
    for name,value in tokenizer_files.items():
        _hash(value,f'tokenizer file {name}')
    return {**build_result.audit,'schema_version':1,'source_sha256':source_sha256,
        'frozen_dataset_sha256':frozen_dataset_sha256,
        'encoder':{'id':encoder_id,'revision':encoder_revision},
        'tokenizer':{'id':encoder_id,'revision':encoder_revision,'class':tokenizer_class,
                     'files_sha256':dict(sorted(tokenizer_files.items()))},
        'dataset_sha256':dataset_sha256,'semantic_sha256':semantic_sha256_value,
        'builder_git_commit':builder_git_commit,
        'window_policy':'frozen_roberta_token_boundaries_to_character_slices_v1',
        'relation_endpoint_policy':'frozen_from_id_to_id_v1',
        'parity_status':'not_checked'}


def validate_roberta_parity(manifest):
    if manifest.get('parity_status')!='passed':
        raise ValueError('parity_status must be passed')
    if manifest.get('frozen_dataset_sha256')!=FROZEN_SHA256:
        raise ValueError('frozen_dataset_sha256 mismatch')
    for key in ('encoder','tokenizer'):
        package=manifest.get(key,{})
        if package.get('id')!=ROBERTA_ID or package.get('revision')!=ROBERTA_REVISION:
            raise ValueError(f'{key} is not approved RoBERTa package')
    for key,value in FROZEN_COUNTS.items():
        if type(manifest.get(key)) is not int or manifest[key]!=value:
            raise ValueError(f'{key} mismatch')
    for key in ('source_sha256','dataset_sha256','semantic_sha256','frozen_semantic_sha256'):
        _hash(manifest.get(key),key)
    if manifest['semantic_sha256']!=manifest['frozen_semantic_sha256']:
        raise ValueError('frozen_semantic_sha256 semantic parity mismatch')


def write_encoder_dataset_bundle(*, output_dir, rows, manifest, audit, exclusions,
        expected_rows=None, require_roberta_parity=False, inventory=None):
    output_dir=Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'output bundle already exists: {output_dir}')
    checked=dict(manifest)
    if require_roberta_parity:
        if expected_rows is None:
            raise ValueError('semantic parity requires frozen rows')
        compare_reconstruction(expected_rows,rows)
        checked['frozen_semantic_sha256']=semantic_sha256(expected_rows)
        checked['parity_status']='passed'
    data=json_bytes(rows)
    if hashlib.sha256(data).hexdigest()!=checked.get('dataset_sha256'):
        raise ValueError('dataset_sha256 mismatch')
    if semantic_sha256(rows)!=checked.get('semantic_sha256'):
        raise ValueError('semantic_sha256 mismatch')
    if inventory is None:
        inventory=LabelInventory.from_json(Path(__file__).resolve().parents[1]/'configs/gate_a_label_inventory.json')
    actual=audit_encoder_build(rows,exclusions,inventory)
    for key in FROZEN_COUNTS:
        if checked.get(key)!=actual[key]:
            raise ValueError(f'{key} differs from actual rows')
    if require_roberta_parity:
        validate_roberta_parity(checked)
    # No files are created before validation. Never replace an existing run.
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'-',dir=output_dir.parent))
    try:
        payloads={'dataset.json':data,'encoder_dataset_manifest.json':json_bytes(checked),
            'tokenizer_audit.json':json_bytes(audit),'alignment_exclusions.json':json_bytes(exclusions)}
        for name,content in payloads.items():
            with (staging/name).open('wb') as handle:
                handle.write(content); handle.flush(); os.fsync(handle.fileno())
        if output_dir.exists():
            raise FileExistsError(str(output_dir))
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return checked
