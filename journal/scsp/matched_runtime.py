"""Fail-closed runtime contract for the approved matched-window comparison."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from . import encoder_dataset_artifacts as art, encoder_dataset as ed, matched_windows as mw
from .data import LabelInventory, load_clean_windows
from .candidates import local_to_global_gold_span

ROOT=Path(__file__).resolve().parents[2]
PACKAGES={'roberta':(art.ROBERTA_ID,art.ROBERTA_REVISION),
          'securebert':(art.SECUREBERT_ID,art.SECUREBERT_REVISION)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(payload):
    return hashlib.sha256(art.json_bytes(payload)).hexdigest()


def config_paths(package):
    if package not in PACKAGES:
        raise ValueError('unknown package')
    return [ROOT/f'journal/configs/matched70_{package}.json',
            ROOT/'journal/configs/gate_a_training.json',ROOT/'journal/configs/gate_a_label_inventory.json']


def validate_settings(candidate, package):
    baseline=read_json(ROOT/'journal/configs/gate_a_plain_spanpair.json')
    allowed={'encoder_model','encoder_revision','experiment_name','output_dir'}
    if set(candidate)!=set(baseline):
        raise ValueError('base config fields differ')
    for key in baseline:
        if key not in allowed and candidate[key]!=baseline[key]:
            raise ValueError(f'changed downstream setting: {key}')
    if (candidate['encoder_model'],candidate['encoder_revision'])!=PACKAGES[package]:
        raise ValueError('encoder package mismatch')


def validate_mode(package, mode):
    if mode not in ('overfit','smoke','dev') or (package=='securebert' and mode=='dev'):
        raise ValueError(f'{package} {mode} not authorized')


def validate_bundle(bundle, package):
    if package not in PACKAGES:
        raise ValueError('unknown package')
    bundle=Path(bundle)
    lock=read_json(ROOT/'journal/configs/matched70_corpus_lock.json')
    # A versioned hash lock prevents coordinated mutation of data AND its manifest.
    for name,expected in lock.items():
        path=bundle/name
        if not path.is_file() or art.file_sha256(path)!=expected:
            raise ValueError(f'corpus manifest lock mismatch: {name}')
    common=read_json(bundle/'common_window_manifest.json')
    parent=read_json(bundle/'roberta_parity_manifest.json')
    art.validate_roberta_parity(parent)
    inventory=LabelInventory.from_json(config_paths(package)[2])
    original_partition=read_json(ROOT/'experiments/cv_manifest/run_partitions_seed_42.json')
    for name,(model,revision) in PACKAGES.items():
        folder=bundle/name
        manifest=read_json(folder/'encoder_dataset_manifest.json')
        rows=read_json(folder/'dataset.json')
        for kind in ('encoder','tokenizer'):
            if (manifest[kind]['id'],manifest[kind]['revision'])!=(model,revision):
                raise ValueError(f'{kind} identity mismatch')
        if manifest['common_window_manifest_sha256']!=digest(common):
            raise ValueError('common manifest hash mismatch')
        if manifest['dataset_sha256']!=art.file_sha256(folder/'dataset.json'):
            raise ValueError('dataset hash mismatch')
        if manifest['semantic_sha256']!=ed.semantic_sha256(rows):
            raise ValueError('semantic hash mismatch')
        split=read_json(folder/'split_manifest.json')
        if split!={**original_partition,'dataset_sha256':manifest['dataset_sha256']}:
            raise ValueError('frozen split membership changed')
        if [{key:r[key] for key in w} for r,w in zip(rows,common['windows'])]!=common['windows'] or len(rows)!=70:
            raise ValueError('common window membership changed')
        audit=ed.audit_encoder_build(rows,[],inventory)
        for key,value in {**art.FROZEN_COUNTS,'window_count':70}.items():
            if audit[key]!=value or manifest[key]!=value:
                raise ValueError(f'{key} mismatch')
        cursors={}
        for w in load_clean_windows(folder/'dataset.json',inventory):
            if len(w.input_ids)>512 or w.token_start_global!=cursors.get(w.doc_id,0):
                raise ValueError('native window capacity/coordinate mismatch')
            if w.token_end_global-w.token_start_global!=w.content_end-w.content_start+1:
                raise ValueError('native content length mismatch')
            cursors[w.doc_id]=w.token_end_global
            for span in w.gold_spans:
                local_to_global_gold_span(w,span)
    mw.validate_membership(read_json(bundle/'roberta/dataset.json'),read_json(bundle/'securebert/dataset.json'),split=False)
    return read_json(bundle/package/'encoder_dataset_manifest.json')


def contract(bundle,package):
    manifest=validate_bundle(bundle,package)
    paths=config_paths(package)
    validate_settings(read_json(paths[0]),package)
    from journal.scripts.train_gate_a import _combined_config_hash
    return {'package':package,'encoder':manifest['encoder'],
        'tokenizer':{'id':manifest['tokenizer']['id'],'revision':manifest['tokenizer']['revision']},
        'dataset_sha256':manifest['dataset_sha256'],
        'dataset_manifest_sha256':art.file_sha256(Path(bundle)/package/'encoder_dataset_manifest.json'),
        'common_window_manifest_sha256':manifest['common_window_manifest_sha256'],
        'split_manifest_sha256':manifest['split_manifest_sha256'],
        'combined_config_sha256':_combined_config_hash(paths),'fold':1,'seed':42,'split_seed':11800,
        'trust_remote_code':False,'test_evaluated':False}


def validate_components(config,embedding_vocab,tokens,shape,required_length):
    if config.model_type!='roberta':
        raise ValueError('expected roberta architecture')
    vocab=tokens['vocabulary_size']
    if vocab!=embedding_vocab or vocab!=config.vocab_size:
        raise ValueError('model/tokenizer vocabulary mismatch')
    specials=tokens['special_token_ids']
    if len(set(specials.values()))!=4 or any(type(i) is not int or not 0<=i<vocab for i in specials.values()):
        raise ValueError('invalid distinct special tokens')
    for key in ('bos','eos','pad'):
        if getattr(config,key+'_token_id')!=specials[key]:
            raise ValueError('model/tokenizer special token mismatch')
    if config.max_position_embeddings < required_length+config.pad_token_id+1:
        raise ValueError('insufficient position capacity')
    if tuple(shape)!=(1,required_length,config.hidden_size):
        raise ValueError('last_hidden_state shape mismatch')


def validate_checkpoint_package(checkpoint,package):
    if checkpoint.get('encoder_package')!=package:
        raise ValueError('checkpoint encoder package mismatch')


def validate_preflight(preflight,expected_contract):
    folder=Path(preflight)
    package=read_json(folder/'encoder_package.json')
    report=read_json(folder/'runtime_compatibility.json')
    if package.get('contract')!=expected_contract:
        raise ValueError('preflight contract mismatch')
    if report.get('status')!='passed' or report.get('test_evaluated') is not False or report.get('encoder_package_sha256')!=digest(package):
        raise ValueError('runtime preflight not passed or hash mismatch')
    return package


def validate_previous_gate(mode,overfit_run,smoke_run,package):
    import math
    if mode=='overfit':
        return
    if overfit_run is None:
        raise ValueError('smoke/dev requires overfit run')
    run=Path(overfit_run)
    config=read_json(run/'training_run_config.json')
    summary=read_json(run/'run_summary.json')
    if config.get('encoder_package')!=package or summary.get('status')!='overfit_complete' or summary.get('test_evaluated') is not False:
        raise ValueError('overfit package or status mismatch')
    start,end=summary.get('initial_loss'),summary.get('final_loss')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (start,end)) or not end<start:
        raise ValueError('overfit loss is not decreasing')
    if mode=='dev':
        if smoke_run is None:
            raise ValueError('dev requires smoke run')
        run=Path(smoke_run)
        config=read_json(run/'training_run_config.json')
        summary=read_json(run/'run_summary.json')
        if config.get('encoder_package')!=package or summary.get('status')!='smoke_complete' or summary.get('test_evaluated') is not False:
            raise ValueError('smoke package or status mismatch')
        if not (run/'best_model.pt').is_file() or not (run/'validation_metrics.json').is_file():
            raise ValueError('smoke artifacts missing')


def validate_snapshot_path(path,revision):
    # Preserve the snapshot symlink path; resolve() would point at a blob hash.
    parts=Path(path).parts
    if len(parts)<3 or parts[-3]!='snapshots' or parts[-2]!=revision:
        raise ValueError('resolved snapshot revision mismatch')


def validate_model_files(package,files):
    expected=package['resolved_model_files_sha256']
    if set(files)!=set(expected):
        raise ValueError('cached model file inventory mismatch')
    for name,path in files.items():
        if not Path(path).is_file() or art.file_sha256(path)!=expected[name]:
            raise ValueError(f'cached model hash mismatch: {name}; restore pinned weights and rerun preflight')


def validate_cached_model(package):
    from huggingface_hub import hf_hub_download
    identity=package['contract']['encoder']
    paths={name:hf_hub_download(identity['id'],name,revision=identity['revision'],local_files_only=True)
           for name in package['resolved_model_files_sha256']}
    for path in paths.values():
        validate_snapshot_path(path,identity['revision'])
    validate_model_files(package,paths)
