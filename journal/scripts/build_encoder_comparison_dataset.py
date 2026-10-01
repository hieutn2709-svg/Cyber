"""Controlled, prediction-independent SecureBERT comparison dataset gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from journal.scsp.data import LabelInventory
from journal.scsp import encoder_dataset as ed
from journal.scsp import encoder_dataset_artifacts as art


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('roberta-parity','securebert'),required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--frozen-dataset',type=Path,required=True)
    p.add_argument('--inventory',type=Path,default=ROOT/'journal/configs/gate_a_label_inventory.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--roberta-parity-manifest',type=Path)
    p.add_argument('--roberta-parity-sha256')
    p.add_argument('--source-exclusions',type=Path)
    return p


def validate_output_path(path):
    if Path(path).resolve().is_relative_to(ROOT):
        raise ValueError('dataset output must be outside the Git repository')
    if Path(path).exists():
        raise FileExistsError(str(path))


def validate_cli_contract(*,mode,model_id,revision,roberta_parity_manifest,
                          roberta_parity_sha256=None,source_sha256=None):
    if not re.fullmatch(r'[0-9a-f]{40}',revision):
        raise ValueError('requires immutable 40-character revision')
    expected={'roberta-parity':(art.ROBERTA_ID,art.ROBERTA_REVISION),
              'securebert':(art.SECUREBERT_ID,art.SECUREBERT_REVISION)}
    if mode not in expected or (model_id,revision)!=expected[mode]:
        raise ValueError('unapproved encoder package or mode')
    if mode=='roberta-parity':
        return None
    if roberta_parity_manifest is None:
        raise ValueError('SecureBERT requires a passed RoBERTa parity manifest')
    path=Path(roberta_parity_manifest)
    if art.file_sha256(path)!=roberta_parity_sha256:
        raise ValueError('RoBERTa parity manifest SHA mismatch')
    parent=json.loads(path.read_text(encoding='utf-8'))
    art.validate_roberta_parity(parent)
    if source_sha256 is not None and parent['source_sha256']!=source_sha256:
        raise ValueError('RoBERTa parity source_sha256 mismatch')
    return parent


def load_source_exclusion_policy(path, source_sha256, frozen_rows):
    if path is None:
        return frozenset()
    policy=json.loads(Path(path).read_text(encoding='utf-8'))
    if policy.get('schema_version')!=1 or policy.get('source_sha256')!=source_sha256:
        raise ValueError('source exclusion policy source_sha256/schema mismatch')
    frozen_ids={(str(r['doc_id']),e['entity_id']) for r in frozen_rows for e in r['entity_spans']}
    exclusions=set()
    for region in policy.get('regions',[]):
        key=(region.get('doc_id'),region.get('entity_id'))
        if any(not isinstance(v,str) or not v for v in key) or region.get('reason')!='unlabeled_unreferenced_region':
            raise ValueError('invalid source exclusion policy entry')
        if key in frozen_ids:
            raise ValueError('source policy cannot exclude frozen entities')
        if key in exclusions:
            raise ValueError('duplicate source exclusion entry')
        exclusions.add(key)
    return frozenset(exclusions)


def load_fast_tokenizer(model_id,revision):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,use_fast=True,trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('encoder dataset construction requires a fast tokenizer')
    return tokenizer


class FastTokenizerAdapter:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        self.unk_token_id=tokenizer.unk_token_id

    def document_offsets(self,text):
        result=self.tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        return tuple(tuple(x) for x in result['offset_mapping'])

    def encode_window(self,text,*,max_length):
        # No implicit truncation. Overflow is handled by the explicit amendment gate.
        result=self.tokenizer(text,add_special_tokens=True,truncation=False,padding=False,
            return_offsets_mapping=True,return_special_tokens_mask=True,return_attention_mask=True)
        return ed.WindowEncoding(tuple(result['input_ids']),tuple(result['attention_mask']),
            tuple(tuple(x) for x in result['offset_mapping']),tuple(result['special_tokens_mask']))


def run(args):
    validate_output_path(args.output_dir)
    if art.file_sha256(args.frozen_dataset)!=art.FROZEN_SHA256:
        raise ValueError('frozen dataset SHA mismatch')
    source_sha=art.file_sha256(args.source)
    model_id,revision=((art.ROBERTA_ID,art.ROBERTA_REVISION) if args.mode=='roberta-parity'
                       else (art.SECUREBERT_ID,art.SECUREBERT_REVISION))
    parent=validate_cli_contract(mode=args.mode,model_id=model_id,revision=revision,
        roberta_parity_manifest=args.roberta_parity_manifest,
        roberta_parity_sha256=args.roberta_parity_sha256,source_sha256=source_sha)
    # Reject invalid input before any remote package download.
    frozen=json.loads(args.frozen_dataset.read_text(encoding='utf-8'))
    source_exclusions=load_source_exclusion_policy(args.source_exclusions,source_sha,frozen)
    documents=ed.load_label_studio_documents(args.source,excluded_unlabeled_regions=source_exclusions)
    if len(documents)!=52:
        raise ValueError('source document_count must be 52')
    windows=ed.load_logical_window_contract(args.frozen_dataset)
    frozen=json.loads(args.frozen_dataset.read_text(encoding='utf-8'))
    identity=list(dict.fromkeys((w.doc_seq_index,w.doc_id) for w in windows))
    if identity!=[(d.doc_seq_index,d.doc_id) for d in documents]:
        raise ValueError('source/frozen ordered document identities differ')
    inventory=LabelInventory.from_json(args.inventory)
    frozen_inventory=LabelInventory.from_json(ROOT/'journal/configs/gate_a_label_inventory.json')
    if inventory!=frozen_inventory:
        raise ValueError('label inventory differs from frozen B3')
    if parent and parent['frozen_semantic_sha256']!=ed.semantic_sha256(frozen):
        raise ValueError('parent frozen semantic digest mismatch')
    tokenizer=load_fast_tokenizer(model_id,revision)
    reference=(tokenizer if args.mode=='roberta-parity' else
               load_fast_tokenizer(art.ROBERTA_ID,art.ROBERTA_REVISION))
    result=ed.build_encoder_windows(documents=documents,logical_windows=windows,
        tokenizer=FastTokenizerAdapter(tokenizer),reference_tokenizer=FastTokenizerAdapter(reference),
        inventory=inventory,encoder_id=model_id,encoder_revision=revision,max_length=512)
    if args.mode=='securebert':
        # Counts alone are insufficient: preserve ordered window/entity/relation membership.
        def membership(rows):
            return [(r['doc_seq_index'],r['doc_id'],r['window_index'],
                [(e['entity_id'],e['type']) for e in r['entity_spans']],r['relations']) for r in rows]
        if membership(result.rows)!=membership(frozen):
            raise ValueError('SecureBERT changed frozen annotation/window membership')
        for key,value in art.FROZEN_COUNTS.items():
            if result.audit[key]!=value:
                raise ValueError(f'{key} differs from frozen comparison')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tokenizer_files={}
    for key in ('tokenizer_file','vocab_file','merges_file'):
        path=getattr(tokenizer,'init_kwargs',{}).get(key)
        if path and Path(path).is_file():
            tokenizer_files[key]=art.file_sha256(path)
    # Serialized fast-tokenizer backend covers the resolved vocabulary when cache paths are hidden.
    tokenizer_files['resolved_backend']=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    manifest=art.build_encoder_dataset_manifest(source_sha256=source_sha,
        frozen_dataset_sha256=art.FROZEN_SHA256,encoder_id=model_id,encoder_revision=revision,
        tokenizer_class=type(tokenizer).__name__,tokenizer_files=tokenizer_files,build_result=result,
        dataset_sha256=hashlib.sha256(art.json_bytes(result.rows)).hexdigest(),
        semantic_sha256_value=ed.semantic_sha256(result.rows),builder_git_commit=commit)
    if args.source_exclusions is not None:
        manifest['source_exclusion_policy_sha256']=art.file_sha256(args.source_exclusions)
    audit={**result.audit,'tokenizer_class':type(tokenizer).__name__,'tokenizer_id':model_id,
        'tokenizer_revision':revision,'vocabulary_size':len(tokenizer),'model_max_length':tokenizer.model_max_length,
        'special_token_ids':tokenizer.all_special_ids,'dataset_sha256':manifest['dataset_sha256'],
        'source_sha256':source_sha,'semantic_sha256':manifest['semantic_sha256']}
    if parent:
        manifest.update(parity_status='passed_ancestor',roberta_parity_manifest_sha256=args.roberta_parity_sha256,
                        roberta_parity_ancestor=parent)
    return art.write_encoder_dataset_bundle(output_dir=args.output_dir,rows=result.rows,
        manifest=manifest,audit=audit,exclusions=result.exclusions,expected_rows=frozen,
        require_roberta_parity=args.mode=='roberta-parity',inventory=inventory)


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        manifest=run(args)
    except ed.ProtocolAmendmentRequired as exc:
        print(json.dumps({'status':'protocol_amendment_required','reason':str(exc),
                          'proposed_character_subwindows':exc.subwindows}),file=sys.stderr)
        return 2
    except (ValueError,OSError) as exc:
        print(json.dumps({'status':'blocked','reason':str(exc)}),file=sys.stderr)
        return 2
    print(json.dumps({'status':'passed','mode':args.mode,'dataset_sha256':manifest['dataset_sha256']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
