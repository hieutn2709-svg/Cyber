"""Build the approved 70-window paired corpus without accessing predictions."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from journal.scsp import encoder_dataset as ed, encoder_dataset_artifacts as art, matched_windows as mw
from journal.scsp.data import LabelInventory, load_clean_windows
from journal.scripts.build_encoder_comparison_dataset import (
    FastTokenizerAdapter, load_fast_tokenizer, load_source_exclusion_policy, validate_output_path)

SOURCE_SHA='dabd5c52c20c7aa912de65723699f246b0c12b3590289283b3c2a2fa6250e776'
PARENT_SHA='7244756f44e40b7ed10d1217652613b3f7c508b1b08b1d83540828561c329797'
PACKAGES={'roberta':(art.ROBERTA_ID,art.ROBERTA_REVISION),
          'securebert':(art.SECUREBERT_ID,art.SECUREBERT_REVISION)}
COUNTS={**art.FROZEN_COUNTS,'window_count':70}
APPROVED_CUTS={('366',4):9792,('366',7):14966,('366',8):15863}


def digest(value):
    return hashlib.sha256(art.json_bytes(value)).hexdigest()


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','frozen-dataset','parity-manifest','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    return p


def run(args):
    validate_output_path(args.output_dir)
    if art.file_sha256(args.frozen_dataset)!=art.FROZEN_SHA256:
        raise ValueError('frozen dataset SHA mismatch')
    if art.file_sha256(args.source)!=SOURCE_SHA:
        raise ValueError('source SHA mismatch')
    if art.file_sha256(args.parity_manifest)!=PARENT_SHA:
        raise ValueError('parity manifest SHA mismatch')
    parent=json.loads(args.parity_manifest.read_text())
    art.validate_roberta_parity(parent)
    frozen=json.loads(args.frozen_dataset.read_text())
    if parent['frozen_semantic_sha256']!=ed.semantic_sha256(frozen):
        raise ValueError('frozen semantic SHA mismatch')
    policy_path=ROOT/'journal/configs/securebert_source_exclusions_v1.json'
    exclusions=load_source_exclusion_policy(policy_path,SOURCE_SHA,frozen)
    if parent['source_exclusion_policy_sha256']!=art.file_sha256(policy_path):
        raise ValueError('source exclusion policy mismatch')
    docs=ed.load_label_studio_documents(args.source,excluded_unlabeled_regions=exclusions)
    inventory=LabelInventory.from_json(ROOT/'journal/configs/gate_a_label_inventory.json')
    logical=ed.load_logical_window_contract(args.frozen_dataset)
    if list(dict.fromkeys((w.doc_seq_index,w.doc_id) for w in logical)) != [(d.doc_seq_index,d.doc_id) for d in docs]:
        raise ValueError('ordered document identity mismatch')
    tokenizers={k:load_fast_tokenizer(*v) for k,v in PACKAGES.items()}
    adapters={k:FastTokenizerAdapter(v) for k,v in tokenizers.items()}
    reference=adapters['roberta']
    # Reproduce the complete original ancestor before deriving any new slices.
    old=ed.build_encoder_windows(documents=docs,logical_windows=logical,tokenizer=reference,
        reference_tokenizer=reference,inventory=inventory,encoder_id=art.ROBERTA_ID,
        encoder_revision=art.ROBERTA_REVISION,max_length=512,restore_frozen_overlap_policy=True)
    ed.compare_reconstruction(frozen,old.rows)
    common=[]; next_index=defaultdict(int); cuts={}
    for w,row in zip(logical,frozen):
        doc=docs[w.doc_seq_index]
        projection=ed.project_logical_window(document=doc,logical_window=w,
            document_offsets=reference.document_offsets(doc.text))
        retained=mw.retained_entities(doc,row)
        relations=tuple(ed.SourceRelation(r['source_id'],r['target_id'],r['type']) for r in row['relations'])
        slices=mw.choose_slices(doc.text,projection.char_start,projection.char_end,
            retained,relations,tuple(adapters.values()))
        if len(slices)>1:
            cuts[(w.doc_id,w.window_index)]=slices[0][1]
        for i,(a,b) in enumerate(slices):
            common.append(dict(doc_seq_index=w.doc_seq_index,doc_id=w.doc_id,
                window_index=next_index[w.doc_id],original_window_index=w.window_index,
                subwindow_index=i,char_start=a,char_end=b))
            next_index[w.doc_id]+=1
    if cuts!=APPROVED_CUTS:
        raise ValueError('derived cuts differ from approved three-window amendment')
    common_manifest=dict(schema_version=1,window_policy=mw.WINDOW_POLICY,
        coordinate_policy=mw.COORDINATE_POLICY,source_sha256=SOURCE_SHA,
        frozen_dataset_sha256=art.FROZEN_SHA256,roberta_parity_manifest_sha256=PARENT_SHA,
        source_exclusion_policy_sha256=art.file_sha256(policy_path),windows=common,
        max_length=512,annotation_membership='exact_frozen_ancestor',**COUNTS)
    common_sha=digest(common_manifest)
    partition_path=ROOT/'experiments/cv_manifest/run_partitions_seed_42.json'
    partition=json.loads(partition_path.read_text())
    if partition['dataset_sha256']!=art.FROZEN_SHA256:
        raise ValueError('original split dataset SHA mismatch')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    payloads={'common_window_manifest.json':common_manifest,'roberta_parity_manifest.json':parent,
        'ancestor_alignment_exclusions.json':old.exclusions}
    manifests={}
    for name,adapter in adapters.items():
        result=mw.encode_common_windows(docs,frozen,common,adapter,inventory)
        for key,value in COUNTS.items():
            if result.audit[key]!=value:
                raise ValueError(f'{name}: {key} mismatch')
        tokenizer=tokenizers[name]; model_id,revision=PACKAGES[name]
        split={**partition,'dataset_sha256':digest(result.rows)}
        manifest={**result.audit,'schema_version':2,'window_policy':mw.WINDOW_POLICY,
            'coordinate_policy':mw.COORDINATE_POLICY,'source_sha256':SOURCE_SHA,
            'frozen_dataset_sha256':art.FROZEN_SHA256,'roberta_parity_manifest_sha256':PARENT_SHA,
            'parity_status':'passed_ancestor','common_window_manifest_sha256':common_sha,
            'source_exclusion_policy_sha256':art.file_sha256(policy_path),
            'encoder':{'id':model_id,'revision':revision},
            'tokenizer':{'id':model_id,'revision':revision,'class':type(tokenizer).__name__,
                'resolved_backend_sha256':hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                'vocabulary_size':len(tokenizer),
                'special_token_ids':{k:getattr(tokenizer,k+'_token_id') for k in ('bos','eos','pad','unk')}},
            'dataset_sha256':digest(result.rows),'semantic_sha256':ed.semantic_sha256(result.rows),
            'split_manifest_sha256':digest(split),'original_split_manifest_sha256':art.file_sha256(partition_path),
            'builder_git_commit':commit,'test_evaluated':False}
        payloads.update({f'{name}/dataset.json':result.rows,f'{name}/encoder_dataset_manifest.json':manifest,
            f'{name}/split_manifest.json':split,f'{name}/tokenizer_audit.json':result.audit,
            f'{name}/alignment_exclusions.json':result.exclusions})
        manifests[name]=manifest
    # Both packages must agree on annotation supervision in each common window.
    mw.validate_membership(payloads['roberta/dataset.json'],payloads['securebert/dataset.json'],split=False)
    mw.write_atomic_bundle(args.output_dir,payloads)
    for name in PACKAGES:
        load_clean_windows(args.output_dir/name/'dataset.json',inventory)
    return {name:{k:m[k] for k in (*COUNTS,'dataset_sha256')} for name,m in manifests.items()}


def main(argv=None):
    try:
        result=run(build_parser().parse_args(argv))
    except (ValueError,OSError) as exc:
        print(json.dumps({'status':'blocked','reason':str(exc)}),file=sys.stderr)
        return 2
    print(json.dumps({'status':'passed','packages':result},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
