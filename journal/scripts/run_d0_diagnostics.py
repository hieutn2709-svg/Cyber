"""Validation-only D0 on immutable matched70 selected checkpoints.

Invoke as a module. --diagnose-archival-drift measures historical numeric drift
without relaxing decision, metric or exact current on/off parity requirements.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
import torch
import transformers
from transformers import AutoConfig, AutoModel
from journal.scsp.runtime_model import GateASpanPairModel
from journal.scsp.data import LabelInventory, load_clean_windows
from journal.scsp.training import infer_window
from journal.scsp.artifacts import build_prediction_records
from journal.scsp.serialization import read_prediction_jsonl, write_prediction_jsonl
from journal.scsp.inference_diagnostics import compare_archived_records
from journal.evaluation.rescore_gate_a import strict_micro_scores


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_resolved_config(base, contract):
    expected=dict(min_entity_score=.05,max_span_candidates=128,max_relation_token_distance=96,
        schema_mode='none',seed=contract['seed'],fold=contract['fold'],split_seed=contract['split_seed'],
        dataset_sha256=contract['dataset_sha256'],encoder_model=contract['encoder']['id'],
        encoder_revision=contract['encoder']['revision'])
    for key,value in expected.items():
        require(base.get(key)==value,f'resolved inference contract mismatch: {key}')


def write_gzip_jsonl(path, rows):
    """Atomically install a complete, read-back-verified gzip stream."""
    raw=''.join(json.dumps(row)+'\n' for row in rows).encode('utf-8')
    compressed=gzip.compress(raw,mtime=0)
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(compressed);stream.flush();os.fsync(stream.fileno())
        require(temporary.read_bytes()==compressed,'gzip write bytes mismatch')
        require(gzip.decompress(temporary.read_bytes())==raw,'gzip roundtrip mismatch')
        os.replace(temporary,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def identity_difference(current, archived):
    result=[]
    require([r.document_id for r in current]==[r.document_id for r in archived],'document mismatch')
    for a,b in zip(current,archived):
        row=dict(document_id=a.document_id)
        for name,key in [('predicted_spans','typed_key'),('predicted_relations','strict_key')]:
            left=[getattr(x,key) for x in getattr(a,name)]
            right=[getattr(x,key) for x in getattr(b,name)]
            row[name]=dict(same_order=left==right,same_set=set(left)==set(right),
                added=sorted(set(left)-set(right)),removed=sorted(set(right)-set(left)))
        result.append(row)
    return result


def main(args):
    root=args.inputs; repo=Path(__file__).resolve().parents[2]
    out=args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    manifest=root/'external_checkpoints.json'
    entry=next(x for x in read(manifest) if x['model']==args.model and x['seed']==args.seed)
    checkpoint=args.checkpoints/entry['file']
    require(sha(checkpoint)==entry['sha256'], 'checkpoint SHA mismatch')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    run=(root/'inputs/runs70'/('roberta_dev_v3' if args.model=='roberta' else 'securebert_dev')
         if args.seed==42 else root/'runs'/f'{args.model}_s{args.seed}')
    summary=read(run/'run_summary.json'); base=read(run/'resolved_config.json')
    require(summary['status']=='dev_complete' and not summary['test_evaluated'], 'not a completed dev checkpoint')
    require(summary['seed']==args.seed and summary['fold']==1,'wrong seed/fold')
    for k in ('dataset_sha256','config_sha256','git_commit'):
        require(ckpt[k]==summary[k],f'checkpoint metadata mismatch: {k}')
    require(ckpt['epoch']==summary['best_epoch'] and ckpt['threshold']==summary['relation_threshold']==.9 and ckpt['width_cap']==6,'selection mismatch')
    package=ckpt['encoder_package']; contract=package['contract']
    validate_resolved_config(base,contract)
    require(ckpt['config_sha256']==contract['combined_config_sha256'],'combined config link mismatch')
    require(contract['package']==args.model and contract['seed']==args.seed and contract['fold']==1
            and contract['split_seed']==11800 and not contract['test_evaluated'],'contract mismatch')
    data=root/'inputs/matched70'/args.model/'dataset.json'
    require(sha(data)==ckpt['dataset_sha256']==contract['dataset_sha256'],'dataset SHA mismatch')
    cfgpath=args.encoder_configs/args.model/'config.json'
    require(sha(cfgpath)==package['resolved_model_files_sha256']['config.json'],'encoder config SHA mismatch')
    invpath=repo/'journal/configs/gate_a_label_inventory.json'
    inv=LabelInventory.from_json(invpath)
    docs={'365','389','395','400','420'}
    require(set(base['validation_document_ids'])==docs,'validation split mismatch')
    windows=sorted((w for w in load_clean_windows(data,inv) if w.doc_id in docs),key=lambda w:w.doc_id)
    require(len(windows)==5 and {w.doc_id for w in windows}==docs,'validation window mismatch')
    require(sum(len(w.primary_gold_spans) for w in windows)==120 and sum(len(w.gold_relations) for w in windows)==49,'gold counts mismatch')
    raw={(str(r['doc_id']),r['window_index']):r for r in read(data)}
    config=AutoConfig.from_pretrained(str(cfgpath.parent),local_files_only=True,trust_remote_code=False)
    encoder=AutoModel.from_config(config,trust_remote_code=False)
    model=GateASpanPairModel(encoder=encoder,hidden_size=package['hidden_size'],
        num_entity_classes=len(inv.trainable_entity_types)+1,num_relation_types=len(inv.relation_types),
        max_width=6,width_embedding_dim=32,context_dim=128,distance_embedding_dim=32,max_distance=96)
    model.load_state_dict(ckpt['model_state_dict'],strict=True)
    require(all(torch.equal(v,ckpt['model_state_dict'][k]) for k,v in model.state_dict().items()),'tensor load mismatch')
    model.eval()
    source_files=list((repo/'journal/scsp').glob('*.py'))+[Path(__file__),invpath,repo/'journal/evaluation/rescore_gate_a.py']
    source_hashes={str(p.relative_to(repo)):sha(p) for p in source_files}
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    environment=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
        threads=torch.get_num_threads(),device='cpu',attention_implementation=model.encoder.config._attn_implementation,
        platform=platform.platform(),torch_config=torch.__config__.show())
    results=[]
    for cap in (128,96):
        case='baseline' if cap==128 else 'A2_cap096'
        dest=out/case;dest.mkdir()
        reference=(root/'seed42_forward/results'/args.model/case/'validation_predictions.jsonl' if args.seed==42
                   else root/'evaluations'/f'{args.model}_s{args.seed}'/case/'validation_predictions.jsonl')
        cfg=SimpleNamespace(min_entity_score=.05,max_span_candidates=cap,max_relation_token_distance=96)
        inferences=[];diagnostics=[]
        for window in windows:
            rng=torch.get_rng_state().clone()
            kw=dict(width_cap=6,base_config=cfg,relation_chunk_size=512,device=torch.device('cpu'))
            plain=infer_window(model,window,inv,**kw)
            logs=[];logged=infer_window(model,window,inv,diagnostics=logs,**kw)
            require(plain==logged,'D0 changed complete inference/order/logits')
            require(torch.equal(rng,torch.get_rng_state()),'D0 changed RNG')
            require(len(logs)==1,'missing diagnostic window')
            snap=logs[0]; row=raw[(window.doc_id,window.window_index)]
            offset=window.token_start_global-window.content_start
            snap['global_token_offset']=offset
            for span in snap['spans']:
                span.update(global_start=span['start']+offset,global_end=span['end']+offset,
                    char_start=row['char_start']+row['offset_mapping'][span['start']][0],
                    char_end=row['char_start']+row['offset_mapping'][span['end']][1])
            diagnostics.append(snap);inferences.append(logged)
            print(args.model,args.seed,cap,window.doc_id,'exact on/off parity',len(logged.scored_pairs),flush=True)
        records=build_prediction_records(inferences,inv.relation_types,.9,run_id=f'd0_{args.model}_{args.seed}_{cap}_20261022',
            git_commit=git_head,dataset_sha256=sha(data),config_sha256=summary['config_sha256'],fold=1,seed=args.seed,split='validation')
        try:
            archival=compare_archived_records(records,read_prediction_jsonl(reference),strict_scores=False)
        except ValueError as error:
            require(all(torch.equal(v,ckpt['model_state_dict'][k]) for k,v in model.state_dict().items()),'weights changed')
            write_prediction_jsonl(records,dest/'failed_validation_predictions.jsonl')
            write_gzip_jsonl(dest/'failed_candidate_diagnostics.jsonl.gz',diagnostics)
            failure=dict(model=args.model,seed=args.seed,cap=cap,error=str(error),
                status='blocked_prediction_identity',exact_logging_on_off=True,rng_unchanged=True,weights_unchanged=True,
                identity_difference=identity_difference(records,read_prediction_jsonl(reference)),
                metrics=strict_micro_scores(records,entity_labels=set(inv.primary_entity_types)),
                archived_metrics=strict_micro_scores(read_prediction_jsonl(reference),entity_labels=set(inv.primary_entity_types)),
                reference_predictions_sha256=sha(reference),source_files_sha256=source_hashes,
                artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_training=False,test_evaluated=False)
            dump(dest/'failure.json',failure)
            dump(out/'run_summary.json',dict(status='blocked_prediction_identity',cases=results,failed_case=failure))
            raise
        metrics=strict_micro_scores(records,entity_labels=set(inv.primary_entity_types))
        require(metrics==strict_micro_scores(read_prediction_jsonl(reference),entity_labels=set(inv.primary_entity_types)),'strict metrics changed')
        write_prediction_jsonl(records,dest/'validation_predictions.jsonl')
        require(tuple(read_prediction_jsonl(dest/'validation_predictions.jsonl'))==tuple(records),'prediction roundtrip changed')
        write_gzip_jsonl(dest/'candidate_diagnostics.jsonl.gz',diagnostics)
        pair_rows=[]
        for inf in inferences:
            offset=inf.window.token_start_global-inf.window.content_start
            for score in inf.scored_pairs:
                pair=score.pair
                def endpoint(s):return dict(start=s.start+offset,end=s.end+offset,label=s.label)
                pair_rows.append(dict(document_id=inf.window.doc_id,window_index=inf.window.window_index,
                    source=endpoint(pair.source),target=endpoint(pair.target),token_distance=pair.token_distance,
                    existence_logit=score.existence_logit,relation_type_logits=score.type_logits))
        write_gzip_jsonl(dest/'validation_logits.jsonl.gz',pair_rows)
        pair_count=len(pair_rows)
        require(all(torch.equal(v,ckpt['model_state_dict'][k]) for k,v in model.state_dict().items()),'weights changed')
        result=dict(model=args.model,seed=args.seed,cap=cap,selected_epoch=ckpt['epoch'],threshold=.9,
            metrics=metrics,archival_parity=archival,exact_logging_on_off=True,weights_unchanged=True,rng_unchanged=True,
            pair_count=pair_count,proposal_count=sum(len(s['spans']) for s in diagnostics),
            new_training=False,test_evaluated=False,split='validation',environment=environment,
            checkpoint_sha256=entry['sha256'],dataset_sha256=sha(data),encoder_config_sha256=sha(cfgpath),
            reference_predictions_sha256=sha(reference),checkpoint_manifest_sha256=sha(manifest),
            original_training_git_commit=summary['git_commit'],source_head=git_head,source_files_sha256=source_hashes,
            artifacts_sha256={p.name:sha(p) for p in dest.iterdir()})
        dump(dest/'verification.json',result);results.append(result)
        if not archival['score_guard_passed'] and not args.diagnose_archival_drift:
            dump(out/'run_summary.json',dict(status='blocked_archival_score_guard',cases=results))
            raise ValueError('Archived score guard failed; measurements saved. Explicit diagnostic-only mode required to continue.')
    status=('D0_diagnostic_complete_with_archival_guard_failure'
            if any(not r['archival_parity']['score_guard_passed'] for r in results) else 'D0_complete')
    dump(out/'run_summary.json',dict(status=status,cases=results))
    print(status,args.model,args.seed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['roberta','securebert'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    for flag in ('inputs','checkpoints','encoder-configs','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--diagnose-archival-drift',action='store_true')
    main(parser.parse_args())
