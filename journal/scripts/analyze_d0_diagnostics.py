"""Audit D0 snapshots and diagnose frozen gold without changing predictions."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
from journal.scsp.data import LabelInventory,load_clean_windows
from journal.scsp.candidates import enumerate_content_spans
from journal.scsp.artifacts import _sigmoid
from journal.scsp.serialization import read_prediction_jsonl
from journal.evaluation.rescore_gate_a import strict_micro_scores
from journal.scripts.run_d0_diagnostics import sha,read,dump,require


def classify_endpoint(snapshot,start,end,label):
    """Gold-assisted analysis only; filter order precedes the top1 type check."""
    offset=snapshot['global_token_offset']
    proposals={(s['global_start'],s['global_end']):s for s in snapshot['spans']}
    span=proposals.get((start,end))
    if start-offset<snapshot['content_start'] or end-offset>snapshot['content_end']:
        reason='outside_content'
    elif end-start+1>snapshot['width_cap']:
        reason='width_cap'
    elif span is None:
        reason='proposal_missing'
    elif span['selection_reason']!='retained':
        reason=span['selection_reason']
    elif span['label']!=label:
        reason='wrong_type'
    else:
        reason='retained'
    result=dict(gold_start=start,gold_end=end,gold_label=label,reason=reason,
                gold_type_rank=None,gold_in_top2_proposed=False,gold_in_top2_retained=False)
    if span is not None:
        labels=snapshot['classes'][1:]
        order=sorted(range(len(labels)),key=lambda i:(-span['probabilities'][i+1],i))
        rank=1+order.index(labels.index(label))
        result.update(gold_type_rank=rank,gold_in_top2_proposed=rank<=2,
            gold_in_top2_retained=rank<=2 and span['selection_reason']=='retained',
            predicted_label=span['label'],entity_score=span['entity_score'],p_none=span['p_none'],
            score_rank=span['score_rank'],selection_reason=span['selection_reason'],
            char_start=span.get('char_start'),char_end=span.get('char_end'))
    return result


def gzip_rows(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream]


def summarize(rows):
    return dict(observations=len(rows),reasons=dict(Counter(r['reason'] for r in rows)),
        gold_in_top2_proposed=sum(r['gold_in_top2_proposed'] for r in rows),
        gold_in_top2_retained=sum(r['gold_in_top2_retained'] for r in rows),
        unique_mentions=len({(r['model'],r['document_id'],r['gold_start'],r['gold_end'],r['gold_label']) for r in rows}),
        unique_mentions_note='model-specific token coordinates, deduplicated over seeds and relation roles')


def main(args):
    inv=LabelInventory.from_json(Path(__file__).resolve().parents[1]/'configs/gate_a_label_inventory.json')
    review=read(args.review); require(len(review['cases'])==22,'persistent case count')
    all_entities=[];endpoints=[];relations=[];configs=[]
    for model in ('roberta','securebert'):
        data=args.inputs/'inputs/matched70'/model/'dataset.json'
        windows={(w.doc_id,w.window_index):w for w in load_clean_windows(data,inv)}
        raw={(str(r['doc_id']),r['window_index']):r for r in read(data)}
        for seed in (42,43,44):
            run=args.results/f'{model}_s{seed}'
            summary=read(run/'run_summary.json')
            require(summary['status'] in {'D0_complete','D0_diagnostic_complete_with_archival_guard_failure'},'incomplete run')
            for cap in (128,96):
                dest=run/('baseline' if cap==128 else 'A2_cap096')
                verification=read(dest/'verification.json')
                for name,digest in verification['artifacts_sha256'].items():
                    require(sha(dest/name)==digest,'output artifact hash mismatch')
                require(verification['exact_logging_on_off'] and verification['weights_unchanged'] and verification['rng_unchanged'],'D0 parity failed')
                predictions=read_prediction_jsonl(dest/'validation_predictions.jsonl')
                require(strict_micro_scores(predictions,entity_labels=set(inv.primary_entity_types))==verification['metrics'],'rescoring mismatch')
                snaps=gzip_rows(dest/'candidate_diagnostics.jsonl.gz')
                logits=gzip_rows(dest/'validation_logits.jsonl.gz')
                require(len(snaps)==5 and len(logits)==verification['pair_count'],'stream counts mismatch')
                require(len({s['document_id'] for s in snaps})==5,'duplicate documents')
                by_doc={s['document_id']:s for s in snaps};expected_pairs=[]
                primary=[]
                for snap in snaps:
                    w=windows[(snap['document_id'],snap['window_index'])];row=raw[(w.doc_id,w.window_index)]
                    require(snap['classes']==['NONE']+list(inv.trainable_entity_types),'class order mismatch')
                    require([(s['start'],s['end']) for s in snap['spans']]==[(s.start,s.end) for s in enumerate_content_spans(w,6)],'proposal domain mismatch')
                    require(snap['label_mask']==list(w.label_mask) and snap['attention_mask']==list(w.attention_mask),'mask mismatch')
                    ordered=sorted(snap['spans'],key=lambda s:(-s['entity_score'],s['start'],s['end'],s['label']))
                    for rank,span in enumerate(ordered,1):
                        probs=span['probabilities']
                        require(len(probs)==16 and all(math.isfinite(p) and 0<=p<=1 for p in probs) and abs(sum(probs)-1)<1e-6,'posterior invalid')
                        require(span['score_rank']==rank and abs(span['entity_score']-(1-probs[0]))<1e-7,'rank/score mismatch')
                        require(span['selection_reason']==('threshold' if span['entity_score']<.05 else 'retained' if rank<=cap else 'cap'),'selection mismatch')
                        top=sorted(range(15),key=lambda i:(-probs[i+1],i))[:2]
                        require(span['top_types']==[inv.trainable_entity_types[i] for i in top] and span['label']==span['top_types'][0],'type ranking mismatch')
                        offset=w.token_start_global-w.content_start
                        require((span['global_start'],span['global_end'])==(span['start']+offset,span['end']+offset),'token coordinate mismatch')
                        require((span['char_start'],span['char_end'])==(row['char_start']+row['offset_mapping'][span['start']][0],row['char_start']+row['offset_mapping'][span['end']][1]),'character coordinate mismatch')
                    for pair in snap['generated_pairs']:
                        a,b=[snap['spans'][pair[k]] for k in ('source_index','target_index')]
                        require(a['selection_reason']==b['selection_reason']=='retained','pair uses discarded span')
                        expected_pairs.append((w.doc_id,a['global_start'],a['global_end'],a['label'],b['global_start'],b['global_end'],b['label'],pair['token_distance']))
                    for g in w.primary_gold_spans:
                        offset=w.token_start_global-w.content_start
                        item=classify_endpoint(snap,g.start+offset,g.end+offset,g.label)
                        item.update(model=model,seed=seed,cap=cap,document_id=w.doc_id)
                        primary.append(item)
                def key(r):
                    a,b=r['source'],r['target']
                    return (str(r['document_id']),a['start'],a['end'],a['label'],b['start'],b['end'],b['label'])
                require([(*key(r),r['token_distance']) for r in logits]==expected_pairs,'logged pair order mismatch')
                require(len(set(key(r) for r in logits))==len(logits),'duplicate pair')
                require(all(len(r['relation_type_logits'])==13 and all(math.isfinite(x) for x in [r['existence_logit']]+r['relation_type_logits']) for r in logits),'invalid relation logits')
                pairs={key(r):r for r in logits}
                require(len(primary)==120,'primary mention count mismatch')
                all_entities.extend(primary)
                for case in review['cases']:
                    doc=case['document_id'];snap=by_doc[doc]
                    prior=next(r for r in case['outcomes'] if (r['model'],r['seed'],r['cap'])==(model,seed,cap))
                    obs=[]
                    for side in ('source','target'):
                        a,b,label=prior[side+'_endpoint']['gold_token_key']
                        item=classify_endpoint(snap,a,b,label)
                        item.update(model=model,seed=seed,cap=cap,document_id=doc,case_id=case['case_id'],role=side)
                        endpoints.append(item);obs.append(item)
                    source,target=[prior[s+'_endpoint']['gold_token_key'] for s in ('source','target')]
                    pair=pairs.get((doc,*source,*target))
                    probability=label=None
                    if any(x['reason']!='retained' for x in obs):stage='endpoint_loss'
                    elif pair is None:stage='pair_missing'
                    else:
                        probability=_sigmoid(pair['existence_logit'])
                        label=inv.relation_types[max(range(13),key=lambda i:pair['relation_type_logits'][i])]
                        stage='below_threshold' if probability<.9 else 'correct' if label==case['identity'][2] else 'wrong_type'
                    require(stage==prior['stage'],f'persistent stage changed: {model}/{seed}/{cap}/{case["case_id"]}')
                    relations.append(dict(model=model,seed=seed,cap=cap,case_id=case['case_id'],document_id=doc,
                        stage=stage,existence_probability=probability,argmax_relation_type=label))
                configs.append(dict(model=model,seed=seed,cap=cap,primary=summarize(primary),metrics=verification['metrics'],
                    archival_parity=verification['archival_parity'],proposal_count=verification['proposal_count'],pair_count=len(logits)))
    require(len(all_entities)==1440 and len(endpoints)==528 and len(relations)==264,'matrix incomplete')
    base_end=[r for r in endpoints if r['cap']==128];base_rel=[r for r in relations if r['cap']==128]
    require(dict(Counter(r['stage'] for r in base_rel))==review['summary']['baseline_stage_counts'],'baseline reconciliation failed')
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    for filename,rows in [('all_entity_diagnostics.jsonl',all_entities),('persistent_endpoint_diagnostics.jsonl',endpoints),('persistent_relation_diagnostics.jsonl',relations)]:
        (output/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    result=dict(date='2026-10-22',configurations=configs,persistent_baseline_endpoints=summarize(base_end),
        persistent_baseline_relation_stages=dict(Counter(r['stage'] for r in base_rel)),
        unique_persistent_mentions_character_ids=len({(c['document_id'],*c['identity'][i]) for c in review['cases'] for i in (1,3)}),
        all_primary_observations=len(all_entities),all_persistent_endpoint_observations=len(endpoints),all_persistent_relation_observations=len(relations),
        all_streams_complete=True,all_saved_predictions_rescored=True,all_prior_persistent_stages_equal=True,
        archival_score_guard_failed_configurations=[f"{c['model']}_s{c['seed']}_cap{c['cap']}" for c in configs if not c['archival_parity']['score_guard_passed']],
        exact_archival_scores_all=all(c['archival_parity']['exact_scores'] for c in configs),
        review_sha256=sha(args.review),analysis_source_sha256=sha(__file__),new_training=False,test_evaluated=False,
        top2_limitation='Retrospective exact-boundary gold-type coverage, not deployable precision/F1 or recovered relation TP.',
        numeric_limitation='Archived numerical drift is measured, not resolved; the1e-5 guard is post-observation, not preregistered.')
    dump(output/'d0_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='configurations'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','review','results','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
