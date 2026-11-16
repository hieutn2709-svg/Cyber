"""Replay six frozen cap128 D0 runs; save gold-free predictions before scoring."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from journal.evaluation.rescore_gate_a import strict_micro_scores
from journal.scsp.multi_type_decode import decode_snapshots, span_key, relation_key
from journal.scsp.serialization import read_prediction_jsonl


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_input(root, path, manifest):
    name = str(path.relative_to(root))
    if name not in manifest or sha(path) != manifest[name]:
        raise ValueError(f'input hash missing or changed: {name}')
    return manifest[name]


def prepare_output(path):
    path.mkdir(parents=True, exist_ok=False)


def require_parity(predicted, baseline):
    if [r['document_id'] for r in predicted] != [r['document_id'] for r in baseline]:
        raise ValueError('baseline document order mismatch')
    for a, b in zip(predicted, baseline):
        for key in ['predicted_spans', 'predicted_relations']:
            if a[key] != b[key]:
                raise ValueError(f'baseline exact prediction parity failed: {a["document_id"]} {key}')


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in rows))


def run(d0, output):
    repo = Path(__file__).resolve().parents[2]
    inventory = json.loads((repo/'journal/configs/gate_a_label_inventory.json').read_text())
    primary = set(inventory['primary_entity_types'])
    manifest = json.loads((d0/'SHA256SUMS.json').read_text())
    files = {}
    for model, seed in itertools.product(['roberta','securebert'],[42,43,44]):
        folder = d0/'output'/f'{model}_s{seed}'/'baseline'
        for name in ['candidate_diagnostics.jsonl','validation_logits.jsonl','validation_predictions.jsonl']:
            path = folder/name
            files[str(path.relative_to(d0))] = verify_input(d0,path,manifest)
    prepare_output(output)
    base_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    code_files = ['journal/scsp/multi_type_decode.py','journal/scripts/run_e1_multi.py',
                  'journal/scsp/serialization.py','journal/evaluation/rescore_gate_a.py',
                  'journal/configs/gate_a_label_inventory.json']
    provenance = dict(repository_base_commit=base_commit,code_sha256={f:sha(repo/f) for f in code_files},
                      input_sha256=files,python=platform.python_version(),platform=platform.platform(),
                      run_scope='exploratory validation on unchanged frozen gold',human_adjudication='pending',
                      protocol=dict(cap=128,width=6,min_entity_score=.05,max_distance=96,relation_threshold=.9,k_values=[1,2]))
    fingerprint = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results=[]
    for model, seed in itertools.product(['roberta','securebert'],[42,43,44]):
        source = d0/'output'/f'{model}_s{seed}'/'baseline'
        snapshots = read_rows(source/'candidate_diagnostics.jsonl')
        pair_rows = read_rows(source/'validation_logits.jsonl')
        if sorted(str(s['document_id']) for s in snapshots) != ['365','389','395','400','420']:
            raise ValueError('unexpected validation documents/windows')
        for s in snapshots:
            actual = (s['max_span_candidates'],s['width_cap'],s['min_entity_score'],s['max_relation_token_distance'])
            if actual != (128,6,.05,96):
                raise ValueError('D0 protocol differs from frozen E1 scope')
        target = output/f'{model}_s{seed}'
        target.mkdir()
        predictions={};costs={};prediction_hashes={}
        for k in [1,2]:
            tracemalloc.start()
            start = time.perf_counter()
            predictions[k] = decode_snapshots(snapshots,pair_rows,inventory['relation_types'],k=k)
            seconds = time.perf_counter()-start
            _,peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            path = target/f'k{k}_predictions_only.jsonl'
            write_rows(path,predictions[k])
            prediction_hashes[k] = sha(path)
            costs[k] = dict(decode_seconds_with_tracemalloc=seconds,peak_traced_python_bytes=peak,
                            scored_boundary_pairs=len(pair_rows),typed_combinations_before_threshold=k*k*len(pair_rows))
        # First gold read occurs only AFTER both prediction files are saved and hashed.
        baseline = read_rows(source/'validation_predictions.jsonl')
        if any(r['split']!='validation' for r in baseline):
            raise ValueError('non-validation baseline')
        require_parity(predictions[1],baseline)
        for a,b in zip(predictions[1],predictions[2]):
            if not {span_key(s) for s in a['predicted_spans']} <= {span_key(s) for s in b['predicted_spans']}:
                raise ValueError('top1 entity subset check failed')
            if not {relation_key(r) for r in a['predicted_relations']} <= {relation_key(r) for r in b['predicted_relations']}:
                raise ValueError('top1 relation subset check failed')
        metrics={}
        for k in [1,2]:
            joined=[]
            for original,pred in zip(baseline,predictions[k]):
                joined.append({**original,**pred,'run_id':f'e1_multi_{model}_{seed}_k{k}',
                               'git_commit':base_commit,'config_sha256':fingerprint})
            path=target/f'k{k}_scored_records.jsonl'
            write_rows(path,joined)
            records=read_prediction_jsonl(path)
            metrics[k]=dict(all_labels=strict_micro_scores(records),
                            primary_entity_all_relation=strict_micro_scores(records,entity_labels=primary),
                            core_to_core=strict_micro_scores(records,entity_labels=primary,relation_endpoint_labels=primary),
                            per_document={r.document_id:strict_micro_scores([r],entity_labels=primary) for r in records})
            costs[k]['emitted_relations_after_merge']=sum(len(r['predicted_relations']) for r in predictions[k])
            costs[k]['emitted_typed_entities_after_merge']=sum(len(r['predicted_spans']) for r in predictions[k])
            multiplicities=Counter((r['document_id'],s['start'],s['end']) for r in predictions[k] for s in r['predicted_spans'])
            costs[k]['unique_boundaries_after_merge']=len(multiplicities)
            costs[k]['multi_type_boundaries']=sum(n>1 for n in multiplicities.values())
            edge_counts=Counter((r['document_id'],e['source']['start'],e['source']['end'],e['target']['start'],e['target']['end']) for r in predictions[k] for e in r['predicted_relations'])
            costs[k]['multi_type_relation_boundary_pairs']=sum(n>1 for n in edge_counts.values())
        # Stable sigmoid is identical to decoder; count emissions before cross-window merge.
        def positive(v):
            if v>=0:return 1/(1+math.exp(-v))>=.9
            z=math.exp(v);return z/(1+z)>=.9
        positive_pairs=sum(positive(r['existence_logit']) for r in pair_rows)
        for k in [1,2]:costs[k]['typed_combinations_after_threshold_before_merge']=positive_pairs*k*k
        item=dict(model=model,seed=seed,exact_top1_prediction_parity=True,top1_subset_top2=True,
                  prediction_only_sha256=prediction_hashes,costs=costs,metrics=metrics)
        results.append(item)
        (target/'result.json').write_text(json.dumps(item,indent=2)+'\n')
        print(f'{model} seed{seed}: top1 exact; relation TP/FP/FN '+str([metrics[k]['all_labels']['relation'] for k in [1,2]]),flush=True)
    aggregate={}
    for model in ['roberta','securebert']:
        aggregate[model]={}
        subset=[r for r in results if r['model']==model]
        for task in ['entity','relation']:
            aggregate[model][task]={}
            for measure in ['precision','recall','f1']:
                values={k:[r['metrics'][k]['primary_entity_all_relation'][task][measure] for r in subset] for k in [1,2]}
                aggregate[model][task][measure]={str(k):dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v)) for k,v in values.items()}
                deltas=[b-a for a,b in zip(values[1],values[2])]
                aggregate[model][task][measure]['paired_delta']=dict(mean=statistics.mean(deltas),sample_sd=statistics.stdev(deltas))
    summary=dict(provenance=provenance,results=results,aggregate=aggregate,new_training=False,new_forward=False,gold_changed=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d0',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.d0.resolve(),args.output.resolve())
