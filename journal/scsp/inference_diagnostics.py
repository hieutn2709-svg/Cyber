"""Read-only D0 snapshots and explicit historical numerical-drift measurement."""
from __future__ import annotations

import math
import torch
from .spans import prune_span_candidates


def candidate_snapshot(window, inventory, entity_logits, scored, pruned, pairs, *, width_cap, base_config):
    """Record the official decisions; neither consult gold nor change candidates."""
    probabilities = torch.softmax(entity_logits.detach(), dim=-1).cpu().tolist()
    types = list(inventory.trainable_entity_types)
    ranked = prune_span_candidates(scored, max_candidates=max(1, len(scored)), min_entity_score=0.)
    ranks = {s.typed_key:i+1 for i,s in enumerate(ranked)}
    kept = {s.typed_key for s in pruned}
    indices = {s.typed_key:i for i,s in enumerate(scored)}
    spans = []
    for span, probs in zip(scored, probabilities):
        top = sorted(range(len(types)), key=lambda i:(-probs[i+1],i))[:2]
        reason = ('threshold' if span.entity_score < base_config.min_entity_score
                  else 'retained' if span.typed_key in kept else 'cap')
        spans.append(dict(start=span.start,end=span.end,width=span.width,label=span.label,
                          entity_score=span.entity_score,probabilities=probs,p_none=probs[0],
                          top_types=[types[i] for i in top],score_rank=ranks[span.typed_key],
                          selection_reason=reason))
    return dict(document_id=window.doc_id,window_index=window.window_index,
                content_start=window.content_start,content_end=window.content_end,
                label_mask=list(window.label_mask),attention_mask=list(window.attention_mask),
                width_cap=width_cap,min_entity_score=base_config.min_entity_score,
                max_span_candidates=base_config.max_span_candidates,
                max_relation_token_distance=base_config.max_relation_token_distance,
                classes=['NONE']+types,spans=spans,
                generated_pairs=[dict(source_index=indices[p.source.typed_key],
                                      target_index=indices[p.target.typed_key],
                                      token_distance=p.token_distance) for p in pairs])


def compare_archived_records(current, archived, *, strict_scores=True):
    """Always require exact identities/order/gold; report rather than hide drift.

    1e-5 is an exploratory guard introduced after the interrupted Oct20 run,
    not a preregistered equivalence bound. Diagnostic mode NEVER turns a failed
    numerical guard into a pass and NEVER permits changed prediction decisions.
    """
    if [r.document_id for r in current] != [r.document_id for r in archived]:
        raise ValueError('document identities/order changed')
    entity_diffs, relation_diffs = [], []
    for a,b in zip(current,archived):
        if a.gold_spans!=b.gold_spans or a.gold_relations!=b.gold_relations:
            raise ValueError('gold identities changed')
        for name,key,score,out in [('predicted_spans','typed_key','entity_score',entity_diffs),
                                   ('predicted_relations','strict_key','relation_score',relation_diffs)]:
            left,right=getattr(a,name),getattr(b,name)
            if [getattr(x,key) for x in left] != [getattr(x,key) for x in right]:
                raise ValueError('prediction identities/order changed')
            for x,y in zip(left,right):
                if name=='predicted_spans' and x.proposal_source!=y.proposal_source:
                    raise ValueError('proposal identities changed')
                values=getattr(x,score),getattr(y,score)
                if not all(math.isfinite(v) for v in values):
                    raise ValueError('nonfinite score')
                out.append(abs(values[0]-values[1]))
    entity=max(entity_diffs,default=0.);relation=max(relation_diffs,default=0.)
    passed=max(entity,relation)<=1e-5
    result=dict(exact_prediction_keys_and_order=True,exact_scores=max(entity,relation)==0.,
                max_entity_score_abs_diff=entity,max_relation_score_abs_diff=relation,
                score_abs_tolerance=1e-5,score_guard_passed=passed)
    if strict_scores and not passed:
        raise ValueError(f'archived score guard failed: {result}')
    return result
