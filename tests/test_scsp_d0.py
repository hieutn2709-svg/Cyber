from dataclasses import replace
from types import SimpleNamespace
import importlib.util
import inspect
import pytest
import torch
from journal.scsp.training import infer_window
from tests import test_scsp_training_core as core


def test_diagnostics_preserve_full_inference_weights_rng_and_ignore_gold():
    assert 'diagnostics' in inspect.signature(infer_window).parameters
    fixture = core.TrainingCoreTests()
    model = fixture._model().eval()
    window = fixture._window()
    inv = fixture._inventory()
    cfg = SimpleNamespace(max_span_candidates=2, min_entity_score=0., max_relation_token_distance=10)
    kwargs = dict(width_cap=2, base_config=cfg, relation_chunk_size=4, device=torch.device('cpu'))
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    plain = infer_window(model, window, inv, **kwargs)
    logs = []
    logged = infer_window(model, window, inv, diagnostics=logs, **kwargs)
    assert plain == logged
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
    assert len(logs)==1 and len(logs[0]['spans'])==9
    assert sorted(x['score_rank'] for x in logs[0]['spans'])==list(range(1,10))
    assert sum(x['selection_reason']=='retained' for x in logs[0]['spans'])==2
    assert all(len(x['probabilities'])==4 and abs(sum(x['probabilities'])-1)<1e-6 for x in logs[0]['spans'])
    other=[]
    infer_window(model,replace(window,gold_spans=(),gold_relations=()),inv,diagnostics=other,**kwargs)
    assert other==logs


def test_threshold_empty_case_still_logs_full_posterior_and_non_none_type():
    assert 'diagnostics' in inspect.signature(infer_window).parameters
    fixture=core.TrainingCoreTests();model=fixture._model().eval()
    head=model.heads.entity_head
    # All candidates tie; NONE is highest but the official type excludes NONE.
    for p in head.parameters(): p.data.zero_()
    list(head.parameters())[-1].data.copy_(torch.tensor([10.,2.,1.,0.]))
    logs=[]
    result=infer_window(model,fixture._window(),fixture._inventory(),width_cap=2,
        base_config=SimpleNamespace(max_span_candidates=2,min_entity_score=.99,max_relation_token_distance=10),
        relation_chunk_size=4,device=torch.device('cpu'),diagnostics=logs)
    assert not result.predicted_spans and not result.scored_pairs
    assert logs[0]['generated_pairs']==[]
    assert all(s['selection_reason']=='threshold' and s['label']=='intrusion-set' and s['top_types'][0]=='intrusion-set' for s in logs[0]['spans'])
    assert [(s['start'],s['end']) for s in sorted(logs[0]['spans'],key=lambda s:s['score_rank'])]==sorted((s['start'],s['end']) for s in logs[0]['spans'])


def test_archive_drift_is_measured_and_failed_guard_remains_failed():
    assert importlib.util.find_spec('journal.scsp.inference_diagnostics') is not None
    from journal.scsp.inference_diagnostics import compare_archived_records
    from journal.scsp.structures import SpanCandidate
    def record(score=.8,label='malware'):
        return SimpleNamespace(document_id='a',gold_spans=(),gold_relations=(),predicted_spans=(SpanCandidate('a',1,1,label,score,'predicted'),),predicted_relations=())
    old=[record()]
    small=compare_archived_records([record(.800001)],old)
    assert small['score_guard_passed'] and not small['exact_scores']
    with pytest.raises(ValueError,match='score'):
        compare_archived_records([record(.81)],old)
    drift=compare_archived_records([record(.81)],old,strict_scores=False)
    assert not drift['score_guard_passed'] and not drift['exact_scores']
    with pytest.raises(ValueError,match='identit'):
        compare_archived_records([record(label='tool')],old,strict_scores=False)
