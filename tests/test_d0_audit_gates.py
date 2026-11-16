import importlib, gzip,json
from types import SimpleNamespace
import pytest


def test_resolved_contract_is_checked():
    r=importlib.import_module('journal.scripts.run_d0_diagnostics')
    assert hasattr(r,'validate_resolved_config')
    c=dict(seed=42,fold=1,split_seed=11800,dataset_sha256='abc',encoder=dict(id='e',revision='r'))
    b=dict(min_entity_score=.05,max_span_candidates=128,max_relation_token_distance=96,schema_mode='none',seed=42,fold=1,split_seed=11800,dataset_sha256='abc',encoder_model='e',encoder_revision='r')
    r.validate_resolved_config(b,c)
    for key,value in [('min_entity_score',.1),('max_span_candidates',96),('max_relation_token_distance',32),('schema_mode','x'),('seed',43),('encoder_revision','other')]:
        with pytest.raises(ValueError):r.validate_resolved_config({**b,key:value},c)


def test_case_identity_rejects_mislabel_and_incomplete_contract():
    a=importlib.import_module('journal.scripts.analyze_d0_diagnostics')
    assert hasattr(a,'validate_case_identity')
    v=dict(model='roberta',seed=42,cap=128,dataset_sha256='abc',split='validation',test_evaluated=False,new_training=False,
        artifacts_sha256={k:'h' for k in ('candidate_diagnostics.jsonl.gz','validation_logits.jsonl.gz','validation_predictions.jsonl')})
    s=dict(cases=[v]);a.validate_case_identity(v,s,'roberta',42,128,'abc')
    with pytest.raises(ValueError):a.validate_case_identity(v,s,'securebert',42,128,'abc')
    with pytest.raises(ValueError):a.validate_case_identity(v,s,'roberta',42,96,'abc')
    with pytest.raises(ValueError):a.validate_case_identity(v,s,'roberta',42,128,'wrong')
    v['artifacts_sha256']={}
    with pytest.raises(ValueError):a.validate_case_identity(v,s,'roberta',42,128,'abc')


def test_partial_matrix_explicit_and_every_baseline_required():
    a=importlib.import_module('journal.scripts.analyze_d0_diagnostics')
    assert hasattr(a,'validate_matrix')
    full=[dict(model=m,seed=s,cap=c) for m in ('roberta','securebert') for s in (42,43,44) for c in (128,96)]
    part=[x for x in full if (x['model'],x['seed'],x['cap'])!=('securebert',43,96)]
    with pytest.raises(ValueError):a.validate_matrix(part,False)
    assert a.validate_matrix(part,True)==['securebert_s43_cap96']
    with pytest.raises(ValueError):a.validate_matrix([x for x in part if x['cap']!=128],True)


def test_atomic_gzip_roundtrip_and_serialization_failure(tmp_path):
    r=importlib.import_module('journal.scripts.run_d0_diagnostics')
    assert hasattr(r,'write_gzip_jsonl')
    path=tmp_path/'rows.gz';rows=[{'i':i,'x':'data'*30} for i in range(20)]
    r.write_gzip_jsonl(path,rows)
    assert [json.loads(x) for x in gzip.decompress(path.read_bytes()).splitlines()]==rows
    old=path.read_bytes()
    with pytest.raises(TypeError):r.write_gzip_jsonl(path,[{'x':object()}])
    assert path.read_bytes()==old


def test_failed_identity_case_is_saved_but_not_accepted():
    r=importlib.import_module('journal.scripts.run_d0_diagnostics')
    assert hasattr(r,'identity_difference')
    from journal.scsp.structures import SpanCandidate
    def rec(start):return SimpleNamespace(document_id='400',predicted_spans=(SpanCandidate('400',start,start,'identity',.5,'predicted'),),predicted_relations=())
    diff=r.identity_difference([rec(1)],[rec(2)])
    assert diff[0]['predicted_spans']['added']==[('400',1,1,'identity')]
    assert diff[0]['predicted_spans']['removed']==[('400',2,2,'identity')]
    assert diff[0]['predicted_relations']['same_order']
