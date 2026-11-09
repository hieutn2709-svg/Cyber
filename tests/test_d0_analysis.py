import importlib.util


def test_endpoint_reasons_follow_filter_order_and_top2_is_gold_informed():
    assert importlib.util.find_spec('journal.scripts.analyze_d0_diagnostics') is not None
    from journal.scripts.analyze_d0_diagnostics import classify_endpoint
    snap=dict(content_start=1,content_end=5,global_token_offset=-1,width_cap=2,classes=['NONE','a','b'],spans=[
        dict(start=1,end=1,global_start=0,global_end=0,label='a',entity_score=.8,p_none=.2,
             probabilities=[.2,.5,.3],top_types=['a','b'],score_rank=1,selection_reason='retained'),
        dict(start=2,end=2,global_start=1,global_end=1,label='a',entity_score=.7,p_none=.3,
             probabilities=[.3,.4,.3],top_types=['a','b'],score_rank=2,selection_reason='cap')])
    assert classify_endpoint(snap,0,0,'a')['reason']=='retained'
    wrong=classify_endpoint(snap,0,0,'b')
    assert wrong['reason']=='wrong_type' and wrong['gold_type_rank']==2 and wrong['gold_in_top2_retained']
    cap=classify_endpoint(snap,1,1,'b')
    assert cap['reason']=='cap' and not cap['gold_in_top2_retained']
    assert classify_endpoint(snap,0,2,'a')['reason']=='width_cap'
    assert classify_endpoint(snap,0,5,'a')['reason']=='outside_content'
    assert classify_endpoint(snap,2,2,'a')['reason']=='proposal_missing'
