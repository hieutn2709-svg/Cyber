import copy
import math
import unittest

from journal.scsp.multi_type_decode import decode_snapshots


def fixture():
    spans = [dict(start=1,end=1,label='tool',entity_score=.8,selection_reason='retained',probabilities=[.2,.5,.3]),
             dict(start=4,end=4,label='malware',entity_score=.9,selection_reason='retained',probabilities=[.1,.4,.5]),
             dict(start=8,end=8,label='tool',entity_score=.01,selection_reason='threshold',probabilities=[.99,.006,.004])]
    snapshot = dict(document_id='d',window_index=0,global_token_offset=-1,classes=['NONE','tool','malware'],spans=spans,
                    generated_pairs=[dict(source_index=0,target_index=1,token_distance=2)])
    pair=dict(document_id='d',window_index=0,source=dict(start=0,end=0,label='tool'),target=dict(start=3,end=3,label='malware'),token_distance=2,existence_logit=3.,relation_type_logits=[1.,1.])
    return [snapshot],[pair]


class MultiTypeDecodeTests(unittest.TestCase):
    def test_top1_exact_offset_score_and_tie_order(self):
        s,p=fixture();out=decode_snapshots(s,p,['uses','targets'],k=1)[0]
        self.assertEqual([(x['start'],x['label']) for x in out['predicted_spans']],[(0,'tool'),(3,'malware')])
        self.assertTrue(all(x['proposal_source']=='predicted' for x in out['predicted_spans']))
        r=out['predicted_relations'][0]
        self.assertEqual(r['label'],'uses');self.assertEqual(r['relation_score'],1/(1+math.exp(-3)))
        self.assertEqual(set(out),{'document_id','predicted_spans','predicted_relations'})

    def test_top2_adds_true_positive_and_false_positives(self):
        s,p=fixture();a=decode_snapshots(s,p,['uses','targets'],k=1)[0];b=decode_snapshots(s,p,['uses','targets'],k=2)[0]
        def keys(o):return {(r['source']['label'],r['label'],r['target']['label']) for r in o['predicted_relations']}
        gold={('malware','uses','malware')}
        self.assertEqual((len(keys(a)&gold),len(keys(a)-gold)),(0,1))
        self.assertEqual((len(keys(b)&gold),len(keys(b)-gold)),(1,3))
        self.assertTrue(keys(a)<=keys(b));self.assertEqual(len(b['predicted_spans']),4)

    def test_none_excluded_and_entity_tie_uses_inventory(self):
        s,p=fixture();s[0]['spans'][0].update(probabilities=[.8,.1,.1],entity_score=.2)
        out=decode_snapshots(s,p,['uses','targets'],k=1)[0]
        self.assertEqual(out['predicted_spans'][0]['label'],'tool')

    def test_threshold_and_empty_pairs(self):
        s,p=fixture();p[0]['existence_logit']=-1000
        self.assertEqual(decode_snapshots(s,p,['uses','targets'],k=2)[0]['predicted_relations'],[])
        s[0]['generated_pairs']=[]
        self.assertEqual(decode_snapshots(s,[],['uses','targets'],k=2)[0]['predicted_relations'],[])

    def test_duplicate_windows_merge_max_score(self):
        s,p=fixture();t=copy.deepcopy(s[0]);t['window_index']=1;t['spans'][0]['entity_score']=.85
        t['spans'][0]['probabilities']=[.15,.55,.3]
        q=copy.deepcopy(p[0]);q['window_index']=1;q['existence_logit']=4
        out=decode_snapshots(s+[t],p+[q],['uses','targets'],k=2)[0]
        self.assertEqual(len(out['predicted_relations']),4)
        self.assertTrue(all(r['relation_score']==1/(1+math.exp(-4)) for r in out['predicted_relations']))
        self.assertEqual(out['predicted_spans'][0]['entity_score'],.85)

    def test_rejects_missing_duplicate_unexpected_pairs(self):
        for mode in ['missing','duplicate','unexpected']:
            with self.subTest(mode=mode):
                s,p=fixture()
                if mode=='missing':p=[]
                elif mode=='duplicate':p=p+p
                else:p[0]['source']['start']=1
                with self.assertRaises(ValueError):decode_snapshots(s,p,['uses','targets'],k=2)

    def test_rejects_invalid_probabilities_logits_and_pair(self):
        for mode in ['nan_prob','negative_prob','bad_sum','nan_logit','same_boundary','missing_posterior','wrong_top1','duplicate_window','bad_distance']:
            with self.subTest(mode=mode):
                s,p=fixture()
                if mode=='nan_prob':s[0]['spans'][0]['probabilities'][1]=float('nan')
                elif mode=='negative_prob':s[0]['spans'][0]['probabilities']=[.2,-.1,.9]
                elif mode=='bad_sum':s[0]['spans'][0]['probabilities']=[.2,.2,.2]
                elif mode=='nan_logit':p[0]['existence_logit']=float('nan')
                elif mode=='same_boundary':s[0]['generated_pairs'][0]['target_index']=0
                elif mode=='missing_posterior':del s[0]['spans'][0]['probabilities']
                elif mode=='wrong_top1':s[0]['spans'][0]['label']='malware'
                elif mode=='duplicate_window':s=s+s
                else:p[0]['token_distance']=0
                with self.assertRaises((ValueError,KeyError)):decode_snapshots(s,p,['uses','targets'],k=2)

    def test_rejects_invalid_options(self):
        for k,threshold in [(0,.9),(3,.9),(2,float('nan')),(2,1.1)]:
            with self.subTest(k=k,threshold=threshold):
                s,p=fixture()
                with self.assertRaises(ValueError):decode_snapshots(s,p,['uses','targets'],k=k,threshold=threshold)
