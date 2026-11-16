"""Real fixture gradients/recovery and CLI boundaries for negative-policy opt-in."""
import argparse
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from journal.scsp import training_recovery as recovery, matched_runtime as mr
from journal.scsp.training import compute_training_window_loss
from journal.scripts import train_gate_a as gate, train_matched_encoder as encoder, train_matched_seed as repeat
from tests import test_training_pair_diagnostics_integration as fixture


class BoundaryPolicyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture.TrainingPairDiagnosticsIntegrationTests();self.f.setUp()

    def test_actual_loss_backward_reduces_conflicts_preserves_entity_and_positive_loss(self):
        f=self.f; f.base.relation_negative_ratio=200
        original=f.model();results=[]
        for policy in ('typed','boundary_replace'):
            model=copy.deepcopy(original);torch.manual_seed(100);stats={}
            result=compute_training_window_loss(model,f.window,f.inventory,width_cap=2,
                base_config=f.base,train_config=f.train,seed=42,device=torch.device('cpu'),
                pair_diagnostics=stats,relation_negative_policy=policy)
            result.total_loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            results.append((result,stats,torch.get_rng_state()))
        a,sa,ra=results[0];b,sb,rb=results[1]
        self.assertGreater(sa['sampled']['collision_groups'],0)
        self.assertEqual(sb['sampled']['collision_groups'],0)
        self.assertEqual(a.relation_positive_count,b.relation_positive_count)
        self.assertTrue(torch.equal(a.entity_loss,b.entity_loss))
        self.assertTrue(torch.equal(a.relation_type_loss,b.relation_type_loss))
        self.assertFalse(torch.equal(a.relation_existence_loss,b.relation_existence_loss))
        self.assertTrue(torch.equal(ra,rb))

    def test_metadata_and_full_guard_preserve_legacy_default(self):
        self.assertEqual(gate._negative_policy_metadata(argparse.Namespace(mode='dev')), {})
        self.assertEqual(gate._negative_policy_metadata(argparse.Namespace(mode='dev',relation_negative_policy='typed')), {})
        expected={'relation_negative_policy':'boundary_replace'}
        self.assertEqual(gate._negative_policy_metadata(argparse.Namespace(mode='dev',**expected)),expected)
        # Must reject before paths, outputs or model downloads are touched.
        with self.assertRaisesRegex(ValueError,'full/test'):
            gate.run(argparse.Namespace(mode='full',**expected))
        with self.assertRaisesRegex(ValueError,'negative policy'):
            gate._negative_policy_metadata(argparse.Namespace(mode='dev',relation_negative_policy='bad'))

    def test_real_policy_epoch_resume_exact_and_cross_policy_rejected(self):
        f=self.f;f.base.relation_negative_ratio=200
        def parts():
            m=f.model()
            return m,torch.optim.AdamW(m.parameters(),lr=.001),torch.amp.GradScaler('cuda',enabled=False)
        def epoch(p,n):
            return gate._train_epoch(*p,(f.window,),f.inventory,width_cap=2,base_config=f.base,
                training_config=f.train,seed=42,epoch=n,device=torch.device('cpu'),
                train_pair_diagnostics=True,relation_negative_policy='boundary_replace')
        identity={'epoch_budget':2,**gate._negative_policy_metadata(argparse.Namespace(mode='dev',relation_negative_policy='boundary_replace'))}
        active=parts();history=[{'epoch':1,'train':epoch(active,1)}]
        self.assertEqual(history[0]['train']['train_pair_diagnostics'][0]['sampled']['collision_groups'],0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'recovery.pt'
            recovery.save(path,identity=identity,model=active[0],optimizer=active[1],scaler=active[2],
                progress={'epoch':1,'history':history,'best_epoch':1},
                best_checkpoint={'epoch':1,'model_state_dict':copy.deepcopy(active[0].state_dict()),'relation_negative_policy':'boundary_replace'})
            expected=epoch(active,2);weights=copy.deepcopy(active[0].state_dict());rng=torch.get_rng_state()
            optimizer_state=copy.deepcopy(active[1].state_dict())
            resumed=parts();recovery.restore(recovery.load(path,identity),*resumed,Path(d)/'best.pt')
            self.assertEqual(epoch(resumed,2),expected)
            for k,v in weights.items():self.assertTrue(torch.equal(v,resumed[0].state_dict()[k]),k)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(optimizer_state["param_groups"],resumed[1].state_dict()["param_groups"])
            for key,state in optimizer_state["state"].items():
                for name,value in state.items():
                    self.assertTrue(torch.equal(value,resumed[1].state_dict()["state"][key][name]))
            for altered in ({'epoch_budget':2},{'epoch_budget':2,'relation_negative_policy':'typed'}):
                with self.assertRaisesRegex(ValueError,'identity'):recovery.load(path,altered)

    def test_both_matched_wrappers_forward_policy(self):
        for cli in (encoder,repeat):
            with self.subTest(cli=cli.__name__), tempfile.TemporaryDirectory() as d,ExitStack() as stack:
                package={'contract':{'seed':42}};(Path(d)/'run_summary.json').write_text('{}')
                args=['--package','roberta','--bundle',d,'--preflight',d,'--output-dir',str(Path(d)/'out'),'--relation-negative-policy','boundary_replace']
                args+=['--mode','dev'] if cli is encoder else ['--seed','43','--baseline-run',d]
                parsed=cli.build_parser().parse_args(args)
                for name in ('contract','validate_preflight'):stack.enter_context(patch.object(mr,name,return_value=package))
                for name in ('validate_cached_model','validate_previous_gate','validate_seed_baseline'):stack.enter_context(patch.object(mr,name))
                stack.enter_context(patch.object(gate,'run',side_effect=lambda a:a))
                delegated=cli.run(parsed)
                self.assertEqual(delegated.relation_negative_policy,'boundary_replace')
                self.assertEqual(delegated.mode,'dev')

    def test_native_smoke_runner_persists_policy_in_all_artifacts(self):
        # Only external data loading/preflight and pretrained model construction
        # are substituted. Training, inference, scoring and writes are real.
        f=self.f
        vspans=tuple(replace(s,document_id='v') for s in f.window.gold_spans)
        vrelations=(replace(f.window.gold_relations[0],source=vspans[0],target=vspans[1]),)
        validation=replace(f.window,doc_id='v',gold_spans=vspans,gold_relations=vrelations)
        partition=SimpleNamespace(train_document_ids=('d',),validation_document_ids=('v',),test_document_ids=())
        root=Path(__file__).resolve().parents[1]
        for policy in ('typed','boundary_replace'):
            with self.subTest(policy=policy),tempfile.TemporaryDirectory() as d,ExitStack() as stack:
                d=Path(d);dataset=d/'fixture.json';dataset.write_text('{}')
                inventory=d/'inventory.json';inventory.write_text(json.dumps({
                    'primary_entity_types':['intrusion-set','malware'],'auxiliary_entity_types':['tactic'],
                    'relation_types':['uses','targets']}))
                config=json.loads((root/'journal/configs/matched70_roberta.json').read_text())
                config.update(max_span_candidates=32,min_entity_score=0,relation_negative_ratio=200)
                config_path=d/'config.json';config_path.write_text(json.dumps(config))
                args=argparse.Namespace(mode='smoke',config=str(config_path),
                    training_config=str(root/'journal/configs/gate_a_training.json'),inventory=str(inventory),
                    manifest=str(d/'fixture_manifest.json'),dataset=str(dataset),fold=1,seed=42,
                    output_dir=str(d/'run'),device='cpu',overfit_epochs=2,encoder_package={'fixture':True},
                    epoch_recovery=True,resume=False,train_pair_diagnostics=True,relation_negative_policy=policy)
                stack.enter_context(patch('journal.scripts.run_gate_a.build_preflight',return_value={}))
                stack.enter_context(patch('journal.scripts.run_gate_a.write_preflight_artifacts'))
                stack.enter_context(patch('journal.scsp.splits.load_fold_partition',return_value=partition))
                stack.enter_context(patch('journal.scsp.data.load_clean_windows',return_value=(f.window,validation)))
                stack.enter_context(patch.object(gate,'_make_model',side_effect=lambda *a:f.model()))
                summary=gate.run(args)
                out=d/'run';checkpoint=torch.load(out/'best_model.pt',weights_only=False)
                state=torch.load(out/'recovery.pt',weights_only=False)
                artifacts=(summary,checkpoint,state['identity'],json.loads((out/'training_run_config.json').read_text()))
                for artifact in artifacts:
                    if policy=='typed':self.assertNotIn('relation_negative_policy',artifact)
                    else:self.assertEqual(artifact['relation_negative_policy'],policy)
                self.assertEqual(summary['status'],'smoke_complete')
                self.assertFalse(summary['test_evaluated'])
                self.assertEqual(state['progress']['epoch'],2)
                records=[r for e in state['progress']['history'] for r in e['train']['train_pair_diagnostics']]
                if policy=='boundary_replace':self.assertTrue(all(r['sampled']['collision_groups']==0 for r in records))
                self.assertFalse(list(out.glob('test_*')))


if __name__=='__main__':unittest.main()
