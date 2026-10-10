import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from journal.scsp import matched_runtime as mr

ROOT=Path(__file__).resolve().parents[1]

class MatchedRuntimeTests(unittest.TestCase):
    def test_downstream_change_rejected_and_paired_package_accepted(self):
        baseline=json.loads((ROOT/'journal/configs/gate_a_plain_spanpair.json').read_text())
        candidate={**baseline,'encoder_model':'ehsanaghaei/SecureBERT',
            'encoder_revision':'3a47918dd874e5c769efd152b51c5756a953fb67'}
        mr.validate_settings(candidate,'securebert')
        candidate['max_span_candidates']=256
        with self.assertRaisesRegex(ValueError,'max_span_candidates'):
            mr.validate_settings(candidate,'securebert')

    def test_vocab_positions_and_output_shape_are_checked(self):
        cfg=SimpleNamespace(model_type='roberta',vocab_size=100,hidden_size=8,
            max_position_embeddings=514,pad_token_id=1,bos_token_id=0,eos_token_id=2)
        tokens={'vocabulary_size':100,'special_token_ids':{'bos':0,'pad':1,'eos':2,'unk':3}}
        mr.validate_components(cfg,100,tokens,(1,512,8),512)
        for embeddings,shape in [(99,(1,512,8)),(100,(1,512,7))]:
            with self.assertRaises(ValueError):
                mr.validate_components(cfg,embeddings,tokens,shape,512)
        cfg.max_position_embeddings=512
        with self.assertRaisesRegex(ValueError,'position'):
            mr.validate_components(cfg,100,tokens,(1,512,8),512)

    def test_approved_paired_dev_allowed_but_full_test_and_unknown_package_rejected(self):
        for package in ('roberta','securebert'):
            mr.validate_mode(package,'dev')
            for mode in ('full','test'):
                with self.assertRaisesRegex(ValueError,'not authorized'):
                    mr.validate_mode(package,mode)
        with self.assertRaisesRegex(ValueError,'not authorized'):
            mr.validate_mode('unapproved_encoder','dev')

    def test_checkpoint_other_package_rejected_before_model_state_load(self):
        from journal.scripts import train_gate_a as gate
        class Model:
            loaded=False
            def load_state_dict(self,state): self.loaded=True
        m=Model(); package={'tokenizer':{'id':'x','revision':'a'*40}}
        changed=copy.deepcopy(package);changed['tokenizer']['revision']='b'*40
        with self.assertRaisesRegex(ValueError,'checkpoint encoder package'):
            gate.load_checkpoint_state(m,{'encoder_package':changed,'model_state_dict':{}},package)
        self.assertFalse(m.loaded)
        gate.load_checkpoint_state(m,{'encoder_package':package,'model_state_dict':{}},package)
        self.assertTrue(m.loaded)

    def test_original_gate_checkpoint_without_package_still_loads(self):
        from journal.scripts import train_gate_a as gate
        class Model:
            def load_state_dict(self,state): self.state=state
        m=Model();gate.load_checkpoint_state(m,{'model_state_dict':{'old':1}},None)
        self.assertEqual(m.state,{'old':1})

    def test_wrong_bundle_hash_rejected_before_any_runtime_import(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'roberta').mkdir()
            (p/'roberta/encoder_dataset_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest'):
                mr.validate_bundle(p,'roberta')

    def test_help_and_parser_never_offer_test_mode(self):
        from journal.scripts import train_matched_encoder as cli
        p=cli.build_parser()
        self.assertNotIn('full',p.format_help())
        with self.assertRaises(SystemExit):
            p.parse_args(['--package','roberta','--bundle','/tmp/a','--preflight','/tmp/b',
                '--output-dir','/tmp/c','--mode','full'])

    def test_smoke_and_dev_require_same_package_successful_previous_gates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);pkg={'contract':{'package':'roberta'}}
            (p/'training_run_config.json').write_text(json.dumps({'encoder_package':pkg}))
            (p/'run_summary.json').write_text(json.dumps({'status':'overfit_complete','test_evaluated':False,
                'initial_loss':4.0,'final_loss':5.0}))
            with self.assertRaisesRegex(ValueError,'decreasing'):
                mr.validate_previous_gate('smoke',p,None,pkg)
            (p/'run_summary.json').write_text(json.dumps({'status':'overfit_complete','test_evaluated':False,
                'initial_loss':4.0,'final_loss':1.0}))
            mr.validate_previous_gate('smoke',p,None,pkg)
            with self.assertRaisesRegex(ValueError,'smoke'):
                mr.validate_previous_gate('dev',p,None,pkg)

    def test_preflight_help_without_model_download(self):
        import subprocess,sys
        r=subprocess.run([sys.executable,'-m','journal.scripts.preflight_matched_encoders','--help'],text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_model_loader_explicitly_disables_remote_code(self):
        import sys
        from unittest.mock import patch
        from journal.scsp.runtime_model import GateASpanPairModel
        calls=[]
        class StopAfterLoad(Exception): pass
        def loader(*a,**kw):
            calls.append(kw)
            raise StopAfterLoad()
        with patch.dict(sys.modules,{'transformers':SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=loader))}):
            with self.assertRaises(StopAfterLoad):
                GateASpanPairModel.from_pretrained(model_name='x',revision='a'*40,num_entity_classes=2,
                    num_relation_types=1,max_width=3,width_embedding_dim=2,context_dim=2,
                    distance_embedding_dim=2,max_distance=4)
        self.assertIs(calls[0].get('trust_remote_code'),False)

    def test_resolved_snapshot_rejects_wrong_revision_or_non_snapshot(self):
        mr.validate_snapshot_path('/cache/models--x/snapshots/'+'a'*40+'/config.json','a'*40)
        for path in ['/cache/models--x/snapshots/'+'b'*40+'/config.json','/tmp/config.json']:
            with self.assertRaisesRegex(ValueError,'snapshot revision'):
                mr.validate_snapshot_path(path,'a'*40)

    def test_model_cache_corruption_rejected_against_preflight_hash(self):
        import tempfile,hashlib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.safetensors';p.write_bytes(b'complete-weights')
            package={'resolved_model_files_sha256':{'model.safetensors':hashlib.sha256(b'complete-weights').hexdigest()}}
            mr.validate_model_files(package,{'model.safetensors':p})
            p.write_bytes(b'complete')
            with self.assertRaisesRegex(ValueError,'cached model hash'):
                mr.validate_model_files(package,{'model.safetensors':p})
