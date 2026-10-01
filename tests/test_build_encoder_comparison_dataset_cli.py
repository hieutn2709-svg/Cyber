import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from journal.scripts import build_encoder_comparison_dataset as cli
from journal.scsp import encoder_dataset_artifacts as art
from tests.test_scsp_encoder_dataset import source_row
from tests.test_scsp_encoder_dataset_artifacts import passed_manifest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'journal/scripts/build_encoder_comparison_dataset.py'


class BuildEncoderComparisonDatasetCliTests(unittest.TestCase):
    def test_help_is_lazy_and_no_model_or_prediction_modes_are_accepted(self):
        result=subprocess.run([sys.executable,str(SCRIPT),'--help'],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('roberta-parity',result.stdout)
        self.assertIn('securebert',result.stdout)
        code='import sys; import journal.scripts.build_encoder_comparison_dataset; assert "transformers" not in sys.modules; assert "torch" not in sys.modules'
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for flag in ('--checkpoint','--threshold','--fold'):
            result=subprocess.run([sys.executable,str(SCRIPT),flag,'sentinel'],cwd=ROOT,capture_output=True)
            self.assertNotEqual(result.returncode,0)

    def test_mutable_or_wrong_revision_is_rejected(self):
        for revision in ('main','a'*40):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                cli.validate_cli_contract(mode='securebert',model_id=art.SECUREBERT_ID,
                    revision=revision,roberta_parity_manifest=None)

    def test_securebert_requires_manifest_hash_and_passed_source_chain(self):
        args=dict(mode='securebert',model_id=art.SECUREBERT_ID,revision=art.SECUREBERT_REVISION)
        with self.assertRaisesRegex(ValueError,'RoBERTa parity'):
            cli.validate_cli_contract(**args,roberta_parity_manifest=None)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'parity.json'; p.write_bytes(art.json_bytes(passed_manifest()))
            with self.assertRaisesRegex(ValueError,'manifest.*SHA'):
                cli.validate_cli_contract(**args,roberta_parity_manifest=p,roberta_parity_sha256='b'*64)
            with self.assertRaisesRegex(ValueError,'source_sha256'):
                cli.validate_cli_contract(**args,roberta_parity_manifest=p,
                    roberta_parity_sha256=art.file_sha256(p),source_sha256='c'*64)
            got=cli.validate_cli_contract(**args,roberta_parity_manifest=p,
                roberta_parity_sha256=art.file_sha256(p),source_sha256='a'*64)
            self.assertEqual(got['parity_status'],'passed')

    def test_bad_frozen_hash_fails_before_loading_tokenizer_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'source.json'; source.write_text(json.dumps([source_row()]))
            frozen=root/'frozen.json'; frozen.write_text('[]'); out=root/'out'
            args=cli.build_parser().parse_args(['--mode','roberta-parity','--source',str(source),
                '--frozen-dataset',str(frozen),'--output-dir',str(out)])
            with patch.object(cli,'load_fast_tokenizer',side_effect=AssertionError('must not load')):
                with self.assertRaisesRegex(ValueError,'frozen.*SHA'):
                    cli.run(args)
            self.assertFalse(out.exists())

    def test_output_inside_git_repository_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            cli.validate_output_path(ROOT/'journal'/'private_data')


class SourceExclusionPolicyTests(unittest.TestCase):
    def test_policy_is_bound_to_source_and_forbids_excluding_frozen_entities(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'policy.json'
            p.write_text(json.dumps({'schema_version':1,'source_sha256':'a'*64,
                'regions':[{'doc_id':'420','entity_id':'e1','reason':'unlabeled_unreferenced_region'}]}))
            with self.assertRaisesRegex(ValueError,'source_sha256'):
                cli.load_source_exclusion_policy(p,'b'*64,[])
            with self.assertRaisesRegex(ValueError,'frozen'):
                cli.load_source_exclusion_policy(p,'a'*64,[{'doc_id':'420','entity_spans':[{'entity_id':'e1'}]}])
            self.assertEqual(cli.load_source_exclusion_policy(p,'a'*64,[]),frozenset({('420','e1')}))
