import copy
import json
import tempfile
import unittest
from pathlib import Path
from journal.scsp import matched_windows as mw
from journal.scsp import encoder_dataset as ed
from tests.test_scsp_encoder_dataset import WhitespaceTokenizer, inventory, load_source, source_row, clean_row


class MatchedWindowsTests(unittest.TestCase):
    def test_largest_safe_cut_fits_both_packages(self):
        t = WhitespaceTokenizer()
        spans = mw.choose_slices('a b c d e', 0, 9, (), (), (t,t), 5)
        self.assertEqual(spans, ((0,6),(6,9)))

    def test_cut_is_after_whitespace_as_in_approved_capacity_audit(self):
        class TrailingSpace(WhitespaceTokenizer):
            def document_offsets(self,text):
                offsets=super().document_offsets(text)
                return offsets+(((len(text)-1,len(text)),) if text[-1:].isspace() else ())
        self.assertEqual(mw.choose_slices('a b c d e',0,9,(),(),
            (TrailingSpace(),)*2,5),((0,4),(4,9)))

    def test_relation_cannot_be_separated_even_when_both_parts_fit(self):
        entities=(ed.SourceEntity('a',0,1,'malware'), ed.SourceEntity('e',8,9,'tool'))
        rel=(ed.SourceRelation('a','e','uses'),)
        with self.assertRaisesRegex(ValueError,'safe shared cut'):
            mw.choose_slices('a b c d e',0,9,entities,rel,(WhitespaceTokenizer(),)*2,5)

    def test_entity_cannot_be_split_at_internal_whitespace(self):
        entity=ed.SourceEntity('a',0,7,'malware')
        with self.assertRaisesRegex(ValueError,'safe shared cut'):
            mw.choose_slices('a b c d e',0,9,(entity,),(),(WhitespaceTokenizer(),)*2,5)

    def test_one_overflowing_tokenizer_requires_shared_cut(self):
        class Characters(WhitespaceTokenizer):
            def document_offsets(self,text):
                return tuple((i,i+1) for i,c in enumerate(text) if not c.isspace())
        self.assertEqual(mw.choose_slices('aa bb cc',0,8,(),(),
            (WhitespaceTokenizer(),Characters()),6),((0,6),(6,8)))

    def test_build_preserves_supervision_and_uses_native_coordinates(self):
        doc=load_source([source_row()])[0]
        row=clean_row()
        row['token_start_global']=100  # reference coordinates must not leak into native stream
        row['token_end_global']=103
        common=[dict(doc_seq_index=0,doc_id='420',window_index=0,original_window_index=0,
            subwindow_index=0,char_start=0,char_end=15)]
        got=mw.encode_common_windows((doc,),[row],common,WhitespaceTokenizer(),inventory())
        self.assertEqual((got.rows[0]['token_start_global'],got.rows[0]['token_end_global']),(0,3))
        self.assertEqual(got.rows[0]['entity_spans'][1]['char_start'],11)
        self.assertEqual(got.rows[0]['entity_spans'][1]['role'],'ROLE_2')
        self.assertEqual(got.rows[0]['relations'],[{'source_id':'e1','target_id':'e2','type':'uses'}])
        self.assertEqual(got.rows[0]['offset_mapping'],[[0,0],[0,5],[6,10],[11,15],[0,0]])

    def test_changed_source_coordinates_rejected(self):
        doc=load_source([source_row()])[0]
        row=clean_row(); row['entity_spans'][0]['char_end']=4
        with self.assertRaisesRegex(ValueError,'source entity'):
            mw.retained_entities(doc,row)

    def test_membership_checks_roles_and_coordinates_not_only_counts(self):
        rows=[clean_row()]; changed=copy.deepcopy(rows)
        changed[0]['entity_spans'][0]['char_end']=4
        with self.assertRaisesRegex(ValueError,'membership'):
            mw.validate_membership(rows,changed,split=False)

    def test_pair_serialization_failure_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pair'
            with self.assertRaises(ValueError):
                mw.write_atomic_bundle(p,{'roberta/dataset.json':[], 'securebert/dataset.json':float('nan')})
            self.assertFalse(p.exists())
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_pair_writer_never_overwrites_existing_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pair'
            mw.write_atomic_bundle(p,{'roberta/dataset.json':[1], 'securebert/dataset.json':[2]})
            with self.assertRaises(FileExistsError):
                mw.write_atomic_bundle(p,{'roberta/dataset.json':[3]})
            self.assertEqual(json.loads((p/'roberta/dataset.json').read_text()),[1])

class MatchedCliTests(unittest.TestCase):
    def test_invalid_frozen_hash_fails_before_tokenizer_load_and_creates_nothing(self):
        from journal.scripts import build_matched_encoder_datasets as cli
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'fake.json').write_text('[]')
            args=cli.build_parser().parse_args(['--source',str(p/'fake.json'),
                '--frozen-dataset',str(p/'fake.json'),'--parity-manifest',str(p/'fake.json'),
                '--output-dir',str(p/'out')])
            with self.assertRaisesRegex(ValueError,'frozen dataset SHA'):
                cli.run(args)
            self.assertFalse((p/'out').exists())

    def test_cli_help_works_without_runtime_dependencies(self):
        import subprocess, sys
        r=subprocess.run([sys.executable,'-m','journal.scripts.build_matched_encoder_datasets','--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
