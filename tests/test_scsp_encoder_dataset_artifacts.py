import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from journal.scsp import encoder_dataset_artifacts as art
from journal.scsp import encoder_dataset as ed
from tests.test_scsp_encoder_dataset import clean_row, inventory


def manifest_fixture():
    rows = (clean_row(),)
    result = ed.BuildResult(rows, (), ed.audit_encoder_build(rows, (), inventory()))
    return art.build_encoder_dataset_manifest(source_sha256='a'*64,
        frozen_dataset_sha256=art.FROZEN_SHA256,
        encoder_id=art.ROBERTA_ID,encoder_revision=art.ROBERTA_REVISION,
        tokenizer_class='SyntheticTokenizer',tokenizer_files={},build_result=result,
        dataset_sha256=hashlib.sha256(art.json_bytes(rows)).hexdigest(),
        semantic_sha256_value=ed.semantic_sha256(rows),builder_git_commit='b'*40)


def passed_manifest():
    m = manifest_fixture(); m.update(art.FROZEN_COUNTS)
    m['parity_status']='passed'; m['frozen_semantic_sha256']=m['semantic_sha256']
    return m


class EncoderDatasetArtifactTests(unittest.TestCase):
    def test_roberta_parity_requires_counts_semantics_and_identity(self):
        manifest = passed_manifest(); art.validate_roberta_parity(manifest)
        for field, value in [('relation_count',563),('frozen_dataset_sha256','a'*64),
                             ('frozen_semantic_sha256','f'*64),('parity_status','failed')]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,field):
                art.validate_roberta_parity({**manifest,field:value})
        manifest['tokenizer']['revision']='a'*40
        with self.assertRaisesRegex(ValueError,'tokenizer'):
            art.validate_roberta_parity(manifest)

    def test_writer_leaves_no_partial_bundle_when_parity_fails(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'bundle'; different=clean_row(); different['input_ids'][1]=999
            with self.assertRaisesRegex(ValueError,'semantic parity'):
                art.write_encoder_dataset_bundle(output_dir=out,rows=(clean_row(),),
                    manifest=manifest_fixture(),audit={},exclusions=(),
                    expected_rows=(different,),require_roberta_parity=True)
            self.assertFalse(out.exists())
            self.assertEqual(list(Path(td).iterdir()),[])

    def test_writer_verifies_hash_and_recomputed_counts_before_creating_directory(self):
        with tempfile.TemporaryDirectory() as td:
            for field,value in [('dataset_sha256','f'*64),('relation_count',99)]:
                out=Path(td)/field
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):
                    art.write_encoder_dataset_bundle(output_dir=out,rows=(clean_row(),),
                        manifest={**manifest_fixture(),field:value},audit={},exclusions=(),
                        inventory=inventory())
                self.assertFalse(out.exists())

    def test_writer_is_deterministic_and_never_overwrites_existing_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'bundle'; m=manifest_fixture()
            art.write_encoder_dataset_bundle(output_dir=out,rows=(clean_row(),),
                manifest=m,audit={},exclusions=(),inventory=inventory())
            self.assertEqual(set(p.name for p in out.iterdir()), {'dataset.json',
                'encoder_dataset_manifest.json','tokenizer_audit.json','alignment_exclusions.json'})
            self.assertEqual(hashlib.sha256((out/'dataset.json').read_bytes()).hexdigest(),m['dataset_sha256'])
            with self.assertRaises(FileExistsError):
                art.write_encoder_dataset_bundle(output_dir=out,rows=(clean_row(),),
                    manifest=m,audit={},exclusions=(),inventory=inventory())
