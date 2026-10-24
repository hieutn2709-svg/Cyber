import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from journal.scripts.run_e1_multi import verify_input, prepare_output, require_parity


class E1ArtifactTests(unittest.TestCase):
    def test_hash_verification_rejects_changes_and_missing_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'a.json';p.write_text('{}')
            manifest={'a.json':hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertEqual(verify_input(root,p,manifest),manifest['a.json'])
            p.write_text('[]')
            with self.assertRaises(ValueError):verify_input(root,p,manifest)
            with self.assertRaises(ValueError):verify_input(root,p,{})

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'new';prepare_output(p)
            with self.assertRaises(FileExistsError):prepare_output(p)

    def test_parity_requires_order_score_and_complete_documents(self):
        a=[dict(document_id='d',predicted_spans=[{'start':0,'entity_score':.5},{'start':1,'entity_score':.3}],predicted_relations=[])]
        require_parity(a,a)
        for change in ['order','score','doc']:
            with self.subTest(change=change):
                b=json.loads(json.dumps(a))
                if change=='order':b[0]['predicted_spans'].reverse()
                elif change=='score':b[0]['predicted_spans'][0]['entity_score']+=1e-8
                else:b=[]
                with self.assertRaises(ValueError):require_parity(a,b)
