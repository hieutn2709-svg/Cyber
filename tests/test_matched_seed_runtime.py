"""Seed extension must not silently alter baseline settings or provenance."""
import copy,json,tempfile,unittest
from pathlib import Path
from journal.scsp import matched_runtime as mr

class SeedRuntimeTests(unittest.TestCase):
 def test_only_approved_run_seed_changes_and_parent_is_unchanged(self):
  fn=getattr(mr,'seed_config',None)
  self.assertTrue(callable(fn),'approved seed configuration helper is missing')
  original=mr.read_json(mr.config_paths('roberta')[0]);before=copy.deepcopy(original)
  for seed in [43,44]:
   result=fn('roberta',seed)
   self.assertEqual({k:v for k,v in result.items() if k!='seed'},{k:v for k,v in before.items() if k!='seed'})
   self.assertEqual(result['seed'],seed)
  self.assertEqual(original,before)
  for seed in [42,45,True,'43']:
   with self.assertRaisesRegex(ValueError,'seed'):fn('roberta',seed)
 def test_seed_package_is_new_identity_with_unchanged_parent(self):
  fn=getattr(mr,'seed_package',None);self.assertTrue(callable(fn),'seed package helper is missing')
  parent={'contract':{'package':'roberta','seed':42,'fold':1,'split_seed':11800,'combined_config_sha256':'a'*64,'test_evaluated':False},'width_cap':6}
  original=copy.deepcopy(parent);p=fn(parent,43,'b'*64)
  self.assertEqual(parent,original);self.assertEqual(p['contract']['seed'],43)
  self.assertEqual(p['contract']['combined_config_sha256'],'b'*64)
  self.assertEqual(p['parent_seed42_package_sha256'],mr.digest(parent))
  self.assertEqual(p['contract']['split_seed'],11800)
 def test_baseline_evidence_requires_complete_consistent_seed42_dev(self):
  fn=getattr(mr,'validate_seed_baseline',None);self.assertTrue(callable(fn),'baseline validation helper is missing')
  pkg={'contract':{'seed':42,'dataset_sha256':'d','combined_config_sha256':'c'}}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   summary={'status':'dev_complete','test_evaluated':False,'seed':42,'fold':1,'dataset_sha256':'d','config_sha256':'c','best_epoch':12}
   cfg={'encoder_package':pkg,'seed':42,'epoch_budget':12,'mode':'dev','test_evaluated':False}
   for name,x in [('run_summary.json',summary),('training_run_config.json',cfg),('training_history.json',[{'epoch':i} for i in range(1,13)]),('validation_metrics.json',{})]: (p/name).write_text(json.dumps(x))
   fn(p,pkg)
   summary['seed']=43;(p/'run_summary.json').write_text(json.dumps(summary))
   with self.assertRaisesRegex(ValueError,'baseline'):fn(p,pkg)
   summary['seed']=42;(p/'run_summary.json').write_text(json.dumps(summary));(p/'training_history.json').write_text('[]')
   with self.assertRaisesRegex(ValueError,'baseline'):fn(p,pkg)
 def test_seed_cli_has_only_new_seeds_and_no_test_mode(self):
  import importlib.util
  name='journal.scripts.train_matched_seed'
  self.assertIsNotNone(importlib.util.find_spec(name),'seed CLI is missing')
  from journal.scripts.train_matched_seed import build_parser
  parser=build_parser();base=['--package','roberta','--bundle','/a','--preflight','/b','--baseline-run','/c','--output-dir','/d']
  self.assertEqual(parser.parse_args(base+['--seed','43']).seed,43)
  for extra in [['--seed','42'],['--seed','45'],['--seed','43','--mode','full']]:
   with self.assertRaises(SystemExit):parser.parse_args(base+extra)
if __name__=='__main__':unittest.main()
