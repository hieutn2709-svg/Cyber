"""Regression coverage for the E2 wrapper/recovery review minor."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from journal.scripts import train_gate_a as gate, train_matched_encoder as encoder, train_matched_seed as repeat
from journal.scsp import matched_runtime as mr, training_recovery as recovery
from tests import test_training_pair_diagnostics_integration as fixture

class DiagnosticBoundaryTests(unittest.TestCase):
    def delegate(self, cli, enabled):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            package={'contract':{'seed':42}}
            (Path(d)/'run_summary.json').write_text('{}')
            args=['--package','roberta','--bundle',d,'--preflight',d,'--output-dir',str(Path(d)/'out')]
            args+=['--mode','dev'] if cli is encoder else ['--seed','43','--baseline-run',d]
            if enabled: args+=['--train-pair-diagnostics']
            parsed=cli.build_parser().parse_args(args)
            for name in ('contract','validate_preflight'):
                stack.enter_context(patch.object(mr,name,return_value=package))
            for name in ('validate_cached_model','validate_previous_gate','validate_seed_baseline'):
                stack.enter_context(patch.object(mr,name))
            called=stack.enter_context(patch.object(gate,'run',side_effect=lambda a:a))
            delegated=cli.run(parsed)
            self.assertEqual(delegated.train_pair_diagnostics,enabled)
            self.assertEqual(delegated.mode,'dev')
            self.assertFalse(gate.mode_evaluates_test(delegated.mode))
            called.assert_called_once()

    def test_encoder_wrapper_forwards_observation_flag(self):
        for enabled in (False,True): self.delegate(encoder,enabled)

    def test_repeat_wrapper_forwards_observation_flag(self):
        for enabled in (False,True): self.delegate(repeat,enabled)

    def test_enabled_history_survives_recovery_and_next_epoch_exactly(self):
        f=fixture.TrainingPairDiagnosticsIntegrationTests();f.setUp()
        def components():
            model=f.model()
            return model,torch.optim.AdamW(model.parameters(),lr=.001),torch.amp.GradScaler('cuda',enabled=False)
        def epoch(parts,n):
            return gate._train_epoch(*parts,(f.window,),f.inventory,width_cap=2,base_config=f.base,
                training_config=f.train,seed=42,epoch=n,device=torch.device('cpu'),train_pair_diagnostics=True)
        identity={'epoch_budget':2,'train_pair_diagnostics':True}
        parts=components();first=epoch(parts,1)
        history=[{'epoch':1,'train':first}]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'recovery.pt'
            recovery.save(path,identity=identity,model=parts[0],optimizer=parts[1],scaler=parts[2],
                progress={'epoch':1,'history':history,'best_epoch':1},
                best_checkpoint={'epoch':1,'model_state_dict':copy.deepcopy(parts[0].state_dict())})
            expected=epoch(parts,2);weights=copy.deepcopy(parts[0].state_dict());rng=torch.get_rng_state()
            resumed=components()
            progress=recovery.restore(recovery.load(path,identity),*resumed,Path(d)/'best.pt')
            self.assertEqual(progress['history'],history)
            self.assertEqual(epoch(resumed,2),expected)
            for k,v in weights.items():self.assertTrue(torch.equal(v,resumed[0].state_dict()[k]),k)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for changed in ({'epoch_budget':2},{'epoch_budget':2,'train_pair_diagnostics':False}):
                with self.assertRaisesRegex(ValueError,'identity'):recovery.load(path,changed)

if __name__=='__main__': unittest.main()
