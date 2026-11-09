from __future__ import annotations
import copy
import json
import random
import tempfile
from dataclasses import replace
from pathlib import Path
import unittest
import torch
from journal.scsp.training import compute_training_window_loss
from journal.scripts import train_gate_a as gate
from tests import test_scsp_training_core as fixtures

class TrainingPairDiagnosticsIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1729)
        self.fixture = fixtures.TrainingCoreTests()
        self.window = self.fixture._window()
        self.inventory = self.fixture._inventory()
        self.base = self.fixture.BaseConfig()
        self.base.max_span_candidates = 32
        self.base.max_relation_token_distance = 96
        self.train = self.fixture.TrainConfig()
        self.train.fp16 = False
        self.train.gradient_accumulation_steps = 2
        self.train.max_grad_norm = 1.0

    def model(self):
        model = self.fixture._model()
        model.encoder.embedding = torch.nn.Sequential(model.encoder.embedding, torch.nn.Dropout(.2))
        with torch.no_grad():
            model.heads.entity_head.classifier.weight.zero_()
            model.heads.entity_head.classifier.bias.copy_(torch.tensor([-4., -4., -4., 4.]))
        return model

    def test_loss_gradients_rng_optimizer_step_exact(self):
        self.base.relation_negative_ratio = 200  # Include the contradictory labels in the actual loss.
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                original = self.model()
                states = []
                for enabled in (False, True):
                    model = copy.deepcopy(original)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                    torch.manual_seed(100)
                    random.seed(100)
                    record = {}
                    result = compute_training_window_loss(model, self.window, self.inventory,
                        width_cap=2, base_config=self.base, train_config=self.train, seed=seed,
                        device=torch.device('cpu'), **({'pair_diagnostics': record} if enabled else {}))
                    losses = [getattr(result, k).detach().clone() for k in
                        ('total_loss','entity_loss','relation_existence_loss','relation_type_loss')]
                    result.total_loss.backward()
                    grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
                    optimizer.step()
                    states.append((losses, grads, copy.deepcopy(model.state_dict()), torch.get_rng_state(), random.getstate()))
                    if enabled:
                        self.assertGreater(record['eligible']['collision_groups'], 0)
                        self.assertGreater(record['sampled']['collision_groups'], 0)
                        self.assertEqual(record['sampled']['positive_pairs'], result.relation_positive_count)
                        self.assertEqual(sum(record['sampled'][k] for k in ('positive_pairs','negative_pairs')), result.relation_pair_count)
                for left, right in zip(states[0][:2], states[1][:2]):
                    for a,b in zip(left,right):
                        if a is None: self.assertIsNone(b)
                        else: self.assertTrue(torch.equal(a,b))
                for key in states[0][2]: self.assertTrue(torch.equal(states[0][2][key],states[1][2][key]),key)
                self.assertTrue(torch.equal(states[0][3],states[1][3]))
                self.assertEqual(states[0][4],states[1][4])

    def test_epoch_metrics_history_optimizer_parity(self):
        original = self.model()
        runs = []
        windows = (self.window, replace(self.window, window_index=1, token_start_global=20, token_end_global=26))
        for enabled in (False, True):
            model = copy.deepcopy(original)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            torch.manual_seed(123)
            metrics = gate._train_epoch(model, optimizer, scaler, windows, self.inventory,
                width_cap=2, base_config=self.base, training_config=self.train, seed=42, epoch=2,
                device=torch.device('cpu'), train_pair_diagnostics=enabled)
            runs.append((metrics, copy.deepcopy(model.state_dict()), torch.get_rng_state()))
        records = runs[1][0].pop('train_pair_diagnostics')
        self.assertEqual(runs[0][0],runs[1][0])
        for key in runs[0][1]: self.assertTrue(torch.equal(runs[0][1][key],runs[1][1][key]),key)
        self.assertTrue(torch.equal(runs[0][2],runs[1][2]))
        self.assertEqual(len(records),2)
        self.assertEqual({r['window_index'] for r in records},{0,1})
        for step,r in enumerate(records,1):
            self.assertEqual((r['epoch'],r['step'],r['sample_seed'],r['document_id']), (2,step,200042+step,'d'))
            self.assertEqual(r['global_token_offset'],19 if r['window_index'] else -1)
        row = gate.build_history_row(epoch=2,seconds=0.,train_metrics={'train_pair_diagnostics':records},
            selected={},candidate_diagnostics={k:{'recall':0.} for k in
            ('span_proposal','span_post_pruning_typed','pair_pre_distance','pair_post_distance')})
        self.assertEqual(json.loads(json.dumps(row))['train']['train_pair_diagnostics'],records)

    def test_overfit_history(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        with tempfile.TemporaryDirectory() as directory:
            gate._run_overfit(model,optimizer,torch.amp.GradScaler('cuda',enabled=False),
                (self.window,),self.inventory,epochs=1,width_cap=2,base_config=self.base,
                training_config=self.train,seed=42,device=torch.device('cpu'),
                output_dir=Path(directory),train_pair_diagnostics=True)
            history=json.loads((Path(directory)/'overfit_history.json').read_text())
            self.assertEqual(len(history[0]['train_pair_diagnostics']),1)

    def test_cli_default_off_opt_in(self):
        from journal.scripts import train_matched_encoder as encoder, train_matched_seed as seed
        cases=[(gate._parser(),['--dataset','d','--output-dir','o']),
            (encoder.build_parser(),['--package','roberta','--mode','dev','--bundle','b','--preflight','p','--output-dir','o']),
            (seed.build_parser(),['--package','roberta','--seed','43','--bundle','b','--preflight','p','--baseline-run','r','--output-dir','o'])]
        for parser,args in cases:
            self.assertFalse(parser.parse_args(args).train_pair_diagnostics)
            self.assertTrue(parser.parse_args(args+['--train-pair-diagnostics']).train_pair_diagnostics)

if __name__=='__main__': unittest.main()
