"""Recovery must reproduce uninterrupted AdamW/dropout training exactly."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from journal.scsp import training_recovery as recovery


class TrainingRecoveryTests(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        self.identity = {"mode": "dev", "epoch_budget": 4, "package": "roberta"}

    def make_training(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.4), torch.nn.Linear(4, 1))
        return model, torch.optim.AdamW(model.parameters(), lr=.02), torch.amp.GradScaler("cuda", enabled=False)

    def step(self, model, optimizer):
        model.train()
        optimizer.zero_grad()
        x = torch.randn(5, 3) * (random.random() + np.random.random())
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()

    def save(self, path, model, optimizer, scaler):
        best = {"model_state_dict": copy.deepcopy(model.state_dict()), "epoch": 1}
        recovery.save(path, identity=self.identity, model=model, optimizer=optimizer, scaler=scaler,
                      progress={"epoch": 2, "history": [{"epoch": 1}, {"epoch": 2}],
                                "best_epoch": 1, "best_selection": {"relation_f1": .2},
                                "best_threshold": .5, "stale_epochs": 1}, best_checkpoint=best)

    def test_interruption_restores_exact_optimizer_rng_and_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"recovery.pt"
            model, opt, scaler = self.make_training()
            for _ in range(2):
                self.step(model, opt)
            self.save(path, model, opt, scaler)
            for _ in range(2):
                self.step(model, opt)
            expected = copy.deepcopy(model.state_dict())
            expected_opt = copy.deepcopy(opt.state_dict())
            next_random = (random.random(), np.random.random(), torch.rand(3))
            resumed, resumed_opt, resumed_scaler = self.make_training()
            state = recovery.load(path, self.identity)
            progress = recovery.restore(state, resumed, resumed_opt, resumed_scaler, Path(d)/"best_model.pt")
            self.assertEqual(progress["epoch"], 2)
            self.assertEqual(torch.load(Path(d)/"best_model.pt", weights_only=False)["epoch"], 1)
            for _ in range(2):
                self.step(resumed, resumed_opt)
            for key, value in expected.items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            for key, value in expected_opt["state"].items():
                for name, tensor in value.items():
                    self.assertTrue(torch.equal(tensor, resumed_opt.state_dict()["state"][key][name]))
            self.assertEqual(next_random[0], random.random())
            self.assertEqual(next_random[1], np.random.random())
            self.assertTrue(torch.equal(next_random[2], torch.rand(3)))

    def test_failed_atomic_write_preserves_last_complete_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"recovery.pt"
            model, opt, scaler = self.make_training()
            self.save(path, model, opt, scaler)
            original = path.read_bytes()
            with patch.object(recovery.os, "replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    self.save(path, model, opt, scaler)
            self.assertEqual(original, path.read_bytes())
            self.assertEqual(list(Path(d).iterdir()), [path])

    def test_changed_identity_and_invalid_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"recovery.pt"
            self.save(path, *self.make_training())
            for key, value in [("mode", "full"), ("package", "securebert"), ("epoch_budget", 5)]:
                with self.assertRaisesRegex(ValueError, "identity"):
                    recovery.load(path, {**self.identity, key: value})
            state = torch.load(path, weights_only=False)
            state["progress"]["epoch"] = 5
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "epoch"):
                recovery.load(path, self.identity)

    def test_legacy_weights_only_checkpoint_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"best_model.pt"
            torch.save({"model_state_dict": {}, "epoch": 11}, path)
            with self.assertRaisesRegex(ValueError, "recovery"):
                recovery.load(path, self.identity)

    def test_cli_resume_requires_incomplete_run_with_recovery(self):
        from types import SimpleNamespace
        from journal.scripts import train_matched_encoder as cli
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"run"
            args = SimpleNamespace(output_dir=path, resume=True, mode="dev")
            with self.assertRaises(ValueError):
                cli.validate_output(args)
            path.mkdir()
            (path/"recovery.pt").write_bytes(b"identity checked by loader")
            cli.validate_output(args)
            args.resume = False
            with self.assertRaises(FileExistsError):
                cli.validate_output(args)
            args.resume = True
            (path/"run_summary.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "complete"):
                cli.validate_output(args)

    def test_second_writer_is_rejected_and_lock_released(self):
        with tempfile.TemporaryDirectory() as d:
            with recovery.run_lock(Path(d)):
                with self.assertRaisesRegex(ValueError, "active"):
                    with recovery.run_lock(Path(d)):
                        self.fail("two writers acquired the same run")
            with recovery.run_lock(Path(d)):
                pass
