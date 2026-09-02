"""Regression checks for the Q2 journal foundation cleanup."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class JournalFoundationRegression(unittest.TestCase):
    def test_documented_joint_epoch_cap_matches_canonical_config(self) -> None:
        config = json.loads(
            (ROOT / "Configs" / "reviewer_experiment.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["joint_max_epochs"], 12)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        experiment_readme = (ROOT / "experiments" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Joint training is capped at 12 epochs", readme)
        self.assertNotIn("Joint training is capped at 30 epochs", readme)
        self.assertIn("joint maximum 12 epochs/patience 5", experiment_readme)
        self.assertNotIn("joint maximum 30 epochs/patience 5", experiment_readme)

    def test_stix_mapper_default_config_is_independent_of_working_directory(self) -> None:
        module_path = ROOT / "integration" / "stix_mapper.py"
        spec = importlib.util.spec_from_file_location("journal_stix_mapper", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                mapper = module.STIXMapper()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(mapper.stix_version, "2.1")
        self.assertTrue(mapper.entity_map)

    def test_legacy_pipeline_name_does_not_claim_full_stixnet_augmentation(self) -> None:
        settings = (ROOT / "Configs" / "pipeline_settings.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn('name: "CTI Hybrid Prototype - Legacy Inference Defaults"', settings)
        self.assertNotIn("STIXnet + CyberEntRel Augmented Pipeline", settings)
        self.assertIn("STIXnet-inspired", settings)


if __name__ == "__main__":
    unittest.main()
