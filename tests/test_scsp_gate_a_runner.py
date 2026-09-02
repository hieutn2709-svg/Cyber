from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from journal.scripts.run_gate_a import build_preflight, write_preflight_artifacts


class GateARunnerTests(unittest.TestCase):
    def _files(
        self,
        td: str,
        *,
        schema_mode: str = "none",
        extra: dict[str, object] | None = None,
    ) -> tuple[Path, Path, Path]:
        root = Path(td)
        dataset = root / "dataset.json"
        dataset.write_text("controlled-data", encoding="utf-8")
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
        config: dict[str, object] = {
            "experiment_name": "gate-a",
            "schema_mode": schema_mode,
            "encoder_model": "FacebookAI/roberta-base",
            "encoder_revision": "c2a5e573587885ce23744cf330ee7c402f0df16f",
            "split_seed": 11800,
            "seed": 42,
            "fold": 1,
            "span_width_coverage": 0.995,
            "max_span_candidates": 128,
            "min_entity_score": 0.05,
            "max_relation_token_distance": 96,
            "relation_negative_ratio": 6,
            "output_dir": str(root / "out"),
        }
        if extra:
            config.update(extra)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        manifest = {
            "split_seed": 11800,
            "run_seed": 42,
            "dataset_sha256": digest,
            "outer_folds": [
                {
                    "fold": 1,
                    "train_document_ids": ["a", "b"],
                    "validation_document_ids": ["c"],
                    "test_document_ids": ["d"],
                }
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return config_path, manifest_path, dataset

    def test_dry_run_validates_manifest_without_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config, manifest, _ = self._files(td)
            result = build_preflight(
                config,
                manifest,
                fold=1,
                dataset_path=None,
                dry_run=True,
            )
        self.assertEqual(result["schema_mode"], "none")
        self.assertEqual(result["fold"], 1)
        self.assertEqual(result["dataset_status"], "not_provided")
        self.assertEqual(
            result["split_counts"],
            {"train": 2, "validation": 1, "test": 1},
        )

    def test_real_prepare_requires_matching_dataset_hash_and_writes_provenance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            config, manifest, dataset = self._files(td)
            result = build_preflight(
                config,
                manifest,
                fold=1,
                dataset_path=dataset,
                dry_run=False,
            )
            paths = write_preflight_artifacts(result, Path(td) / "run")
            resolved = json.loads(paths["resolved_config"].read_text())
            provenance = json.loads(paths["provenance"].read_text())
        self.assertEqual(resolved["selection_scope"], "validation-only")
        self.assertEqual(provenance["dataset_sha256"], result["dataset_sha256"])
        self.assertEqual(
            provenance["execution_status"],
            "preflight_complete_training_not_started",
        )

    def test_rejects_schema_mode_and_test_selected_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config, manifest, _ = self._files(td, schema_mode="hard_profile")
            with self.assertRaisesRegex(ValueError, "schema_mode"):
                build_preflight(
                    config,
                    manifest,
                    fold=1,
                    dataset_path=None,
                    dry_run=True,
                )
        with tempfile.TemporaryDirectory() as td:
            config, manifest, _ = self._files(
                td,
                extra={"test_selected_threshold": 0.7},
            )
            with self.assertRaisesRegex(ValueError, "test.*threshold"):
                build_preflight(
                    config,
                    manifest,
                    fold=1,
                    dataset_path=None,
                    dry_run=True,
                )

    def test_rejects_unpinned_encoder_revision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config, manifest, _ = self._files(
                td,
                extra={"encoder_revision": "main"},
            )
            with self.assertRaisesRegex(ValueError, "immutable"):
                build_preflight(
                    config,
                    manifest,
                    fold=1,
                    dataset_path=None,
                    dry_run=True,
                )

    def test_direct_script_entrypoint_imports_from_repo_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "journal" / "scripts" / "run_gate_a.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--dry-run", result.stdout)


if __name__ == "__main__":
    unittest.main()
