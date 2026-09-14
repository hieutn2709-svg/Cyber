from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileBetaDiagnosticsArtifactTests(unittest.TestCase):
    def test_validation_artifacts_preserve_diagnostics_by_beta_report(self) -> None:
        builder = getattr(gate_c_cli, "_build_validation_run_artifacts", None)
        self.assertIsNotNone(builder, "Gate C validation artifact builder must exist")

        prepared = SimpleNamespace(
            inventory=SimpleNamespace(relation_types=("uses", "targets")),
            canonicalization="canonicalization",
            profile="profile",
            device="cpu",
            run_id="gate-c-beta-diagnostics-artifact-test",
            fold=1,
            seed=42,
            git_commit="c" * 40,
            dataset_sha256="a" * 64,
            config_sha256="b" * 64,
            canonicalization_sha256="d" * 64,
            task_profile_sha256="e" * 64,
            gate_a_checkpoint_sha256="f" * 64,
            checkpoint={
                "git_commit": gate_c_cli._FROZEN_GATE_A_COMMIT,
                "dataset_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "epoch": 10,
                "threshold": 0.96,
                "width_cap": 8,
            },
            run_metadata={
                "fold": 1,
                "seed": 42,
                "dataset_sha256": "a" * 64,
                "combined_config_sha256": "b" * 64,
            },
        )
        inferences = ("inference",)
        records = ("record",)
        selection = {
            "best": {
                "beta": 0.25,
                "threshold": 0.96,
                "relation_f1": 0.55,
                "primary_entity_f1": 0.66,
            },
            "grid": [],
        }
        metrics = {
            "all_relation": {"f1": 0.55},
            "primary_entity": {"f1": 0.66},
        }
        diagnostics_by_beta = {
            "selected_threshold": 0.96,
            "by_beta": [
                {
                    "beta": 0.0,
                    "argmax_change_count_vs_gate_a": 0,
                    "argmax_change_rate_vs_gate_a": 0.0,
                    "above_selected_threshold_change_count": 0,
                    "transition_counts": {},
                },
                {
                    "beta": 0.25,
                    "argmax_change_count_vs_gate_a": 3,
                    "argmax_change_rate_vs_gate_a": 0.03,
                    "above_selected_threshold_change_count": 1,
                    "transition_counts": {"uses->targets": 3},
                },
            ],
        }

        with patch.object(
            gate_c_cli,
            "_record_to_dict",
            return_value={"prediction": "row"},
        ), patch.object(
            gate_c_cli,
            "build_gate_c_span_rows",
            return_value=({"posterior": "row"},),
        ), patch.object(
            gate_c_cli,
            "build_gate_c_scored_pair_rows",
            return_value=({"pair": "row"},),
        ), patch.object(
            gate_c_cli,
            "build_gate_c_diagnostics",
            return_value={"diagnostics": True},
        ), patch.object(
            gate_c_cli,
            "_environment",
            return_value={"device": "cpu"},
        ):
            try:
                artifacts = builder(
                    prepared,
                    inferences,
                    records,
                    selection,
                    metrics,
                    beta=0.25,
                    threshold=0.96,
                    mode="dev",
                    diagnostics_by_beta=diagnostics_by_beta,
                )
            except TypeError as exc:
                self.fail(
                    "validation artifact builder must accept diagnostics-by-beta "
                    f"report: {exc}"
                )

        self.assertIn("validation_beta_diagnostics.json", artifacts)
        self.assertEqual(
            artifacts["validation_beta_diagnostics.json"], diagnostics_by_beta
        )
        self.assertEqual(
            artifacts["run_summary.json"]["diagnostics_by_beta"],
            diagnostics_by_beta,
        )
        self.assertEqual(
            artifacts["run_summary.json"]["diagnostics_by_beta"]["selected_threshold"],
            0.96,
        )


if __name__ == "__main__":
    unittest.main()
