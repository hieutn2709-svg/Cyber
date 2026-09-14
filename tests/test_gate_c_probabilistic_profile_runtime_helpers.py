from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import journal.scripts.evaluate_gate_c_probabilistic_profile as gate_c_cli


class GateCProbabilisticProfileRuntimeHelperTests(unittest.TestCase):
    def test_windows_for_ids_materializes_only_requested_documents(self) -> None:
        helper = getattr(gate_c_cli, "_windows_for_ids", None)
        self.assertIsNotNone(helper, "Gate C document-window selector must exist")

        windows = (
            SimpleNamespace(doc_id="train-doc", window_index=0),
            SimpleNamespace(doc_id="val-doc", window_index=0),
            SimpleNamespace(doc_id="test-doc", window_index=0),
            SimpleNamespace(doc_id="val-doc", window_index=1),
        )

        selected = helper(windows, ("val-doc",))

        self.assertEqual(
            [(window.doc_id, window.window_index) for window in selected],
            [("val-doc", 0), ("val-doc", 1)],
        )

    def test_write_artifacts_creates_deterministic_json_files(self) -> None:
        writer = getattr(gate_c_cli, "_write_artifacts", None)
        self.assertIsNotNone(writer, "Gate C artifact writer must exist")

        artifacts = {
            "validation_metrics.json": {
                "z": 1,
                "a": {"f1": 0.5},
            },
            "run_summary.json": {
                "test_evaluated": False,
                "status": "dev_complete",
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "nested" / "gate-c"
            writer(output_dir, artifacts)

            self.assertTrue(output_dir.is_dir())
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                set(artifacts),
            )
            for name, payload in artifacts.items():
                path = output_dir / name
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    json.dumps(
                        payload,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    + "\n",
                )

    def test_write_artifacts_serializes_jsonl_as_one_strict_json_row_per_line(self) -> None:
        writer = getattr(gate_c_cli, "_write_artifacts", None)
        self.assertIsNotNone(writer, "Gate C artifact writer must exist")

        rows = (
            {"z": 2, "a": "first"},
            {"z": 1, "a": "second"},
        )
        artifacts = {"validation_predictions.jsonl": rows}

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "gate-c-jsonl"
            writer(output_dir, artifacts)
            path = output_dir / "validation_predictions.jsonl"
            text = path.read_text(encoding="utf-8")

        expected = "".join(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
        self.assertEqual(text, expected)
        self.assertEqual(
            tuple(json.loads(line) for line in text.splitlines()),
            rows,
        )


if __name__ == "__main__":
    unittest.main()
