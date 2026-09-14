from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
