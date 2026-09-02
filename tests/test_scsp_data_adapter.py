from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.data import LabelInventory, audit_windows, load_clean_windows


class DataAdapterTests(unittest.TestCase):
    def _inventory_payload(self) -> dict[str, list[str]]:
        return {
            "primary_entity_types": [
                "intrusion-set",
                "malware",
                "location",
            ],
            "auxiliary_entity_types": ["tactic"],
            "relation_types": ["uses", "targets"],
        }

    def _dataset_payload(self) -> list[dict[str, object]]:
        return [
            {
                "doc_seq_index": 0,
                "doc_id": "d0",
                "window_index": 0,
                "token_start_global": 0,
                "token_end_global": 4,
                "input_ids": [0, 10, 11, 12, 2],
                "attention_mask": [1, 1, 1, 1, 1],
                "label_mask": [False, True, True, True, False],
                "entity_spans": [
                    {
                        "entity_id": "e1",
                        "type": "intrusion-set",
                        "token_start": 1,
                        "token_end": 1,
                    },
                    {
                        "entity_id": "e2",
                        "type": "malware",
                        "token_start": 2,
                        "token_end": 2,
                    },
                    {
                        "entity_id": "e3",
                        "type": "tactic",
                        "token_start": 3,
                        "token_end": 3,
                    },
                ],
                "relations": [
                    {
                        "source_id": "e1",
                        "target_id": "e2",
                        "type": "uses",
                    },
                    {
                        "source_id": "e2",
                        "target_id": "e3",
                        "type": "uses",
                    },
                ],
            },
            {
                "doc_seq_index": 1,
                "doc_id": "d1",
                "window_index": 0,
                "token_start_global": 0,
                "token_end_global": 3,
                "input_ids": [0, 20, 21, 2],
                "attention_mask": [1, 1, 1, 1],
                "label_mask": [False, True, True, False],
                "entity_spans": [
                    {
                        "entity_id": "x1",
                        "type": "location",
                        "token_start": 1,
                        "token_end": 1,
                    }
                ],
                "relations": [],
            },
        ]

    def test_inventory_separates_primary_and_auxiliary_types(self) -> None:
        inventory = LabelInventory.from_dict(self._inventory_payload())
        self.assertEqual(
            inventory.primary_entity_types,
            ("intrusion-set", "malware", "location"),
        )
        self.assertEqual(inventory.auxiliary_entity_types, ("tactic",))
        self.assertEqual(
            inventory.trainable_entity_types,
            ("intrusion-set", "malware", "location", "tactic"),
        )
        self.assertTrue(inventory.is_primary("malware"))
        self.assertFalse(inventory.is_primary("tactic"))

    def test_inventory_rejects_duplicate_primary_auxiliary_type(self) -> None:
        payload = self._inventory_payload()
        payload["auxiliary_entity_types"] = ["malware"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            LabelInventory.from_dict(payload)

    def test_loader_preserves_content_bounds_and_auxiliary_relation_endpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inventory_path = root / "inventory.json"
            data_path = root / "data.json"
            inventory_path.write_text(
                json.dumps(self._inventory_payload()),
                encoding="utf-8",
            )
            data_path.write_text(
                json.dumps(self._dataset_payload()),
                encoding="utf-8",
            )
            inventory = LabelInventory.from_json(inventory_path)
            windows = load_clean_windows(data_path, inventory)

        self.assertEqual(len(windows), 2)
        first = windows[0]
        self.assertEqual((first.content_start, first.content_end), (1, 3))
        self.assertEqual(len(first.gold_spans), 3)
        self.assertEqual(len(first.primary_gold_spans), 2)
        self.assertEqual(len(first.auxiliary_gold_spans), 1)
        self.assertEqual(len(first.gold_relations), 2)
        self.assertEqual(first.gold_relations[1].target.label, "tactic")

    def test_loader_rejects_vector_length_mismatch(self) -> None:
        payload = self._dataset_payload()
        payload[0]["attention_mask"] = [1, 1]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            inventory = LabelInventory.from_dict(self._inventory_payload())
            with self.assertRaisesRegex(ValueError, "length"):
                load_clean_windows(path, inventory)

    def test_audit_reports_primary_auxiliary_and_relation_scope(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data.json"
            path.write_text(
                json.dumps(self._dataset_payload()),
                encoding="utf-8",
            )
            inventory = LabelInventory.from_dict(self._inventory_payload())
            windows = load_clean_windows(path, inventory)
            report = audit_windows(windows, inventory)

        self.assertEqual(report["window_count"], 2)
        self.assertEqual(report["document_count"], 2)
        self.assertEqual(report["primary_entity_count"], 3)
        self.assertEqual(report["auxiliary_entity_count"], 1)
        self.assertEqual(report["relation_count"], 2)
        self.assertEqual(report["relations_with_auxiliary_endpoint"], 1)


if __name__ == "__main__":
    unittest.main()
