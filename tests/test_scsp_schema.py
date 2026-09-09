from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from journal.scsp.schema import (
        canonicalize_relation,
        load_relation_canonicalization,
    )
except ImportError:
    canonicalize_relation = None
    load_relation_canonicalization = None


class SchemaTests(unittest.TestCase):
    def test_relation_canonicalization_maps_deliver_alias(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)
        self.assertIsNotNone(canonicalize_relation)

        payload = {
            "version": 1,
            "rules": [
                {
                    "project_label": "deliver",
                    "canonical_label": "delivers",
                    "swap_endpoints": False,
                    "status": "alias",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "canonicalization.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            table = load_relation_canonicalization(path)

        result = canonicalize_relation("deliver", table)
        self.assertEqual(result.label, "delivers")
        self.assertFalse(result.swap_endpoints)
        self.assertEqual(result.status, "alias")

    def test_versioned_canonicalization_defines_inverse_project_labels(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)
        self.assertIsNotNone(canonicalize_relation)

        path = (
            Path(__file__).resolve().parents[1]
            / "journal"
            / "configs"
            / "stix"
            / "relation_canonicalization_v1.json"
        )
        table = load_relation_canonicalization(path)

        expected = {
            "targeted-by": "targets",
            "used-by": "uses",
        }
        for project_label, canonical_label in expected.items():
            with self.subTest(project_label=project_label):
                result = canonicalize_relation(project_label, table)
                self.assertEqual(result.label, canonical_label)
                self.assertTrue(result.swap_endpoints)
                self.assertEqual(result.status, "inverse")

    def test_versioned_canonicalization_marks_used_in_unresolved(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)
        self.assertIsNotNone(canonicalize_relation)

        path = (
            Path(__file__).resolve().parents[1]
            / "journal"
            / "configs"
            / "stix"
            / "relation_canonicalization_v1.json"
        )
        table = load_relation_canonicalization(path)

        result = canonicalize_relation("used-in", table)
        self.assertIsNone(result.label)
        self.assertFalse(result.swap_endpoints)
        self.assertEqual(result.status, "unresolved")


if __name__ == "__main__":
    unittest.main()
