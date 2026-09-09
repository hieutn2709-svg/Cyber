from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.schema import load_task_relationship_profile


class ProfileMissingFieldsSchemaTests(unittest.TestCase):
    def test_profile_loader_rejects_allowed_triple_missing_required_field(self) -> None:
        valid_triple = {
            "source_type": "intrusion-set",
            "relation_type": "uses",
            "target_type": "malware",
        }
        for missing_field in ("source_type", "relation_type", "target_type"):
            with self.subTest(missing_field=missing_field):
                triple = dict(valid_triple)
                del triple[missing_field]
                payload = {
                    "version": 1,
                    "resolved_entity_types": ["intrusion-set", "malware"],
                    "unresolved_entity_types": [],
                    "allowed_triples": [triple],
                }
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "profile.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "missing required field"):
                        load_task_relationship_profile(path)


if __name__ == "__main__":
    unittest.main()
