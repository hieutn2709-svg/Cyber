from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.schema import load_task_relationship_profile


class ProfileOverlapSchemaTests(unittest.TestCase):
    def test_profile_loader_rejects_overlapping_endpoint_status_sets(self) -> None:
        payload = {
            "version": 1,
            "resolved_entity_types": ["intrusion-set", "malware", "tactic"],
            "unresolved_entity_types": ["tactic"],
            "allowed_triples": [],
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_task_relationship_profile(path)


if __name__ == "__main__":
    unittest.main()
