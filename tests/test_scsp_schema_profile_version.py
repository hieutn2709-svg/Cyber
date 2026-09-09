from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.schema import load_task_relationship_profile


class ProfileVersionSchemaTests(unittest.TestCase):
    def test_profile_loader_rejects_unsupported_version(self) -> None:
        payload = {
            "version": 2,
            "resolved_entity_types": ["intrusion-set", "malware"],
            "unresolved_entity_types": [],
            "allowed_triples": [
                {
                    "source_type": "intrusion-set",
                    "relation_type": "uses",
                    "target_type": "malware",
                    "source_note": "STIX 2.1 Appendix B",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported profile version"):
                load_task_relationship_profile(path)


if __name__ == "__main__":
    unittest.main()
