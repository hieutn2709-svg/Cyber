from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.schema import load_task_relationship_profile


class ProfileUndeclaredEndpointSchemaTests(unittest.TestCase):
    def test_profile_loader_rejects_allowed_triple_with_undeclared_endpoint(self) -> None:
        endpoint_cases = (
            ("unknown-entity", "malware"),
            ("intrusion-set", "unknown-entity"),
        )
        for source_type, target_type in endpoint_cases:
            with self.subTest(source_type=source_type, target_type=target_type):
                payload = {
                    "version": 1,
                    "resolved_entity_types": ["intrusion-set", "malware"],
                    "unresolved_entity_types": ["tactic"],
                    "allowed_triples": [
                        {
                            "source_type": source_type,
                            "relation_type": "uses",
                            "target_type": target_type,
                            "source_note": "STIX 2.1 Appendix B",
                        }
                    ],
                }
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "profile.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "undeclared entity type"):
                        load_task_relationship_profile(path)


if __name__ == "__main__":
    unittest.main()
