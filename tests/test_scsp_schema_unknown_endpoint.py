from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.schema import (
    is_profile_compatible,
    load_relation_canonicalization,
    load_task_relationship_profile,
)


class UnknownEndpointSchemaTests(unittest.TestCase):
    def test_profile_compatibility_rejects_unknown_runtime_endpoint_type(self) -> None:
        canonicalization_payload = {
            "version": 1,
            "rules": [
                {
                    "project_label": "uses",
                    "canonical_label": "uses",
                    "swap_endpoints": False,
                    "status": "direct",
                }
            ],
        }
        profile_payload = {
            "version": 1,
            "resolved_entity_types": ["intrusion-set", "malware"],
            "unresolved_entity_types": ["tactic"],
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
            root = Path(td)
            canonicalization_path = root / "canonicalization.json"
            profile_path = root / "profile.json"
            canonicalization_path.write_text(
                json.dumps(canonicalization_payload), encoding="utf-8"
            )
            profile_path.write_text(json.dumps(profile_payload), encoding="utf-8")
            canonicalization = load_relation_canonicalization(canonicalization_path)
            profile = load_task_relationship_profile(profile_path)

        endpoint_pairs = (
            ("unknown-entity", "malware"),
            ("intrusion-set", "unknown-entity"),
        )
        for source_type, target_type in endpoint_pairs:
            with self.subTest(source_type=source_type, target_type=target_type):
                with self.assertRaisesRegex(ValueError, "unknown entity type"):
                    is_profile_compatible(
                        source_type,
                        "uses",
                        target_type,
                        canonicalization=canonicalization,
                        profile=profile,
                    )


if __name__ == "__main__":
    unittest.main()
