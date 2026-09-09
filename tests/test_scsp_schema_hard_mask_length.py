from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from journal.scsp.schema import (
    hard_profile_mask,
    load_relation_canonicalization,
    load_task_relationship_profile,
)


class HardProfileMaskLengthTests(unittest.TestCase):
    def test_hard_profile_mask_rejects_mismatched_logits_and_labels(self) -> None:
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
            root = Path(td)
            canonicalization_path = root / "canonicalization.json"
            profile_path = root / "profile.json"
            canonicalization_path.write_text(
                json.dumps(canonicalization_payload), encoding="utf-8"
            )
            profile_path.write_text(json.dumps(profile_payload), encoding="utf-8")
            canonicalization = load_relation_canonicalization(canonicalization_path)
            profile = load_task_relationship_profile(profile_path)

        with self.assertRaisesRegex(ValueError, "same length"):
            hard_profile_mask(
                [9.0, 8.0],
                source_type="intrusion-set",
                target_type="malware",
                relation_types=["uses"],
                canonicalization=canonicalization,
                profile=profile,
            )


if __name__ == "__main__":
    unittest.main()
