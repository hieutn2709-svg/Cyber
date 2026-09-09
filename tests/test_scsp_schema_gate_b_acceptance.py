from __future__ import annotations

import unittest
from pathlib import Path

from journal.scsp.data import LabelInventory
from journal.scsp.schema import (
    hard_profile_mask,
    load_relation_canonicalization,
    load_task_relationship_profile,
)


class GateBSchemaAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.inventory = LabelInventory.from_json(
            root / "journal" / "configs" / "gate_a_label_inventory.json"
        )
        cls.canonicalization = load_relation_canonicalization(
            root / "journal" / "configs" / "stix" / "relation_canonicalization_v1.json"
        )
        cls.profile = load_task_relationship_profile(
            root / "journal" / "configs" / "stix" / "task_relationship_profile_v1.json"
        )

    def test_repository_profile_is_complete_and_versioned(self):
        self.assertEqual(len(self.canonicalization.by_project_label), 13)
        self.assertEqual(len(self.profile.resolved_entity_types), 12)
        self.assertEqual(len(self.profile.unresolved_entity_types), 3)
        self.assertEqual(len(self.profile.allowed_triples), 55)
        self.assertEqual(
            self.profile.resolved_entity_types | self.profile.unresolved_entity_types,
            set(self.inventory.trainable_entity_types),
        )
        self.assertEqual(
            set(self.canonicalization.by_project_label),
            set(self.inventory.relation_types),
        )
        for triple in (
            ("attack-pattern", "delivers", "malware"),
            ("malware", "authored-by", "intrusion-set"),
            ("tool", "has", "vulnerability"),
            ("threat-actor", "attributed-to", "identity"),
        ):
            self.assertIn(triple, self.profile.allowed_triples)

    def test_hard_mask_rejects_empty_relation_inventory(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            hard_profile_mask(
                (),
                source_type="intrusion-set",
                target_type="malware",
                relation_types=(),
                canonicalization=self.canonicalization,
                profile=self.profile,
            )

    def test_unresolved_relation_label_survives_mask(self):
        result = hard_profile_mask(
            (8.0, 9.0),
            source_type="intrusion-set",
            target_type="malware",
            relation_types=("uses", "used-in"),
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertEqual(result.compatibility, (True, None))
        self.assertEqual(result.masked_logits, (8.0, 9.0))
        self.assertEqual(result.selected_index, 1)

    def test_unresolved_endpoint_preserves_every_logit(self):
        logits = (9.0, 8.0, 7.0)
        result = hard_profile_mask(
            logits,
            source_type="tactic",
            target_type="malware",
            relation_types=("targets", "uses", "used-in"),
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertEqual(result.compatibility, (None, None, None))
        self.assertEqual(result.masked_logits, logits)
        self.assertEqual(result.selected_index, 0)

    def test_exact_tie_chooses_lowest_surviving_index(self):
        result = hard_profile_mask(
            (4.0, 4.0),
            source_type="intrusion-set",
            target_type="malware",
            relation_types=("uses", "used-in"),
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertEqual(result.selected_index, 0)

    def test_all_resolved_classes_blocked_returns_no_survivor(self):
        result = hard_profile_mask(
            (9.0, 8.0),
            source_type="identity",
            target_type="malware",
            relation_types=("targets", "uses"),
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertEqual(result.compatibility, (False, False))
        self.assertIsNone(result.selected_index)


if __name__ == "__main__":
    unittest.main()
