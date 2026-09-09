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

try:
    from journal.scsp.schema import (
        is_profile_compatible,
        load_task_relationship_profile,
    )
except ImportError:
    is_profile_compatible = None
    load_task_relationship_profile = None


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

    def test_versioned_canonicalization_covers_relation_inventory(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)

        repo_root = Path(__file__).resolve().parents[1]
        canonicalization_path = (
            repo_root
            / "journal"
            / "configs"
            / "stix"
            / "relation_canonicalization_v1.json"
        )
        inventory_path = (
            repo_root / "journal" / "configs" / "gate_a_label_inventory.json"
        )

        table = load_relation_canonicalization(canonicalization_path)
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(table.by_project_label),
            set(inventory["relation_types"]),
        )

    def test_canonicalization_loader_rejects_duplicate_project_label(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)

        payload = {
            "version": 1,
            "rules": [
                {
                    "project_label": "uses",
                    "canonical_label": "uses",
                    "swap_endpoints": False,
                    "status": "direct",
                },
                {
                    "project_label": "uses",
                    "canonical_label": "uses",
                    "swap_endpoints": False,
                    "status": "direct",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "canonicalization.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate project_label"):
                load_relation_canonicalization(path)

    def test_canonicalization_loader_rejects_invalid_status(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)

        payload = {
            "version": 1,
            "rules": [
                {
                    "project_label": "uses",
                    "canonical_label": "uses",
                    "swap_endpoints": False,
                    "status": "renamed",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "canonicalization.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid status"):
                load_relation_canonicalization(path)

    def test_canonicalization_loader_rejects_inconsistent_canonical_label(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)

        invalid_rules = (
            {
                "project_label": "used-in",
                "canonical_label": "uses",
                "swap_endpoints": False,
                "status": "unresolved",
            },
            {
                "project_label": "uses",
                "canonical_label": None,
                "swap_endpoints": False,
                "status": "direct",
            },
        )
        for rule in invalid_rules:
            with self.subTest(rule=rule):
                payload = {"version": 1, "rules": [rule]}
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "canonicalization.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "canonical_label"
                    ):
                        load_relation_canonicalization(path)

    def test_canonicalization_loader_rejects_non_boolean_swap_endpoints(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)

        payload = {
            "version": 1,
            "rules": [
                {
                    "project_label": "uses",
                    "canonical_label": "uses",
                    "swap_endpoints": "false",
                    "status": "direct",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "canonicalization.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "swap_endpoints"):
                load_relation_canonicalization(path)

    def test_canonicalization_loader_rejects_missing_required_field(self) -> None:
        self.assertIsNotNone(load_relation_canonicalization)

        valid_rule = {
            "project_label": "uses",
            "canonical_label": "uses",
            "swap_endpoints": False,
            "status": "direct",
        }
        for missing_field in (
            "project_label",
            "canonical_label",
            "swap_endpoints",
            "status",
        ):
            with self.subTest(missing_field=missing_field):
                rule = dict(valid_rule)
                del rule[missing_field]
                payload = {"version": 1, "rules": [rule]}
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "canonicalization.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "missing required field"):
                        load_relation_canonicalization(path)

    def test_profile_compatibility_allows_explicit_canonical_triple(self) -> None:
        self.assertIsNotNone(load_task_relationship_profile)
        self.assertIsNotNone(is_profile_compatible)
        self.assertIsNotNone(load_relation_canonicalization)

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
            "resolved_entity_types": ["intrusion-set", "malware", "tool"],
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

        result = is_profile_compatible(
            "intrusion-set",
            "uses",
            "malware",
            canonicalization=canonicalization,
            profile=profile,
        )
        self.assertIs(result, True)

    def test_profile_compatibility_returns_none_for_unresolved_relation(self) -> None:
        self.assertIsNotNone(load_task_relationship_profile)
        self.assertIsNotNone(is_profile_compatible)
        self.assertIsNotNone(load_relation_canonicalization)

        canonicalization_payload = {
            "version": 1,
            "rules": [
                {
                    "project_label": "used-in",
                    "canonical_label": None,
                    "swap_endpoints": False,
                    "status": "unresolved",
                }
            ],
        }
        profile_payload = {
            "version": 1,
            "resolved_entity_types": ["intrusion-set", "malware"],
            "unresolved_entity_types": [],
            "allowed_triples": [],
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

        result = is_profile_compatible(
            "intrusion-set",
            "used-in",
            "malware",
            canonicalization=canonicalization,
            profile=profile,
        )
        self.assertIsNone(result)

    def test_profile_compatibility_returns_none_for_unresolved_endpoint(self) -> None:
        self.assertIsNotNone(load_task_relationship_profile)
        self.assertIsNotNone(is_profile_compatible)
        self.assertIsNotNone(load_relation_canonicalization)

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
            ("tactic", "malware"),
            ("intrusion-set", "tactic"),
        )
        for source_type, target_type in endpoint_pairs:
            with self.subTest(source_type=source_type, target_type=target_type):
                result = is_profile_compatible(
                    source_type,
                    "uses",
                    target_type,
                    canonicalization=canonicalization,
                    profile=profile,
                )
                self.assertIsNone(result)

    def test_profile_compatibility_swaps_inverse_relation_endpoints(self) -> None:
        self.assertIsNotNone(load_task_relationship_profile)
        self.assertIsNotNone(is_profile_compatible)
        self.assertIsNotNone(load_relation_canonicalization)

        canonicalization_payload = {
            "version": 1,
            "rules": [
                {
                    "project_label": "used-by",
                    "canonical_label": "uses",
                    "swap_endpoints": True,
                    "status": "inverse",
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

        result = is_profile_compatible(
            "malware",
            "used-by",
            "intrusion-set",
            canonicalization=canonicalization,
            profile=profile,
        )
        self.assertIs(result, True)


if __name__ == "__main__":
    unittest.main()
