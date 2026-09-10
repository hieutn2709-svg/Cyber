from __future__ import annotations

import unittest

try:
    from journal.scsp.gate_c import conditional_non_none_posterior
except ImportError:
    conditional_non_none_posterior = None

try:
    from journal.scsp.gate_c import probabilistic_compatibility
except ImportError:
    probabilistic_compatibility = None

from journal.scsp.schema import (
    CanonicalRelation,
    CanonicalizationTable,
    TaskRelationshipProfile,
)


class GateCPosteriorMathTests(unittest.TestCase):
    def test_conditional_non_none_posterior_normalizes(self) -> None:
        self.assertIsNotNone(
            conditional_non_none_posterior,
            "Gate C posterior API must exist",
        )
        result = conditional_non_none_posterior(
            [0.20, 0.40, 0.30, 0.10],
            ["intrusion-set", "threat-actor", "malware"],
            span_key=("doc", 1, 2),
        )
        self.assertAlmostEqual(
            sum(result.conditional_probabilities), 1.0, places=12
        )
        self.assertAlmostEqual(result.none_probability, 0.20, places=12)
        self.assertAlmostEqual(result.entity_probability, 0.80, places=12)
        self.assertEqual(result.top1_entity_type, "intrusion-set")

    def test_conditional_non_none_posterior_rejects_zero_non_none_mass(self) -> None:
        self.assertIsNotNone(
            conditional_non_none_posterior,
            "Gate C posterior API must exist",
        )
        with self.assertRaisesRegex(ValueError, "non-NONE mass"):
            conditional_non_none_posterior(
                [1.0, 0.0, 0.0],
                ["malware", "tool"],
                span_key=("doc", 0, 0),
            )


class GateCCompatibilityMathTests(unittest.TestCase):
    ENTITY_TYPES = ("intrusion-set", "malware", "identity", "tactic")

    def setUp(self) -> None:
        self.canonicalization = CanonicalizationTable(
            by_project_label={
                "uses": CanonicalRelation("uses", False, "direct"),
                "targets": CanonicalRelation("targets", False, "direct"),
                "targeted-by": CanonicalRelation("targets", True, "inverse"),
                "used-in": CanonicalRelation(None, False, "unresolved"),
            }
        )
        self.profile = TaskRelationshipProfile(
            allowed_triples=frozenset(
                {
                    ("intrusion-set", "uses", "malware"),
                    ("intrusion-set", "targets", "identity"),
                }
            ),
            resolved_entity_types=frozenset(
                {"intrusion-set", "malware", "identity"}
            ),
            unresolved_entity_types=frozenset({"tactic"}),
        )

    def posterior(self, values, name):
        self.assertIsNotNone(conditional_non_none_posterior)
        return conditional_non_none_posterior(
            [0.0, *values],
            self.ENTITY_TYPES,
            span_key=(name, 0, 0),
        )

    def test_one_hot_resolved_compatible_triple_scores_one(self) -> None:
        self.assertIsNotNone(
            probabilistic_compatibility,
            "Gate C probabilistic compatibility API must exist",
        )
        source = self.posterior([1.0, 0.0, 0.0, 0.0], "source")
        target = self.posterior([0.0, 1.0, 0.0, 0.0], "target")
        scores = probabilistic_compatibility(
            source,
            target,
            ["uses"],
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertEqual(scores, (1.0,))

    def test_one_hot_resolved_incompatible_triple_scores_zero(self) -> None:
        self.assertIsNotNone(
            probabilistic_compatibility,
            "Gate C probabilistic compatibility API must exist",
        )
        source = self.posterior([1.0, 0.0, 0.0, 0.0], "source")
        target = self.posterior([0.0, 0.0, 1.0, 0.0], "target")
        scores = probabilistic_compatibility(
            source,
            target,
            ["uses"],
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertEqual(scores, (0.0,))

    def test_unresolved_relation_used_in_scores_one(self) -> None:
        self.assertIsNotNone(
            probabilistic_compatibility,
            "Gate C probabilistic compatibility API must exist",
        )
        source = self.posterior([0.6, 0.0, 0.0, 0.4], "source")
        target = self.posterior([0.0, 0.3, 0.7, 0.0], "target")
        scores = probabilistic_compatibility(
            source,
            target,
            ["used-in"],
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertAlmostEqual(scores[0], 1.0, places=12)

    def test_unresolved_endpoint_mass_contributes_neutrally(self) -> None:
        self.assertIsNotNone(
            probabilistic_compatibility,
            "Gate C probabilistic compatibility API must exist",
        )
        # intrusion-set -> identity is incompatible for "uses", while any
        # hypothetical endpoint assignment involving unresolved tactic is neutral.
        source = self.posterior([0.5, 0.0, 0.0, 0.5], "source")
        target = self.posterior([0.0, 0.0, 1.0, 0.0], "target")
        scores = probabilistic_compatibility(
            source,
            target,
            ["uses"],
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertAlmostEqual(scores[0], 0.5, places=12)

    def test_inverse_relation_swaps_types_for_lookup_only(self) -> None:
        self.assertIsNotNone(
            probabilistic_compatibility,
            "Gate C probabilistic compatibility API must exist",
        )
        # identity --targeted-by--> intrusion-set canonicalizes for lookup to
        # intrusion-set --targets--> identity.
        source = self.posterior([0.0, 0.0, 1.0, 0.0], "source")
        target = self.posterior([1.0, 0.0, 0.0, 0.0], "target")
        scores = probabilistic_compatibility(
            source,
            target,
            ["targeted-by"],
            canonicalization=self.canonicalization,
            profile=self.profile,
        )
        self.assertEqual(scores, (1.0,))


if __name__ == "__main__":
    unittest.main()
