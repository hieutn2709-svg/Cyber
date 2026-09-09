from __future__ import annotations

import unittest

try:
    from journal.scsp.gate_c import conditional_non_none_posterior
except ImportError:
    conditional_non_none_posterior = None


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


if __name__ == "__main__":
    unittest.main()
