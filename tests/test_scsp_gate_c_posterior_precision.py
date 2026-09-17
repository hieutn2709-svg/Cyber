from __future__ import annotations

import unittest

from journal.scsp.gate_c import conditional_non_none_posterior


class GateCPosteriorPrecisionTests(unittest.TestCase):
    def test_accepts_observed_float32_softmax_normalization_drift(self) -> None:
        probabilities = (
            0.002495083725079894,
            0.9774715304374695,
            2.16659846046241e-05,
            5.207580397836864e-05,
            0.00048131856601685286,
            9.052789391716942e-05,
            4.325969348428771e-05,
            3.188583650626242e-05,
            0.0005746583919972181,
            0.018564701080322266,
            1.1753726539609488e-05,
            1.0502305485715624e-05,
            1.9548906493582763e-05,
            1.1020794772775844e-05,
            0.00011113828077213839,
            9.311467692896258e-06,
        )
        entity_types = tuple(f"type-{index}" for index in range(15))

        result = conditional_non_none_posterior(
            probabilities,
            entity_types,
            span_key=("420", 64, 64, "attack-pattern"),
        )

        self.assertAlmostEqual(
            sum(result.conditional_probabilities),
            1.0,
            places=12,
        )
        self.assertEqual(result.top1_entity_type, "type-0")

    def test_still_rejects_material_probability_sum_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "must sum to one"):
            conditional_non_none_posterior(
                [0.20, 0.30, 0.40],
                ["malware", "tool"],
                span_key=("doc", 0, 0),
            )


if __name__ == "__main__":
    unittest.main()
