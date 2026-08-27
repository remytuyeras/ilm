import unittest

from experiments.create_permuted_tokenizer import permute_mapping
from experiments.analyze_permutation_controls import permutation_metrics


def weighted_coordinate_counts(mapping, frequencies):
    counts = []
    for coordinate in range(2):
        values = {}
        for token, code in mapping["direct"].items():
            value = int(code.split(":")[coordinate])
            values[value] = values.get(value, 0) + frequencies[token]
        counts.append(values)
    return counts


class TestPermutedTokenizer(unittest.TestCase):
    def setUp(self):
        self.mapping = {
            "direct": {
                "alpha": "0:0",
                "beta": "0:1",
                "gamma": "1:0",
                "delta": "1:1",
                "epsilon": "2:2",
            },
            "reverse": {
                "0:0": "alpha",
                "0:1": "beta",
                "1:0": "gamma",
                "1:1": "delta",
                "2:2": "epsilon",
            },
            "metadata": {},
        }

    def test_exact_frequency_control_preserves_weighted_coordinate_marginals(self):
        frequencies = {"alpha": 7, "beta": 7, "gamma": 3, "delta": 3, "epsilon": 1}
        permuted = permute_mapping(
            self.mapping,
            314159,
            frequency_control="exact",
            frequencies=frequencies,
        )

        self.assertEqual(
            weighted_coordinate_counts(self.mapping, frequencies),
            weighted_coordinate_counts(permuted, frequencies),
        )
        self.assertEqual(permuted["direct"]["epsilon"], "2:2")

    def test_global_control_retains_existing_single_shuffle_behavior(self):
        first = permute_mapping(self.mapping, 42)
        second = permute_mapping(self.mapping, 42)
        self.assertEqual(first["direct"], second["direct"])
        self.assertEqual(set(first["direct"].values()), set(self.mapping["direct"].values()))

    def test_displacement_metrics_report_token_mass(self):
        target = {
            "direct": {
                "alpha": "0:1",
                "beta": "0:0",
                "gamma": "1:0",
                "delta": "1:1",
                "epsilon": "2:2",
            },
            "reverse": {},
            "metadata": {},
        }
        training_frequencies = {"alpha": 10, "beta": 1, "gamma": 2, "delta": 2, "epsilon": 1}
        test_frequencies = {"alpha": 1, "beta": 10, "gamma": 2, "delta": 2, "epsilon": 1}
        metrics = permutation_metrics(
            self.mapping,
            target,
            training_frequencies,
            top_tokens=2,
            test_frequencies=test_frequencies,
        )

        self.assertEqual(metrics["moved_type_count"], 2)
        self.assertEqual(metrics["training"]["moved_event_count"], 11)
        self.assertAlmostEqual(metrics["training"]["moved_event_rate"], 11 / 16)
        self.assertFalse(metrics["training"]["frequency_weighted_marginals_match"])
        self.assertFalse(metrics["test"]["frequency_weighted_marginals_match"])

    def test_exact_control_can_preserve_train_but_not_test_marginals(self):
        target = {
            "direct": {
                "alpha": "0:1",
                "beta": "0:0",
                "gamma": "1:0",
                "delta": "1:1",
                "epsilon": "2:2",
            },
            "reverse": {},
            "metadata": {},
        }
        training_frequencies = {"alpha": 7, "beta": 7, "gamma": 3, "delta": 3, "epsilon": 1}
        test_frequencies = {"alpha": 11, "beta": 2, "gamma": 1, "delta": 1, "epsilon": 1}
        metrics = permutation_metrics(
            self.mapping,
            target,
            training_frequencies,
            top_tokens=2,
            test_frequencies=test_frequencies,
        )

        self.assertTrue(metrics["training"]["frequency_weighted_marginals_match"])
        self.assertEqual(metrics["training"]["frequency_weighted_marginal_max_total_variation"], 0.0)
        self.assertFalse(metrics["test"]["frequency_weighted_marginals_match"])
        self.assertGreater(metrics["test"]["frequency_weighted_marginal_max_total_variation"], 0.0)


if __name__ == "__main__":
    unittest.main()
