import unittest
import sys
sys.path.insert(1,"./")
from ilm.tokenizer.intuit import classify_tokens, weight_classified_tokens

class TestTokenClassification(unittest.TestCase):
    def test_classify_tokens(self):
        # Create a dummy weighted token dictionary.
        weighted_tokens = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
        token_bins = classify_tokens(weighted_tokens)
        self.assertIsInstance(token_bins, dict)
        # All tokens should be classified.
        self.assertEqual(set(token_bins.keys()), set(weighted_tokens.keys()))

    def test_weight_classified_tokens(self):
        classified = {"a": 1, "b": 1, "c": 2}
        weights = {"a": 0.1, "b": 0.2, "c": 0.3}
        bins = weight_classified_tokens(classified, weights)
        self.assertIsInstance(bins, dict)
        self.assertIn(1, bins)
        self.assertIn("a", bins[1])
        self.assertEqual(bins[1]["a"], 0.1)

if __name__ == "__main__":
    unittest.main()
