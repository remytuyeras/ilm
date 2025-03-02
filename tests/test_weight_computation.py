import unittest
import tempfile
import os
import sys
sys.path.insert(1,"./")
from tokenizer.intuit  import compute_token_weights

class TestComputeTokenWeights(unittest.TestCase):
    def test_weights_non_empty(self):
        # Create a temporary file with sample text.
        sample_text = "This is a test. Another sentence, with commas."
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp_file:
            tmp_file.write(sample_text)
            tmp_filename = tmp_file.name

        # Test for segmentation level 1 (split on dot).
        weights = compute_token_weights(tmp_filename, 1)
        self.assertIsInstance(weights, dict)
        self.assertGreater(len(weights), 0)

        # Clean up
        os.remove(tmp_filename)

if __name__ == "__main__":
    unittest.main()
