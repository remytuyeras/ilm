import unittest
import sys
sys.path.insert(1,"./")
from ilm.tokenizer.intuit import find_tokens

class TestFindTokens(unittest.TestCase):
    def test_simple_sentence(self):
        text = "Hello, world."
        tokens = find_tokens(text)
        # Check that basic words and punctuation are captured.
        self.assertTrue("Hello" in tokens or "Hello" in [t.strip() for t in tokens])
        self.assertIn(",", tokens)
        self.assertIn("world", [t.strip() for t in tokens])

    def test_latex_tokens(self):
        text = r"\[ x + y \]"
        tokens = find_tokens(text)
        self.assertIn(r"\[", tokens)
        self.assertIn(r"\]", tokens)

if __name__ == "__main__":
    unittest.main()
