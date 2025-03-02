import unittest
import sys
sys.path.insert(1,"./")
from tokenizer.intuit import generate_tokenizer, generate_detokenizer

class TestTokenizerDetokenizer(unittest.TestCase):
    def test_tokenizer_detokenizer_roundtrip(self):
        # Define a simple mapping.
        direct_mapping = {"Hello": "0:0:0", "world": "0:0:1"}
        reverse_mapping = {"0:0:0": "Hello", "0:0:1": "world"}
        tokenizer = generate_tokenizer(direct_mapping)
        detokenizer = generate_detokenizer(reverse_mapping)
        
        # Use a simple text that produces a token exactly matching one in our mapping.
        text = "Hello"
        token_codes = tokenizer(text)
        # We expect to get the code for "Hello"; other tokens not in mapping may return None.
        self.assertIn("0:0:0", token_codes)

        # Detokenize and check roundtrip.
        tokens = [code for code in token_codes if code is not None]
        restored_tokens = detokenizer(tokens)
        self.assertIn("Hello", restored_tokens)

if __name__ == "__main__":
    unittest.main()
