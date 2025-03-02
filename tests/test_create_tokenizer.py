import unittest
import tempfile
import os
import sys
sys.path.insert(1,"./")
from tokenizer.intuit import create_tokenizer, load_tokenizer

class TestTokenizerIntegration(unittest.TestCase):
    def test_create_and_load_tokenizer(self):
        # Create a temporary file to act as the training input.
        training_text = "Hello world. This is a test. Another sentence."
        with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as tmp_train:
            tmp_train.write(training_text)
            train_filename = tmp_train.name

        # Create a temporary file path for the tokenizer mapping with a .json extension.
        mapping_filename = tempfile.mktemp(suffix=".json")

        # Create tokenizer and save the mapping.
        tokenizer, detokenizer = create_tokenizer(train_filename, mapping_filename)
        
        # Load the tokenizer from the saved mapping.
        loaded_tokenizer, loaded_detokenizer = load_tokenizer(mapping_filename)

        # Test tokenization and detokenization on a sample sentence.
        sample_sentence = "Hello world."
        tokens1 = tokenizer(sample_sentence)
        tokens2 = loaded_tokenizer(sample_sentence)
        self.assertEqual(tokens1, tokens2)

        detok1 = detokenizer(tokens1)
        detok2 = loaded_detokenizer(tokens2)
        self.assertEqual(detok1, detok2)

        # Clean up temporary files.
        os.remove(train_filename)
        os.remove(mapping_filename)

if __name__ == "__main__":
    unittest.main()

