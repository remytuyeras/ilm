import unittest
import sys
sys.path.insert(1,"./")
from tokenizer.intuit import force_json_extension

class TestForceJsonExtension(unittest.TestCase):
    def test_extension_added(self):
        self.assertEqual(force_json_extension("data/file"), "data/file.json")

    def test_extension_replaced(self):
        self.assertEqual(force_json_extension("data/file.txt"), "data/file.json")

    def test_extension_unchanged_if_json(self):
        self.assertEqual(force_json_extension("data/file.json"), "data/file.json")

if __name__ == "__main__":
    unittest.main()
