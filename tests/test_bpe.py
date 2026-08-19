import tempfile
import unittest
from pathlib import Path

from mini_transformer.bpe import BPETokenizer


class BPETokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = "The theater speaks softly.\nThe theater speaks clearly.\n"
        self.tokenizer = BPETokenizer.from_text(self.text, vocab_size=64, min_frequency=2)

    def test_special_ids_and_round_trip(self) -> None:
        self.assertEqual(self.tokenizer.pad_id, 0)
        self.assertEqual(self.tokenizer.bos_id, 1)
        self.assertEqual(self.tokenizer.eos_id, 2)
        self.assertEqual(self.tokenizer.unk_id, 3)
        ids = self.tokenizer.encode(self.text)
        self.assertEqual(self.tokenizer.decode(ids), self.text)

    def test_special_tokens_round_trip(self) -> None:
        ids = self.tokenizer.encode("The", add_special_tokens=True)
        self.assertEqual(ids[0], self.tokenizer.bos_id)
        self.assertEqual(ids[-1], self.tokenizer.eos_id)
        self.assertEqual(self.tokenizer.decode(ids), "The")
        self.assertIn("<bos>", self.tokenizer.decode(ids, skip_special_tokens=False))

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            restored = BPETokenizer.load(path)
            self.assertEqual(restored.state_dict(), self.tokenizer.state_dict())
            self.assertEqual(restored.decode(restored.encode(self.text)), self.text)


if __name__ == "__main__":
    unittest.main()
