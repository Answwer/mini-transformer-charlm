import tempfile
import unittest
from pathlib import Path

from mini_transformer.tokenizer import CharacterTokenizer


class CharacterTokenizerTest(unittest.TestCase):
    def test_round_trip_and_special_ids(self) -> None:
        tokenizer = CharacterTokenizer.from_text("Hello, 世界!\n")
        self.assertEqual(
            [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id],
            [0, 1, 2, 3],
        )
        text = "Hello, 世界!\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.decode([], skip_special_tokens=True), "")

    def test_unknown_character_and_special_tokens(self) -> None:
        tokenizer = CharacterTokenizer.from_text("abc")
        encoded = tokenizer.encode("a?")
        self.assertEqual(encoded[1], tokenizer.unk_id)
        self.assertEqual(tokenizer.decode([tokenizer.bos_id, *encoded, tokenizer.eos_id]), "a")
        self.assertEqual(tokenizer.encode("abc", add_special_tokens=True)[0], tokenizer.bos_id)

    def test_save_and_load(self) -> None:
        tokenizer = CharacterTokenizer.from_text("abc\n你好")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = CharacterTokenizer.load(path)
        self.assertEqual(restored.state_dict(), tokenizer.state_dict())
        self.assertEqual(restored.decode(restored.encode("你好\n")), "你好\n")


if __name__ == "__main__":
    unittest.main()
