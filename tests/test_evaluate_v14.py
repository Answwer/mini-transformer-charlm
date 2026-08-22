import unittest

from scripts.evaluate_v14 import generation_flags, parse_seeds


class EvaluateV14Test(unittest.TestCase):
    def test_generation_flags_identify_half_sentence_and_speaker_switch(self) -> None:
        flags = generation_flags("HAMLET:\nTo be.\nOPHELIA:\nWith a broken")
        self.assertTrue(flags["half_sentence_tail"])
        self.assertEqual(flags["speaker_label_count"], 2)
        self.assertEqual(flags["speaker_label_switch_count"], 1)

    def test_parse_seeds_requires_unique_values(self) -> None:
        self.assertEqual(parse_seeds("7, 17,29"), [7, 17, 29])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            parse_seeds("7,7")


if __name__ == "__main__":
    unittest.main()
