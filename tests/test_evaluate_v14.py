import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_v14 import generation_flags, parse_seeds, split_texts


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

    def test_split_texts_omits_empty_optional_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "format_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "work_records": [
                            {"split": "train", "char_start": 0, "char_end": 5},
                            {"split": "validation", "char_start": 5, "char_end": 10},
                            {"split": "test", "char_start": 10, "char_end": 15},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            splits = split_texts(report_path, "trainvalidtest")
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertNotIn("development_validation", splits)


if __name__ == "__main__":
    unittest.main()
