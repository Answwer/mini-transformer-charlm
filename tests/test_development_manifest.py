from __future__ import annotations

import json
import unittest
from pathlib import Path

from mini_transformer.config import load_config
from scripts.create_development_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]


class DevelopmentManifestTests(unittest.TestCase):
    def test_manifest_moves_only_train_works(self) -> None:
        report = json.loads(
            (ROOT / "data/expanded/format_report.json").read_text(encoding="utf-8")
        )
        derived = build_manifest(report, development_count=3)
        original = report["splits"]["works"]
        updated = derived["splits"]["works"]
        self.assertEqual(updated["validation"], original["validation"])
        self.assertEqual(updated["test"], original["test"])
        self.assertEqual(len(updated["development_validation"]), 3)
        self.assertEqual(
            set(updated["train"]) | set(updated["development_validation"]),
            set(original["train"]),
        )
        self.assertTrue(
            set(updated["train"]).isdisjoint(updated["development_validation"])
        )

    def test_v9_config_uses_development_selection(self) -> None:
        config = load_config(ROOT / "configs/v9_bpe1024_development.yaml")
        self.assertEqual(config.bpe_vocab_size, 1024)
        self.assertEqual(config.development_manifest_path, "data/expanded/development_manifest_v9.json")
        self.assertEqual(config.train.selection_extra_metric, "development_validation")
        self.assertAlmostEqual(config.train.selection_extra_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
