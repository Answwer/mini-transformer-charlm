import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mini_transformer.config import ModelConfig, TrainConfig
from mini_transformer.dataset import CharDataset
from mini_transformer.tokenizer import CharacterTokenizer
from mini_transformer.train import train_model


class TrainingControlsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = ("The quick brown fox jumps over the lazy dog.\n" * 8)
        tokenizer = CharacterTokenizer.from_text(self.text)
        self.dataset = CharDataset(self.text, tokenizer, block_size=8, train_split=0.75)
        self.tokenizer = tokenizer

    def test_evaluation_windows_are_deterministic(self) -> None:
        first = list(
            self.dataset.iter_evaluation_batches("val", 2, 3, device="cpu")
        )
        second = list(
            self.dataset.iter_evaluation_batches("val", 2, 3, device="cpu")
        )
        self.assertEqual(len(first), len(second))
        for (first_x, first_y), (second_x, second_y) in zip(first, second):
            self.assertTrue(torch.equal(first_x, second_x))
            self.assertTrue(torch.equal(first_y, second_y))

    def test_early_stopping_keeps_best_checkpoint(self) -> None:
        model_config = ModelConfig(block_size=8, n_layer=1, n_head=1, n_embd=8)
        with tempfile.TemporaryDirectory() as directory:
            train_config = TrainConfig(
                batch_size=2,
                learning_rate=0.01,
                max_steps=10,
                eval_interval=1,
                eval_steps=1,
                early_stopping_patience=2,
                device="cpu",
                checkpoint_dir=directory,
            )
            metrics = iter(
                [
                    {"train": 1.0, "val": 1.0},
                    {"train": 0.8, "val": 1.1},
                    {"train": 0.6, "val": 1.2},
                ]
            )
            with patch("mini_transformer.train.estimate_loss", side_effect=metrics):
                result = train_model(
                    self.dataset,
                    self.tokenizer,
                    model_config,
                    train_config,
                )

            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["stop_reason"], "early_stopping")
            self.assertEqual(result["step"], 3)
            self.assertEqual(result["best_step"], 1)
            self.assertTrue(Path(directory, "best.pt").is_file())
            checkpoint = torch.load(
                Path(directory, "last.pt"), map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["best_step"], 1)
            self.assertEqual(checkpoint["tokens_seen"], 3 * 2 * 8)


if __name__ == "__main__":
    unittest.main()
