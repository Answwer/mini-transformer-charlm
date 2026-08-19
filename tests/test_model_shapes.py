import unittest

import torch

from mini_transformer.config import ModelConfig
from mini_transformer.model import CharTransformerLM


class ModelShapeTest(unittest.TestCase):
    def test_forward_loss_and_backward(self) -> None:
        config = ModelConfig(block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
        model = CharTransformerLM(vocab_size=11, config=config)
        input_ids = torch.randint(0, 11, (3, 8))
        targets = torch.roll(input_ids, shifts=-1, dims=1)
        logits, loss = model(input_ids, targets)
        self.assertEqual(tuple(logits.shape), (3, 8, 11))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_sequence_limit_and_config_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(block_size=8, n_layer=1, n_head=3, n_embd=16)
        model = CharTransformerLM(7, ModelConfig(block_size=4, n_layer=1, n_head=1, n_embd=8))
        with self.assertRaisesRegex(ValueError, "block_size"):
            model(torch.zeros(1, 5, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
