import unittest

import torch

from mini_transformer.config import ModelConfig
from mini_transformer.model import CharTransformerLM


class OverfitTest(unittest.TestCase):
    def test_tiny_batch_can_be_memorized(self) -> None:
        torch.manual_seed(7)
        config = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = CharTransformerLM(vocab_size=5, config=config)
        x = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2], [1, 2, 3, 4, 0, 1, 2, 3]])
        y = torch.tensor([[1, 2, 3, 4, 0, 1, 2, 3], [2, 3, 4, 0, 1, 2, 3, 4]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        initial_loss = None
        for _ in range(160):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
        self.assertIsNotNone(initial_loss)
        self.assertLess(loss.item(), initial_loss * 0.15)
        self.assertLess(loss.item(), 0.25)


if __name__ == "__main__":
    unittest.main()
