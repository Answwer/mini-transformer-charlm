import unittest

import torch

from mini_transformer.attention import CausalSelfAttention


class AttentionTest(unittest.TestCase):
    def test_shapes_and_causal_weights(self) -> None:
        attention = CausalSelfAttention(n_embd=12, n_head=3, block_size=8, dropout=0.0)
        x = torch.randn(2, 5, 12)
        output, weights = attention(x, need_weights=True)
        self.assertEqual(tuple(output.shape), (2, 5, 12))
        self.assertEqual(tuple(weights.shape), (2, 3, 5, 5))
        self.assertTrue(torch.all(weights.triu(diagonal=1) == 0))

    def test_invalid_head_dimension_is_clear(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            CausalSelfAttention(n_embd=10, n_head=3, block_size=8)


if __name__ == "__main__":
    unittest.main()
