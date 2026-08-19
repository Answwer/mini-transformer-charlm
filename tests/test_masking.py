import unittest

import torch

from mini_transformer.masking import causal_mask, scaled_dot_product_attention


class MaskingTest(unittest.TestCase):
    def test_causal_mask_only_exposes_past_and_present(self) -> None:
        mask = causal_mask(4)
        expected = torch.tensor(
            [[True, False, False, False], [True, True, False, False],
             [True, True, True, False], [True, True, True, True]]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_attention_weights_are_normalized_and_future_is_zero(self) -> None:
        query = torch.randn(2, 3, 4, 5)
        key = torch.randn(2, 3, 4, 5)
        value = torch.randn(2, 3, 4, 5)
        output, weights = scaled_dot_product_attention(
            query,
            key,
            value,
            mask=causal_mask(4).view(1, 1, 4, 4),
            return_weights=True,
        )
        self.assertEqual(tuple(output.shape), (2, 3, 4, 5))
        self.assertEqual(tuple(weights.shape), (2, 3, 4, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 4)))
        self.assertTrue(torch.all(weights.triu(diagonal=1) == 0))


if __name__ == "__main__":
    unittest.main()
