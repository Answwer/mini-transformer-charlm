import unittest

import torch
from torch import nn

from mini_transformer.config import ModelConfig
from mini_transformer.generate import generate
from mini_transformer.tokenizer import CharacterTokenizer


class ScriptedModel(nn.Module):
    """Tiny deterministic model used to test decoding controls."""

    def __init__(self, vocab_size: int, next_ids: list[int]) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = ModelConfig(block_size=8, n_layer=1, n_head=1, n_embd=8)
        self.vocab_size = vocab_size
        self.next_ids = next_ids
        self.calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*input_ids.shape, self.vocab_size), -torch.inf, device=input_ids.device
        )
        next_id = self.next_ids[min(self.calls, len(self.next_ids) - 1)]
        logits[:, -1, next_id] = 0.0
        self.calls += 1
        return logits


class GenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer.from_text("Ab.c")

    def test_sentence_guard_does_not_return_a_trailing_fragment(self) -> None:
        token_id = self.tokenizer.token_to_id
        model = ScriptedModel(
            self.tokenizer.vocab_size,
            [token_id["b"], token_id["."], token_id["c"]],
        )
        result = generate(
            model,
            self.tokenizer,
            prompt="A",
            max_new_tokens=8,
            min_new_tokens=2,
            stop_on_sentence_end=True,
            temperature=0,
        )
        self.assertEqual(result, "Ab.")
        self.assertEqual(model.calls, 2)

    def test_sentence_guard_validates_minimum_length(self) -> None:
        model = ScriptedModel(self.tokenizer.vocab_size, [self.tokenizer.eos_id])
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            generate(
                model,
                self.tokenizer,
                prompt="A",
                max_new_tokens=1,
                min_new_tokens=2,
            )


if __name__ == "__main__":
    unittest.main()
