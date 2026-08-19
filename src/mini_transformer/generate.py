"""Autoregressive text generation for a trained character language model."""

from __future__ import annotations

import torch
from torch import nn

from .tokenizer import CharacterTokenizer


def generate(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """Generate text, recomputing the full context at every step.

    ``temperature <= 0`` selects greedy decoding. For sampling, ``top_k``
    restricts the distribution before multinomial sampling.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    device = next(model.parameters()).device
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    context_ids = tokenizer.encode(prompt)
    if not context_ids:
        context_ids = [tokenizer.bos_id]
    generated_ids = list(context_ids)
    block_size = model.config.block_size
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated_ids[-block_size:]], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1).item()
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    values, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1).item()
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_id:
                break
    return tokenizer.decode(generated_ids)
