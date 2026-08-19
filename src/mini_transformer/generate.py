"""Autoregressive text generation for a trained language model."""

from __future__ import annotations

import torch
from torch import nn
import re

from .tokenizer import CharacterTokenizer


_SENTENCE_END = re.compile(r"[.!?][\"')\]}]*$")


def _has_sentence_end(text: str) -> bool:
    """Return whether the generated text currently ends at sentence punctuation."""

    return bool(_SENTENCE_END.search(text.rstrip()))


def generate(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    min_new_tokens: int = 0,
    stop_on_sentence_end: bool = False,
) -> str:
    """Generate text, recomputing the full context at every step.

    ``temperature <= 0`` selects greedy decoding. For sampling, ``top_k``
    restricts the distribution before multinomial sampling. When
    ``stop_on_sentence_end`` is enabled, generation continues until at least
    ``min_new_tokens`` have been produced and then stops at ``.``, ``!`` or
    ``?``. This avoids presenting a truncated final clause as a complete
    result; it is a decoding guard, not a semantic correctness guarantee.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if min_new_tokens < 0:
        raise ValueError("min_new_tokens cannot be negative")
    if min_new_tokens > max_new_tokens:
        raise ValueError("min_new_tokens cannot exceed max_new_tokens")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
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
        for generated_count in range(1, max_new_tokens + 1):
            input_ids = torch.tensor([generated_ids[-block_size:]], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1).item()
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    values, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probabilities = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove = cumulative_probabilities > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, sorted_indices, sorted_logits
                    )
                probabilities = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1).item()
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_id and generated_count >= min_new_tokens:
                break
            if (
                stop_on_sentence_end
                and generated_count >= min_new_tokens
                and _has_sentence_end(tokenizer.decode(generated_ids))
            ):
                break
    return tokenizer.decode(generated_ids)
