"""Causal masking and scaled dot-product attention primitives."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """Return a boolean [T, T] mask where True means the position is visible."""

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
    mask: Tensor | None = None,
    return_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute attention for tensors shaped [B, H, T, D]."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, H, T, D]")
    if query.shape[:-1] != key.shape[:-1] or key.shape[:-1] != value.shape[:-1]:
        raise ValueError("query, key, and value must have matching batch/head/length shapes")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=training)
    output = weights @ value
    if return_weights:
        return output, weights
    return output
