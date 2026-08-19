"""Multi-head causal self-attention implemented from basic PyTorch layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .masking import causal_mask, scaled_dot_product_attention


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            causal_mask(block_size).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, channels = x.shape
        return x.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, -1)

    def forward(self, x: Tensor, need_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("attention input must have shape [B, T, C]")
        if x.shape[1] > self.causal_mask.shape[-1]:
            raise ValueError("sequence length exceeds configured block_size")
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_dropout,
            training=self.training,
            mask=self.causal_mask[:, :, : x.shape[1], : x.shape[1]],
            return_weights=need_weights,
        )
        if need_weights:
            attended, weights = attended
        output = self.resid_dropout(self.out_proj(self._merge_heads(attended)))
        if need_weights:
            return output, weights
        return output
