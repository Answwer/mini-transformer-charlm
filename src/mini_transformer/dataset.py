"""Memory-efficient contiguous token windows for next-token prediction."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .tokenizer import CharacterTokenizer


class CharWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """A view over one contiguous token stream without materializing windows."""

    def __init__(self, ids: Tensor, block_size: int) -> None:
        if ids.ndim != 1:
            raise ValueError("ids must be a one-dimensional tensor")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if ids.numel() < block_size + 1:
            raise ValueError("split must contain at least block_size + 1 tokens")
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return self.ids.numel() - self.block_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        x = self.ids[index : index + self.block_size]
        y = self.ids[index + 1 : index + self.block_size + 1]
        return x, y


class CharDataset:
    """Build the fixed 90/10 contiguous train/validation character split."""

    def __init__(
        self,
        text: str,
        tokenizer: CharacterTokenizer,
        block_size: int = 128,
        train_split: float = 0.9,
    ) -> None:
        if not 0.0 < train_split < 1.0:
            raise ValueError("train_split must be strictly between 0 and 1")
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        split_index = int(len(ids) * train_split)
        if split_index < block_size + 1 or len(ids) - split_index < block_size + 1:
            raise ValueError("Both data splits must contain block_size + 1 characters")
        self.block_size = block_size
        self.train_ids = ids[:split_index]
        self.val_ids = ids[split_index:]
        self.train = CharWindowDataset(self.train_ids, block_size)
        self.val = CharWindowDataset(self.val_ids, block_size)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: CharacterTokenizer,
        block_size: int = 128,
        train_split: float = 0.9,
    ) -> "CharDataset":
        text = Path(path).read_text(encoding="utf-8")
        return cls(text, tokenizer, block_size=block_size, train_split=train_split)

    def split(self, name: str) -> CharWindowDataset:
        if name == "train":
            return self.train
        if name in {"val", "validation"}:
            return self.val
        raise ValueError(f"Unknown split: {name}")

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        source = self.split(split)
        starts = torch.randint(len(source), (batch_size,), generator=generator)
        x = torch.stack([source[start][0] for start in starts.tolist()])
        y = torch.stack([source[start][1] for start in starts.tolist()])
        return x.to(device), y.to(device)

    def iter_evaluation_batches(
        self,
        split: str,
        batch_size: int,
        eval_steps: int,
        device: torch.device | str = "cpu",
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield a fixed, evenly spaced set of evaluation windows.

        Training batches are intentionally random, but validation must not
        change from one checkpoint to the next just because a new random
        sample happened to be drawn.  Evenly spaced starts cover the whole
        split while keeping the evaluation budget bounded by ``eval_steps``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        source = self.split(split)
        count = min(len(source), batch_size * eval_steps)
        max_start = len(source) - 1
        if count == 1:
            starts = torch.zeros(1, dtype=torch.long)
        else:
            starts = torch.linspace(0, max_start, count, dtype=torch.float64).round().long()
            starts = torch.unique_consecutive(starts)
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size].tolist()
            x = torch.stack([source[start][0] for start in batch_starts])
            y = torch.stack([source[start][1] for start in batch_starts])
            yield x.to(device), y.to(device)
