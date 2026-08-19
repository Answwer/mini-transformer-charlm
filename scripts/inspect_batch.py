from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformer.config import load_config  # noqa: E402
from mini_transformer.dataset import CharDataset  # noqa: E402
from mini_transformer.tokenizer import CharacterTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one character-level training batch.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "tiny.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    data_path = ROOT / config.data_path
    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    dataset = CharDataset(text, tokenizer, config.model.block_size, config.train_split)
    x, y = dataset.get_batch(
        "train", batch_size=2, device="cpu", generator=torch.Generator().manual_seed(config.train.seed)
    )
    print(f"data={data_path} characters={len(text)} vocab_size={tokenizer.vocab_size}")
    print(f"train_chars={len(dataset.train_ids)} val_chars={len(dataset.val_ids)}")
    print(f"x.shape={tuple(x.shape)} y.shape={tuple(y.shape)}")
    print(f"x[0][:32]={x[0, :32].tolist()}")
    print(f"y[0][:32]={y[0, :32].tolist()}")
    print(f"x_text={tokenizer.decode(x[0, :32].tolist())!r}")
    print(f"y_text={tokenizer.decode(y[0, :32].tolist())!r}")


if __name__ == "__main__":
    main()
