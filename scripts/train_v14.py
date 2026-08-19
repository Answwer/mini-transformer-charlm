from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformer.bpe import BPETokenizer  # noqa: E402
from mini_transformer.config import load_config  # noqa: E402
from mini_transformer.dataset import CharDataset  # noqa: E402
from mini_transformer.train import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v14 BPE Transformer.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "v14_bpe.yaml"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.tokenizer != "bpe":
        raise ValueError("train_v14.py requires tokenizer: bpe")
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.device is not None:
        config.train.device = args.device

    data_path = ROOT / config.data_path
    text = data_path.read_text(encoding="utf-8")
    split_index = int(len(text) * config.train_split)
    tokenizer = BPETokenizer.from_text(
        text[:split_index],
        vocab_size=config.bpe_vocab_size,
        min_frequency=config.bpe_min_frequency,
    )
    dataset = CharDataset(text, tokenizer, config.model.block_size, config.train_split)
    print(
        f"loaded {len(text):,} characters from {data_path}; "
        f"BPE vocabulary={tokenizer.vocab_size:,}"
    )
    train_model(dataset, tokenizer, config.model, config.train, resume_path=args.resume)


if __name__ == "__main__":
    main()
