from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformer.config import load_config  # noqa: E402
from mini_transformer.dataset import CharDataset  # noqa: E402
from mini_transformer.tokenizer import CharacterTokenizer  # noqa: E402
from mini_transformer.train import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny character Transformer.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "tiny.yaml"))
    parser.add_argument("--max-steps", type=int, default=None, help="Override config max_steps")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda")
    parser.add_argument("--resume", default=None, help="Resume from a .pt checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.device is not None:
        config.train.device = args.device
    data_path = ROOT / config.data_path
    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    dataset = CharDataset(text, tokenizer, config.model.block_size, config.train_split)
    print(f"loaded {len(text):,} characters from {data_path}")
    train_model(dataset, tokenizer, config.model, config.train, resume_path=args.resume)


if __name__ == "__main__":
    main()
