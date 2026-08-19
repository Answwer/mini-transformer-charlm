from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformer.config import ModelConfig  # noqa: E402
from mini_transformer.generate import generate  # noqa: E402
from mini_transformer.model import CharTransformerLM  # noqa: E402
from mini_transformer.tokenizer import CharacterTokenizer  # noqa: E402
from mini_transformer.utils import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a saved checkpoint.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    model_config = ModelConfig(**checkpoint["model_config"])
    model = CharTransformerLM(tokenizer.vocab_size, model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    result = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
