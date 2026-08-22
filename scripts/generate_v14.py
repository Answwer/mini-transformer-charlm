from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformer.bpe import BPETokenizer  # noqa: E402
from mini_transformer.config import ModelConfig  # noqa: E402
from mini_transformer.generate import generate  # noqa: E402
from mini_transformer.model import CausalTransformerLM  # noqa: E402
from mini_transformer.utils import resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a v14 BPE checkpoint.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "v14_bpe" / "best.pt"))
    parser.add_argument("--prompt", default="To be, or not to be:")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=24,
        help="do not stop at sentence punctuation before this many new tokens",
    )
    parser.add_argument(
        "--stop-at-sentence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop after a complete terminal-punctuation boundary (default: on)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-last",
        action="store_true",
        help="allow the intentionally overfit last.pt checkpoint for comparison",
    )
    args = parser.parse_args()

    if Path(args.checkpoint).name.lower() == "last.pt" and not args.allow_last:
        raise ValueError(
            "Refusing to generate from last.pt; use best.pt, or add --allow-last "
            "only for an explicit overfitting comparison."
        )

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    model_config = ModelConfig(**checkpoint["model_config"])
    model = CausalTransformerLM(tokenizer.vocab_size, model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(
        generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            min_new_tokens=args.min_new_tokens,
            stop_on_sentence_end=args.stop_at_sentence,
        )
    )


if __name__ == "__main__":
    main()
