"""Evaluate a v14/v4 BPE checkpoint on fixed expanded, test, and old splits."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformer.bpe import BPETokenizer  # noqa: E402
from mini_transformer.config import ModelConfig, TrainConfig  # noqa: E402
from mini_transformer.dataset import CharDataset  # noqa: E402
from mini_transformer.generate import generate  # noqa: E402
from mini_transformer.model import CausalTransformerLM  # noqa: E402
from mini_transformer.train import estimate_validation_loss  # noqa: E402
from mini_transformer.utils import resolve_device  # noqa: E402


PROMPTS = [
    "The king",
    "Before we proceed",
    "To be, or not to be:",
    "Friends, Romans, countrymen",
    "All the world's a stage",
]


def split_texts(report_path: Path, text: str) -> dict[str, str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("work_records", [])
    if not records:
        raise ValueError("format report has no work_records")
    groups: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for record in records:
        split = str(record.get("split", ""))
        if split not in groups:
            raise ValueError(f"invalid split: {split}")
        groups[split].append(text[int(record["char_start"]) : int(record["char_end"])])
    return {key: "\n\n".join(value) for key, value in groups.items()}


def split_loss(
    model: CausalTransformerLM,
    text: str,
    tokenizer: BPETokenizer,
    block_size: int,
    train_config: TrainConfig,
    device: torch.device,
) -> float:
    dataset = CharDataset.from_split_texts(text, text, tokenizer, block_size)
    return estimate_validation_loss(model, dataset, train_config, device)


def generation_flags(text: str) -> dict[str, int | bool]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    trigrams = [tuple(text.split()[index : index + 3]) for index in range(max(0, len(text.split()) - 2))]
    repeated_trigrams = len(trigrams) - len(set(trigrams))
    duplicate_lines = len(lines) - len(set(lines))
    return {
        "sentence_terminated": bool(re.search(r"[.!?][\"')\]}]*$", text.rstrip())),
        "repeated_trigram_count": repeated_trigrams,
        "duplicate_line_count": duplicate_lines,
        "half_sentence_tail": not bool(re.search(r"[.!?:;][\"')\]}]*$", text.rstrip())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expanded", default=str(ROOT / "data/expanded/dataset-expand.txt"))
    parser.add_argument("--report", default=str(ROOT / "data/expanded/format_report.json"))
    parser.add_argument("--old-data", default=str(ROOT / "data/tiny_shakespeare.txt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    model_config = ModelConfig(**checkpoint["model_config"])
    model = CausalTransformerLM(tokenizer.vocab_size, model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    train_config = TrainConfig(
        batch_size=16,
        eval_steps=args.eval_steps,
        device=str(device),
        max_steps=1,
    )
    expanded = split_texts(Path(args.report), Path(args.expanded).read_text(encoding="utf-8"))
    old_text = Path(args.old_data).read_text(encoding="utf-8")
    metrics = {
        "expanded_validation_loss": split_loss(model, expanded["validation"], tokenizer, model_config.block_size, train_config, device),
        "expanded_test_loss": split_loss(model, expanded["test"], tokenizer, model_config.block_size, train_config, device),
        "old_validation_loss": estimate_validation_loss(
            model,
            CharDataset(old_text, tokenizer, model_config.block_size, 0.9),
            train_config,
            device,
        ),
    }
    metrics["expanded_validation_ppl"] = math.exp(metrics["expanded_validation_loss"])
    metrics["expanded_test_ppl"] = math.exp(metrics["expanded_test_loss"])
    metrics["old_validation_ppl"] = math.exp(metrics["old_validation_loss"])
    generations: dict[str, dict[str, object]] = {}
    if not args.skip_generation:
        for prompt in PROMPTS:
            generated = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_new_tokens=120,
                min_new_tokens=24,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.05,
                seed=args.seed,
                stop_on_sentence_end=True,
            )
            generations[prompt] = {"text": generated, "flags": generation_flags(generated)}
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": checkpoint.get("step"),
        "best_step": checkpoint.get("best_step"),
        "tokens_seen": checkpoint.get("tokens_seen"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": metrics,
        "generations": generations,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
