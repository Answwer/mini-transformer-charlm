from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformer.bpe import BPETokenizer  # noqa: E402
from mini_transformer.config import load_config  # noqa: E402
from mini_transformer.dataset import CharDataset, ReplayWindowDataset  # noqa: E402
from mini_transformer.train import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v5 BPE Transformer.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "v5_bpe_expanded_optimize.yaml"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--reset-best", action="store_true", help="reset best loss for the new validation set")
    parser.add_argument("--keep-best", action="store_true", help="keep parent validation best state when resuming")
    parser.add_argument("--replay-data-path", default=None)
    parser.add_argument("--old-data-ratio", type=float, default=None)
    args = parser.parse_args()

    if args.reset_best and args.keep_best:
        raise ValueError("--reset-best and --keep-best are mutually exclusive")

    config = load_config(args.config)
    if config.tokenizer != "bpe":
        raise ValueError("train_v5.py requires tokenizer: bpe")
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.device is not None:
        config.train.device = args.device

    if args.replay_data_path is not None:
        config.replay_data_path = args.replay_data_path
    if args.old_data_ratio is not None:
        config.old_data_ratio = args.old_data_ratio
    if args.reset_best:
        config.reset_best_on_resume = True
    elif args.keep_best:
        config.reset_best_on_resume = False

    def resolve_repo_path(raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else ROOT / path

    def data_metadata(path: Path, text: str, tokenizer: BPETokenizer) -> dict[str, object]:
        encoded = tokenizer.encode(text)
        unknown = sum(token_id == tokenizer.unk_id for token_id in encoded)
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "utf8_bytes": path.stat().st_size,
            "characters": len(text),
            "tokens": len(encoded),
            "unknown_tokens": unknown,
            "unknown_ratio": unknown / len(encoded) if encoded else 0.0,
        }

    def split_texts(path: Path, text: str) -> tuple[dict[str, str], dict[str, object]]:
        report = json.loads(path.read_text(encoding="utf-8"))
        split_info = report.get("splits", {})
        work_groups = split_info.get("works", {})
        if not isinstance(work_groups, dict) or not work_groups.get("train"):
            raise ValueError("split report must contain a non-empty train work list")
        # The report stores exact offsets; use those rather than inferring titles.
        records = report.get("work_records", [])
        if not records:
            raise ValueError("split report is missing work_records with character offsets")
        allowed_splits = {"train", "validation", "test", "development_validation"}
        texts: dict[str, list[str]] = {
            "train": [],
            "validation": [],
            "test": [],
            "development_validation": [],
        }
        for record in records:
            split = str(record.get("split", ""))
            if split not in allowed_splits:
                raise ValueError(f"invalid work split: {split}")
            start = int(record["char_start"])
            end = int(record["char_end"])
            texts[split].append(text[start:end])
        metadata = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "work_counts": {name: len(value) for name, value in texts.items()},
        }
        return (
            {
                name: "\n\n".join(value)
                for name, value in texts.items()
                if value
            },
            metadata,
        )

    data_path = resolve_repo_path(config.data_path)
    text = data_path.read_text(encoding="utf-8")
    manifest_path = None
    if config.development_manifest_path:
        manifest_path = resolve_repo_path(config.development_manifest_path)
    elif config.split_manifest_path:
        manifest_path = resolve_repo_path(config.split_manifest_path)
    split_bundle: dict[str, str] | None = None
    split_metadata: dict[str, object] = {}
    if manifest_path is not None:
        split_bundle, split_metadata = split_texts(manifest_path, text)

    resume_path = resolve_repo_path(args.resume) if args.resume else None
    checkpoint: dict[str, object] | None = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        tokenizer_state = checkpoint.get("tokenizer")
        if not isinstance(tokenizer_state, dict) or tokenizer_state.get("type") != "bpe":
            raise ValueError("--resume must point to a BPE checkpoint")
        tokenizer = BPETokenizer.from_state_dict(tokenizer_state)
        if tokenizer.vocab_size != config.bpe_vocab_size:
            raise ValueError(
                "new config bpe_vocab_size must match the resumed checkpoint vocabulary"
            )
        checkpoint_model_config = checkpoint.get("model_config")
        if checkpoint_model_config != config.model.__dict__:
            raise ValueError(
                "new config model settings must match the resumed checkpoint exactly"
            )
        tokenizer_source = "parent_checkpoint"
    else:
        text_for_tokenizer = (
            split_bundle["train"]
            if split_bundle is not None
            else text[: int(len(text) * config.train_split)]
        )
        tokenizer = BPETokenizer.from_text(
            text_for_tokenizer,
            vocab_size=config.bpe_vocab_size,
            min_frequency=config.bpe_min_frequency,
        )
        tokenizer_source = "new_train_text"

    replay_path = (
        resolve_repo_path(config.replay_data_path)
        if config.replay_data_path
        else None
    )
    replay_text = replay_path.read_text(encoding="utf-8") if replay_path else None
    if split_bundle is not None:
        train_text = split_bundle["train"]
        val_text = split_bundle["validation"]
        dataset = CharDataset.from_split_texts(
            train_text, val_text, tokenizer, config.model.block_size
        )
        if replay_text is not None and config.old_data_ratio > 0.0:
            dataset.replay_train = CharDataset(
                replay_text, tokenizer, config.model.block_size, config.train_split
            ).train
            dataset.train_mixture = ReplayWindowDataset(
                dataset.train, dataset.replay_train, config.old_data_ratio
            )
    else:
        dataset = CharDataset(
            text,
            tokenizer,
            config.model.block_size,
            config.train_split,
            replay_text=replay_text,
            old_data_ratio=config.old_data_ratio,
        )
    checkpoint_dir = resolve_repo_path(config.train.checkpoint_dir)
    if resume_path is not None and checkpoint_dir.resolve() == resume_path.parent.resolve():
        raise ValueError(
            "refusing to write a resumed experiment into the parent checkpoint directory; "
            "use a new checkpoint_dir"
        )
    metadata: dict[str, object] = {
        "experiment": "v5_bpe_expanded_optimize",
        "tokenizer_source": tokenizer_source,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "data": data_metadata(data_path, text, tokenizer),
        "old_data_ratio": config.old_data_ratio,
        "split": split_metadata,
    }
    if manifest_path is not None:
        metadata["split_manifest_path"] = str(manifest_path.resolve())
        metadata["development_validation_present"] = (
            split_bundle is not None and "development_validation" in split_bundle
        )
    if replay_path is not None and replay_text is not None:
        metadata["replay_data"] = data_metadata(replay_path, replay_text, tokenizer)
    if checkpoint is not None:
        metadata["parent_checkpoint_step"] = int(checkpoint["step"])
        metadata["parent_checkpoint_best_val_loss"] = float(checkpoint["best_val_loss"])
    print(
        f"loaded {len(text):,} characters from {data_path}; "
        f"BPE vocabulary={tokenizer.vocab_size:,}; "
        f"unk_ratio={metadata['data']['unknown_ratio']:.6f}"
    )
    if replay_path is not None:
        print(
            f"replay={replay_path} old_data_ratio={config.old_data_ratio:.2f}; "
            f"replay_unk_ratio={metadata['replay_data']['unknown_ratio']:.6f}"
        )
    extra_eval_datasets = {}
    if split_bundle is not None and "development_validation" in split_bundle:
        extra_eval_datasets["development_validation"] = CharDataset.from_split_texts(
            split_bundle["train"],
            split_bundle["development_validation"],
            tokenizer,
            config.model.block_size,
        )
    if replay_path is not None:
        extra_eval_datasets["old_val_loss"] = CharDataset(
            replay_text or "", tokenizer, config.model.block_size, config.train_split
        )
    train_model(
        dataset,
        tokenizer,
        config.model,
        config.train,
        resume_path=resume_path,
        reset_best_on_resume=config.reset_best_on_resume,
        checkpoint_metadata=metadata,
        extra_eval_datasets=extra_eval_datasets,
    )


if __name__ == "__main__":
    main()
