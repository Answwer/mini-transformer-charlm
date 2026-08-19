"""Reproducible training loop and checkpoint helpers."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ModelConfig, TrainConfig
from .dataset import CharDataset
from .model import CharTransformerLM
from .tokenizer import CharacterTokenizer
from .utils import ensure_directory, resolve_device, set_seed


def estimate_loss(
    model: CharTransformerLM,
    dataset: CharDataset,
    train_config: TrainConfig,
    device: torch.device,
    generators: dict[str, torch.Generator],
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for split in ("train", "val"):
            values = []
            for _ in range(train_config.eval_steps):
                x, y = dataset.get_batch(
                    split,
                    train_config.batch_size,
                    device=device,
                    generator=generators[split],
                )
                _, loss = model(x, y)
                values.append(loss.item())
            losses[split] = sum(values) / len(values)
    model.train()
    return losses


def save_checkpoint(
    path: str | Path,
    model: CharTransformerLM,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    tokenizer: CharacterTokenizer,
    step: int,
    best_val_loss: float,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "tokenizer": tokenizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "seed": train_config.seed,
    }
    torch.save(checkpoint, path)


def train_model(
    dataset: CharDataset,
    tokenizer: CharacterTokenizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a model and save best.pt/last.pt in the configured directory."""

    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    model = CharTransformerLM(tokenizer.vocab_size, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    start_step = 0
    best_val_loss = float("inf")
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"Resumed from {resume_path} at step {start_step}")

    checkpoint_dir = ensure_directory(train_config.checkpoint_dir)
    generators = {
        "train": torch.Generator().manual_seed(train_config.seed + 1),
        "val": torch.Generator().manual_seed(train_config.seed + 2),
    }
    model.train()
    print(f"device={device} parameters={model.num_parameters():,}")
    print(f"config model={asdict(model_config)} train={asdict(train_config)}")
    last_metrics: dict[str, float] = {"train": float("nan"), "val": float("nan")}
    for step in range(start_step, train_config.max_steps):
        started = time.perf_counter()
        x, y = dataset.get_batch(
            "train",
            train_config.batch_size,
            device=device,
            generator=generators["train"],
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        if train_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        completed_step = step + 1
        should_evaluate = (
            completed_step % train_config.eval_interval == 0
            or completed_step == train_config.max_steps
        )
        if should_evaluate:
            last_metrics = estimate_loss(model, dataset, train_config, device, generators)
            elapsed = time.perf_counter() - started
            print(
                f"step {completed_step:>5d}/{train_config.max_steps} "
                f"train_loss={last_metrics['train']:.4f} "
                f"val_loss={last_metrics['val']:.4f} "
                f"elapsed={elapsed:.2f}s"
            )
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                model_config,
                train_config,
                tokenizer,
                completed_step,
                best_val_loss,
            )
            if last_metrics["val"] < best_val_loss:
                best_val_loss = last_metrics["val"]
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    model_config,
                    train_config,
                    tokenizer,
                    completed_step,
                    best_val_loss,
                )

    return {
        "model": model,
        "optimizer": optimizer,
        "device": device,
        "step": min(train_config.max_steps, max(start_step, train_config.max_steps)),
        "best_val_loss": best_val_loss,
        "last_metrics": last_metrics,
    }
