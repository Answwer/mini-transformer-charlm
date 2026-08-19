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
    generators: dict[str, torch.Generator] | None = None,
) -> dict[str, float]:
    """Evaluate on a deterministic set of windows from each split.

    ``generators`` remains an accepted, unused argument for compatibility
    with the earlier v13 helper.  Validation is deliberately not sampled
    randomly: a checkpoint decision must be based on the same windows every
    time it is evaluated.
    """
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for split in ("train", "val"):
            values = []
            for x, y in dataset.iter_evaluation_batches(
                split,
                train_config.batch_size,
                train_config.eval_steps,
                device=device,
            ):
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
    tokens_seen: int = 0,
    best_step: int = 0,
    evaluations_without_improvement: int = 0,
    metrics: dict[str, float] | None = None,
    train_generator_state: torch.Tensor | None = None,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "tokenizer": tokenizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "tokens_seen": tokens_seen,
        "evaluations_without_improvement": evaluations_without_improvement,
        "metrics": metrics or {},
        "seed": train_config.seed,
    }
    if train_generator_state is not None:
        checkpoint["train_generator_state"] = train_generator_state
    torch.save(checkpoint, path)


def train_model(
    dataset: CharDataset,
    tokenizer: CharacterTokenizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a model and save best.pt/last.pt in the configured directory.

    The returned model is restored from ``best.pt`` when that checkpoint
    exists.  This makes the safe model-selection rule hold for Python callers
    as well as for the generation CLI; callers do not accidentally generate
    from the overfit final step.
    """

    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    model = CharTransformerLM(tokenizer.vocab_size, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    start_step = 0
    best_val_loss = float("inf")
    best_step = 0
    evaluations_without_improvement = 0
    train_generator = torch.Generator().manual_seed(train_config.seed + 1)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        best_val_loss = float(checkpoint["best_val_loss"])
        best_step = int(checkpoint.get("best_step", start_step))
        evaluations_without_improvement = int(
            checkpoint.get("evaluations_without_improvement", 0)
        )
        if "train_generator_state" in checkpoint:
            train_generator.set_state(checkpoint["train_generator_state"])
        print(f"Resumed from {resume_path} at step {start_step}")

    checkpoint_dir = ensure_directory(train_config.checkpoint_dir)
    model.train()
    print(f"device={device} parameters={model.num_parameters():,}")
    print(f"config model={asdict(model_config)} train={asdict(train_config)}")
    last_metrics: dict[str, float] = {"train": float("nan"), "val": float("nan")}
    history: list[dict[str, float | int]] = []
    tokens_seen = start_step * train_config.batch_size * model_config.block_size
    completed_step = start_step
    stop_reason = "max_steps_reached"
    training_started = time.perf_counter()
    for step in range(start_step, train_config.max_steps):
        started = time.perf_counter()
        x, y = dataset.get_batch(
            "train",
            train_config.batch_size,
            device=device,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        if train_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        completed_step = step + 1
        tokens_seen += train_config.batch_size * model_config.block_size
        should_evaluate = (
            completed_step % train_config.eval_interval == 0
            or completed_step == train_config.max_steps
        )
        if should_evaluate:
            last_metrics = estimate_loss(model, dataset, train_config, device)
            improved = last_metrics["val"] < best_val_loss - train_config.early_stopping_min_delta
            if improved:
                best_val_loss = last_metrics["val"]
                best_step = completed_step
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            history.append(
                {
                    "step": completed_step,
                    "train_loss": last_metrics["train"],
                    "val_loss": last_metrics["val"],
                    "tokens_seen": tokens_seen,
                }
            )
            elapsed = time.perf_counter() - started
            print(
                f"step {completed_step:>5d}/{train_config.max_steps} "
                f"train_loss={last_metrics['train']:.4f} "
                f"val_loss={last_metrics['val']:.4f} "
                f"best_step={best_step} "
                f"no_improve={evaluations_without_improvement} "
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
                tokens_seen=tokens_seen,
                best_step=best_step,
                evaluations_without_improvement=evaluations_without_improvement,
                metrics=last_metrics,
                train_generator_state=train_generator.get_state(),
            )
            if improved:
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    model_config,
                    train_config,
                    tokenizer,
                    completed_step,
                    best_val_loss,
                    tokens_seen=tokens_seen,
                    best_step=best_step,
                    evaluations_without_improvement=evaluations_without_improvement,
                    metrics=last_metrics,
                    train_generator_state=train_generator.get_state(),
                )
            if (
                train_config.early_stopping_patience > 0
                and evaluations_without_improvement >= train_config.early_stopping_patience
            ):
                stop_reason = "early_stopping"
                print(
                    "early stopping: "
                    f"validation loss did not improve for {train_config.early_stopping_patience} evaluations"
                )
                break

    best_checkpoint_path = checkpoint_dir / "best.pt"
    if best_checkpoint_path.is_file():
        best_checkpoint = torch.load(
            best_checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(best_checkpoint["model_state"])

    return {
        "model": model,
        "optimizer": optimizer,
        "device": device,
        "step": completed_step,
        "best_val_loss": best_val_loss,
        "last_metrics": last_metrics,
        "best_step": best_step,
        "tokens_seen": tokens_seen,
        "history": history,
        "stopped_early": stop_reason == "early_stopping",
        "stop_reason": stop_reason,
        "training_seconds": time.perf_counter() - training_started,
    }
