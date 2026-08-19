"""Configuration loading without requiring a YAML dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 1000
    eval_interval: int = 100
    eval_steps: int = 20
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps < 0:
            raise ValueError("max_steps cannot be negative")
        if self.eval_interval <= 0 or self.eval_steps <= 0:
            raise ValueError("eval_interval and eval_steps must be positive")
        if self.grad_clip < 0:
            raise ValueError("grad_clip cannot be negative")


@dataclass
class ExperimentConfig:
    data_path: str = "data/tiny_shakespeare.txt"
    train_split: float = 0.9
    tokenizer: str = "char"
    bpe_vocab_size: int = 512
    bpe_min_frequency: int = 2
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be strictly between 0 and 1")
        if self.tokenizer not in {"char", "bpe"}:
            raise ValueError("tokenizer must be 'char' or 'bpe'")
        if self.bpe_vocab_size <= 4:
            raise ValueError("bpe_vocab_size must be greater than 4")
        if self.bpe_min_frequency <= 0:
            raise ValueError("bpe_min_frequency must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_path": self.data_path,
            "train_split": self.train_split,
            "tokenizer": self.tokenizer,
            "bpe_vocab_size": self.bpe_vocab_size,
            "bpe_min_frequency": self.bpe_min_frequency,
            "model": asdict(self.model),
            "train": asdict(self.train),
        }


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Read the deliberately flat YAML subset used by configs/tiny.yaml."""

    values: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid config line {line_number}: {line!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing config key on line {line_number}")
        values[key] = _parse_scalar(raw_value.split(" #", 1)[0])
    return values


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a flat YAML config and validate all model/training settings."""

    path = Path(path)
    raw = _read_simple_yaml(path)
    model_keys = {"block_size", "n_layer", "n_head", "n_embd", "dropout"}
    train_keys = {
        "batch_size",
        "learning_rate",
        "max_steps",
        "eval_interval",
        "eval_steps",
        "grad_clip",
        "seed",
        "device",
        "checkpoint_dir",
    }
    model = ModelConfig(**{key: raw[key] for key in model_keys if key in raw})
    train = TrainConfig(**{key: raw[key] for key in train_keys if key in raw})
    return ExperimentConfig(
        data_path=str(raw.get("data_path", "data/tiny_shakespeare.txt")),
        train_split=float(raw.get("train_split", 0.9)),
        tokenizer=str(raw.get("tokenizer", "char")),
        bpe_vocab_size=int(raw.get("bpe_vocab_size", 512)),
        bpe_min_frequency=int(raw.get("bpe_min_frequency", 2)),
        model=model,
        train=train,
    )
