"""Kaggle v1 launcher for the v14 BPE experiment.

The launcher fetches the public GitHub snapshot only to reproduce the exact
repository contents. The training code reads the bundled dataset and performs
no dataset download.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_URL = "https://github.com/Answwer/mini-transformer-charlm.git"
ROOT = Path("/kaggle/working/mini-transformer-charlm")


def _restart_with_p100_build() -> None:
    """Use a wheel that still contains kernels for Kaggle's Tesla P100."""

    if os.environ.get("MINI_TRANSFORMER_TORCH_CHECKED") == "1":
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    supported = set(torch.cuda.get_arch_list())
    if capability[0] >= 7 or "sm_60" in supported:
        return
    print(f"P100 capability {capability} is not supported by {torch.__version__}; trying compatible wheel")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "torch==2.5.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        check=False,
        timeout=900,
    )
    if result.returncode == 0:
        environment = dict(os.environ)
        environment["MINI_TRANSFORMER_TORCH_CHECKED"] = "1"
        os.execve(sys.executable, [sys.executable, __file__], environment)
    print("Compatible wheel was unavailable; continuing with CPU fallback")


def _prepare_repo() -> None:
    if not (ROOT / ".git").is_dir():
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], check=True)


def main() -> None:
    _restart_with_p100_build()
    _prepare_repo()
    sys.path.insert(0, str(ROOT / "src"))

    import torch

    from mini_transformer.bpe import BPETokenizer
    from mini_transformer.config import load_config
    from mini_transformer.dataset import CharDataset
    from mini_transformer.generate import generate
    from mini_transformer.model import CausalTransformerLM
    from mini_transformer.train import train_model

    print("=== environment ===")
    print(f"python={sys.version}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
        print(f"cuda_capability={torch.cuda.get_device_capability()}")

    print("=== tests ===")
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=ROOT,
        check=True,
    )

    config = load_config(ROOT / "configs" / "v14_bpe.yaml")
    use_cuda = torch.cuda.is_available() and "sm_60" in set(torch.cuda.get_arch_list())
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        use_cuda = True
    config.train.device = "cuda" if use_cuda else "cpu"
    config.train.checkpoint_dir = str(ROOT / "checkpoints" / "v14_bpe")
    text = (ROOT / config.data_path).read_text(encoding="utf-8")
    split_index = int(len(text) * config.train_split)
    tokenizer = BPETokenizer.from_text(
        text[:split_index],
        vocab_size=config.bpe_vocab_size,
        min_frequency=config.bpe_min_frequency,
    )
    dataset = CharDataset(text, tokenizer, config.model.block_size, config.train_split)

    print("=== training ===")
    print(f"tokenizer=bpe vocab_size={tokenizer.vocab_size}")
    result = train_model(dataset, tokenizer, config.model, config.train)
    checkpoint_path = ROOT / "checkpoints" / "v14_bpe" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=result["device"], weights_only=False)
    restored_tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    restored_model = CausalTransformerLM(
        restored_tokenizer.vocab_size,
        config.model,
    ).to(result["device"])
    restored_model.load_state_dict(checkpoint["model_state"])

    generated: dict[str, str] = {}
    for temperature in (0.5, 0.7, 0.9):
        generated[str(temperature)] = generate(
            restored_model,
            restored_tokenizer,
            prompt="To be, or not to be:",
            max_new_tokens=120,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.05,
            seed=7,
        )

    summary = {
        "device": str(result["device"]),
        "torch": torch.__version__,
        "steps": result["step"],
        "max_steps": config.train.max_steps,
        "best_step": result["best_step"],
        "tokens_seen": result["tokens_seen"],
        "stopped_early": result["stopped_early"],
        "stop_reason": result["stop_reason"],
        "training_seconds": result["training_seconds"],
        "best_val_loss": result["best_val_loss"],
        "last_metrics": result["last_metrics"],
        "perplexity": float(torch.exp(torch.tensor(result["best_val_loss"]))),
        "tokenizer": "bpe",
        "vocab_size": restored_tokenizer.vocab_size,
        "generated_text": generated,
    }
    (ROOT / "v14_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("=== generation ===")
    for temperature, sample in generated.items():
        print(f"--- temperature={temperature} ---")
        print(sample)
    print("=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
