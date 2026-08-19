# Mini Transformer Character LM

This repository is a small, self-contained character-level causal language
model built from PyTorch primitives. It trains on the included tiny
Shakespeare corpus, evaluates on a contiguous held-out suffix, saves a
reproducible checkpoint, and generates Shakespeare-like character sequences.

The code is intentionally explicit: the data windows, causal mask, Q/K/V
projections, attention weights, next-token loss, training loop, and sampling
logic are all easy to inspect. It does not download a dataset at runtime and
does not use a pretrained model.

## Requirements

- Python 3.10 or newer
- PyTorch 2.x (CPU is sufficient; CUDA is used automatically when available)

The project metadata intentionally does not install runtime dependencies for
you. Install a PyTorch build appropriate for your machine by following the
official [PyTorch installation selector](https://pytorch.org/get-started/locally/).
For a CPU-only environment, a typical command is:

```bash
python -m pip install torch
```

The source tree is runnable without an editable install because the scripts
add `src/` to `sys.path`. An editable install is optional:

```bash
python -m pip install -e .
```

## Quick start

Run the batch inspection first:

```bash
python scripts/inspect_batch.py --config configs/tiny.yaml
```

Run a short CPU smoke training:

```bash
python scripts/train.py --config configs/tiny.yaml --device cpu --max-steps 200
```

The normal tiny configuration uses 1,000 steps. It writes only local
artifacts to `checkpoints/`; `.pt` files are ignored by Git.

Generate from the best checkpoint:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "The " \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 20 \
  --seed 7 \
  --device cpu
```

Greedy decoding is selected with `--temperature 0`; sampling is selected by
using a positive temperature. `--top-k` may be omitted to sample from the
full vocabulary.

## Repository layout

```text
mini-transformer-charlm/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── configs/tiny.yaml
├── data/
│   ├── tiny_shakespeare.txt
│   └── DATASET.md
├── src/mini_transformer/
│   ├── config.py       # dataclasses and dependency-free config reader
│   ├── tokenizer.py    # deterministic character vocabulary
│   ├── dataset.py      # contiguous 90/10 split and sliding windows
│   ├── masking.py      # boolean causal mask and scaled attention
│   ├── attention.py    # multi-head CausalSelfAttention
│   ├── model.py        # positional encoding, blocks, and LM head
│   ├── train.py        # AdamW loop, evaluation, and checkpoints
│   ├── generate.py     # autoregressive decoding
│   └── utils.py
├── scripts/
│   ├── train.py
│   ├── generate.py
│   └── inspect_batch.py
├── tests/
└── checkpoints/.gitkeep
```

## Data flow and tensor shapes

The tokenizer maps every character to one id. The four reserved ids are
stable: `<pad>=0`, `<bos>=1`, `<eos>=2`, and `<unk>=3`. The remaining
characters are sorted by Unicode code point.

The full character stream is split without shuffling: the first 90% is train
and the final 10% is validation. For each sampled offset `i`, the dataset
returns:

```text
x = ids[i : i + T]       # [T]
y = ids[i + 1 : i + T + 1] # [T]
```

After batching, both are `[B, T]`. The model looks up token embeddings and
adds sinusoidal positions, producing `[B, T, C]`. Each attention layer splits
this into `[B, H, T, D]`, computes scores `[B, H, T, T]`, masks all future
positions, and merges heads back to `[B, T, C]`. The final vocabulary
projection returns logits `[B, T, V]`; cross entropy compares each logit at
position `t` with the target character at position `t` (which is the next
character relative to the original stream).

The Transformer blocks use Pre-LayerNorm residual connections:

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

This is a deliberate small-model stability choice. It differs from the
Post-LN ordering in the original Transformer paper, where normalization is
placed after each residual addition. The FFN uses ReLU to keep the reference
implementation close to that paper; changing it to GELU is a modern variant,
not an accidental implementation detail.

## Default configuration

| Setting | Value |
| --- | ---: |
| block size | 128 |
| layers | 2 |
| attention heads | 4 |
| embedding width | 128 |
| dropout | 0.0 |
| batch size | 32 |
| learning rate | 3e-4 |
| training steps | 1,000 |

The implementation asserts that `n_embd % n_head == 0` and checks sequence
lengths before attention. `--debug-shapes` is not needed because
`inspect_batch.py` exposes the data boundary and the model tests cover the
remaining shapes; keeping forward passes free of print statements makes the
training log readable.

## Checkpoints and reproducibility

`best.pt` and `last.pt` contain model and optimizer state, model/training
configuration, tokenizer vocabulary, current step, best validation loss, and
the seed. Loading uses `map_location`, so a CPU machine can load a checkpoint
created on CUDA. Training seeds Python and PyTorch (including CUDA when
available); batch sampling uses dedicated seeded generators for train and
validation.

Resume a run with:

```bash
python scripts/train.py --config configs/tiny.yaml \
  --device cpu --resume checkpoints/last.pt
```

## Tests and debugging order

Run all tests with the standard library test runner:

```bash
python -m unittest discover -s tests -t . -v
```

If `pytest` is already installed, the same files can also be collected with:

```bash
python -m pytest -q
```

The tiny overfit test is the most useful first diagnostic. If it fails,
inspect the right-shifted targets, the upper-triangular causal mask, the
`[B, H, T, D]` reshape, the logits/targets flattening, the learning rate, and
whether the model is in training mode—in that order.

## Known limitations and next steps

- The tokenizer is character-level, so the vocabulary is simple but sequences
  are longer than with BPE or SentencePiece.
- Generation recomputes the entire context at every step and has no KV cache;
  this is intentionally straightforward and is not production optimized.
- The project is a teaching/reference implementation, not a production LM.
- Future extensions could add BPE/SentencePiece, Hugging Face `datasets`,
  `accelerate`, mixed precision, benchmark tooling, and PEFT/LoRA adapters.
