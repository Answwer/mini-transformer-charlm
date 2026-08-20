# Mini Transformer Language Model

This repository contains two small, self-contained causal language-model
paths built from PyTorch primitives: v13 character-level training for a clear
baseline, and v14 BPE training for better word boundaries and local text
coherence. Both paths train on the included tiny Shakespeare corpus, evaluate
on a held-out suffix, save reproducible checkpoints, and generate new text.

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

## v14 BPE experiment

The original v13 character-level path remains unchanged. The v14 path adds a
dependency-free BPE tokenizer and a larger small Transformer while reusing the
same attention, masking, dataset-window, checkpoint, and test code:

```bash
python scripts/train_v14.py --config configs/v14_bpe.yaml --device cuda
python scripts/generate_v14.py \
  --checkpoint checkpoints/v14_bpe/best.pt \
  --prompt "To be, or not to be:" \
  --max-new-tokens 96 --min-new-tokens 24 \
  --temperature 0.7 --top-k 40 --top-p 0.9 --seed 7
```

The v14 checkpoint uses the current bundled Shakespeare dataset. It is an
educational from-scratch model: BPE improves word boundaries and local
coherence, but it does not guarantee that every generated sentence has
complete semantics. The default v14 configuration also uses validation-based
early stopping, so the final training step is not automatically treated as
the best model.

Both generation CLIs default to `best.pt` and refuse `last.pt` unless
`--allow-last` is explicitly supplied for an overfitting comparison.
The v14 CLI also stops at terminal punctuation after a minimum continuation
length, so the displayed result is less likely to end in a half sentence.

For example, the input prompt `To be, or not to be:` is extended by predicting
one BPE token at a time. A successful run can produce speaker labels, whole
words, punctuation, and line breaks; it is not translating the prompt or
answering a question. It is learning the statistical continuation style of
the training corpus.

## Historical v14 and current v4

The original from-scratch v14 run is preserved as a read-only historical
reference. Its checkpoint, Kaggle notebook, and output bundle live under the
local `work/history_v14_*` directories and are never used as the active
training output.

The active expanded-data result is v4:

- Kaggle notebook: [Mini Transformer BPE LM Expanded Optimize v4](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-optimize-v4)
- Hardware: Tesla P100 GPU; PyTorch: `2.5.1+cu124`
- Current local suite: `21/21 passed`
- Best step: `60,000`; tokens seen: `245,760,000`
- Expanded validation loss: `2.350902`; perplexity: `10.495`
- Historical v14 validation loss: `2.3789`; perplexity: `10.79`
- Old-corpus validation loss: `1.921622`
- v4 best checkpoint SHA-256: `369d43e5428f8229fda84a5bcdaee9a20ad11d01ab1a6f64091aed31b9d4874b`

The v4 result is the active comparison point and is kept in its own
`work/kaggle_expanded_optimize_output_v4` directory. The historical v14
artifacts remain available for fixed-prompt comparison and are not overwritten.

The historical v14 validation loss rose after its best checkpoint, which is
the expected overfitting signal. Both historical v14 and v4 generation load
`best.pt`, not `last.pt`.

Example generated output from the best checkpoint:

```text
To be, or not to be:
And now, I am a fellow, if I am not of you.

DUKE OF YORK:
Why, I know the title of the king before him,
And send it against the pattern of the battlements,
Should not be full of sorrow to the foe.
```

This is still a deliberately small from-scratch model. The output is more
readable than the v13 character baseline, but it is not a guarantee of
complete sentence-level semantics.

The sentence boundary guard only removes avoidable truncation. It cannot
verify whether a sentence is factually or semantically correct. Improving
that requires more varied text and training capacity, or a separately tracked
pretrained-model fine-tuning route.

## Independent expanded-data continuation

The expanded-data experiment is deliberately separate from `checkpoints/v14_bpe`.
It loads the old 512-token BPE state from the parent checkpoint, keeps the old
model dimensions and embedding rows unchanged, evaluates the new validation
split, and writes only to `checkpoints/v14_bpe_expanded_continue`. A 30% replay
stream from the original corpus is enabled to reduce catastrophic forgetting.

The current expanded corpus is the independently prepared Gutenberg
Shakespeare dataset at `data/expanded/dataset-expand.txt`. It was built from
the complete works source
`https://www.gutenberg.org/cache/epub/100/pg100.txt`, with play-only
extraction, format normalization, work/paragraph/n-gram deduplication, and a
work-level train/validation/test manifest. The original `dataset.txt` and the
historical v14 checkpoints are not modified.

To rebuild this exact corpus from the downloaded source:

```bash
python scripts/prepare_shakespeare_expanded.py \
  --baseline C:/Users/86151/Desktop/dataset.txt \
  --source work/download_temp/pg100.full.txt \
  --output C:/Users/86151/Desktop/dataset-expand.txt \
  --report C:/Users/86151/Desktop/format_report.json
```

Copy the resulting TXT and JSON to `data/expanded/` before training. Training
itself is offline. Then resume from the historical v14 `best.pt` (not the local
smoke-test checkpoint):

```bash
python scripts/train_v14.py \
  --config configs/v14_bpe_expanded_continue.yaml \
  --resume /path/to/old/v14_bpe/best.pt \
  --device cuda
```

The new checkpoints record the parent checkpoint, old/new data hashes and
sizes, token counts, `<unk>` ratios, replay ratio, and whether best selection
was reset for the new validation set. Use `scripts/generate_v14.py` with the
new experiment's `best.pt` for generation; the historical v14 checkpoints
remain available for fixed-prompt comparison.

## v4 comparison

| Result | Expanded validation loss | PPL | Old validation loss | Best step | Tokens seen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical v14 | 2.3789 | 10.79 | not measured | selected by validation | not recorded |
| Current v4 | 2.350902 | 10.495 | 1.921622 | 60,000 | 245,760,000 |

Under the same BPE family, v4 improves the expanded validation loss over the
historical v14 result by `0.027998` and reduces perplexity by about `2.7%`.
The generated samples still contain occasional repetition, half-lines, and
semantic jumps, so the metric improvement is not a claim of full sentence-level
understanding.

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

The v13 tokenizer maps every character to one id. The v14 tokenizer maps
characters and learned subword pieces to ids. Both keep the four reserved ids
stable: `<pad>=0`, `<bos>=1`, `<eos>=2`, and `<unk>=3`.

The encoded stream is split without shuffling: the first 90% is train and the
final 10% is validation. For each sampled offset `i`, the dataset returns:

```text
x = ids[i : i + T]       # [T]
y = ids[i + 1 : i + T + 1] # [T]
```

After batching, both are `[B, T]`. For v13 `T` counts characters; for v14 `T`
counts BPE tokens. The model looks up token embeddings and
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
| early stopping | disabled in `tiny.yaml`; enabled in v14 |

The implementation asserts that `n_embd % n_head == 0` and checks sequence
lengths before attention. `--debug-shapes` is not needed because
`inspect_batch.py` exposes the data boundary and the model tests cover the
remaining shapes; keeping forward passes free of print statements makes the
training log readable.

## Checkpoints and reproducibility

`best.pt` and `last.pt` contain model and optimizer state, model/training
configuration, tokenizer vocabulary, current step, best validation loss,
best step, tokens seen, early-stopping counters, and the seed. Loading uses
`map_location`, so a CPU machine can load a checkpoint created on CUDA.
Training seeds Python and PyTorch; training batches use a dedicated seeded
generator, while validation uses a fixed evenly spaced set of windows so
checkpoint comparisons are stable.

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

- The v13 tokenizer is character-level, so its sequences are longer than v14
  BPE or SentencePiece sequences.
- The v14 BPE tokenizer is intentionally compact and educational; it is not a
  drop-in replacement for a production tokenizer.
- Generation recomputes the entire context at every step and has no KV cache;
  this is intentionally straightforward and is not production optimized.
- The project is a teaching/reference implementation, not a production LM.
- Future extensions could add SentencePiece, Hugging Face `datasets`,
  `accelerate`, mixed precision, benchmark tooling, and PEFT/LoRA adapters.
