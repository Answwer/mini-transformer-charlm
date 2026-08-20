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

## Historical v14, v4/v5 baselines, and current v6

The original from-scratch v14 run is preserved as a read-only historical
reference. Its checkpoint, Kaggle notebook, and output bundle live under the
local `work/history_v14_*` directories and are never used as the active
training output.

The active expanded-data result is v6, selected by expanded validation loss
after two independent continuations from the immutable v4 checkpoint:

- Kaggle notebook: [Mini Transformer BPE LM Expanded Optimize v6](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-optimize-v6)
- Hardware: Tesla P100 GPU; PyTorch: `2.5.1+cu124`
- Current local suite: `21/21 passed`
- Best step: `80,000`; tokens seen: `327,680,000`
- Expanded validation loss: `2.344260`; perplexity: `10.426`
- Expanded test loss: `2.003474`; perplexity: `7.415`
- Historical v14 validation loss: `2.3789`; perplexity: `10.79`
- Old-corpus validation loss: `1.906484`; perplexity: `6.729`
- v6 best checkpoint SHA-256: `3e01a1df97b1498c1ce5899397f9aea2dc3ecd8ad6b6921e7d6e4d6b3b3033cd`

The v4 result remains an immutable comparison baseline in its own
`work/kaggle_expanded_optimize_output_v4` directory. The v5 and v6 results are
stored in the separate `work/kaggle_v5_output` and `work/kaggle_v6_output`
directories. Historical v14 artifacts remain available for fixed-prompt
comparison and are not overwritten.

The historical v14 validation loss rose after its best checkpoint, which is
the expected overfitting signal. Historical v14, v4, v5, and v6 generation load
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

## v4, v5, and v6 comparison

| Result | Expanded val loss / PPL | Expanded test loss / PPL | Old val loss / PPL | Best step | Tokens seen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical v14 | 2.3789 / 10.79 | not measured | not measured | selected by validation | not recorded |
| Baseline v4 | 2.350902 / 10.495 | 2.232163 / 9.320 | 1.921622 / 6.832 | 60,000 | 245,760,000 |
| Baseline v5 | 2.347299 / 10.457 | 1.999644 / 7.386 | 1.913685 / 6.778 | 64,000 | 262,144,000 |
| Current v6 | 2.344260 / 10.426 | 2.003474 / 7.415 | 1.906484 / 6.729 | 80,000 | 327,680,000 |

Under the same BPE family, v6 improves expanded validation loss over v4 by
`0.006643` and over v5 by `0.003039`; old-corpus validation is also lowest at
`1.906484`. v5 remains slightly better on expanded test PPL (`7.386` versus
v6's `7.415`). These are next-token metrics, not a claim of full sentence-level
understanding.
The generated samples still contain occasional repetition, half-lines, and
semantic jumps, so the metric improvement is not a claim of full sentence-level
understanding.

The fixed prompt examples use seed `7`, temperature `0.7`, top-k `40`, top-p
`0.9`, repetition penalty `1.05`, and the sentence-boundary guard. For example:

```text
Prompt: The king
v4: The king, my Lord of Somerset, and you / Sent with him at your sister.
v5: The king, my Lord of Somerset, and you / Sent with him at your sister.
v6: The king hath sent to me and Ill pay the way.

Prompt: To be, or not to be:
v4: ...And now the priest could not solicit me.
v5: ...And now the Kings incurable of his love, / And put his business to his evil tongue.
v6: ...And now the Kings incur, when the Duke of York.
```

The complete four-prompt strings are stored in `v4_result.json`,
`v5_result.json`, and `v6_result.json`.

The completed v5 and v6 isolated refinements both used the v4 checkpoint as
their parent and wrote to separate directories. Their Kaggle notebooks are:

- [v5 notebook](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-optimize-v5)
- [v6 notebook](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-optimize-v6)

```bash
python scripts/train_v14.py \
  --config configs/v5_bpe_expanded_optimize.yaml \
  --resume /path/to/v4/best.pt \
  --device cuda
```

The v5 configuration uses a `0.15` old-data replay ratio; v6 uses `0.10`. Both
lower the continuation learning rate to `1e-5`, keep the 512-token BPE
vocabulary and 4-layer/256-width model unchanged, and use separate
`checkpoints/v14_bpe_expanded_optimize_v5` and
`checkpoints/v14_bpe_expanded_optimize_v6` directories. v6 is active by the
expanded-validation criterion; v4 and v5 remain intact. No further run is
started after v6 until explicitly requested.

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
