# Dataset

This repository uses the public-domain tiny Shakespeare corpus from
[Karpathy's char-rnn repository](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

- Dataset name: tiny Shakespeare
- Source URL: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Repository preparation date: 2026-08-19
- File: `data/tiny_shakespeare.txt`
- Encoding: UTF-8, LF line endings
- File size: 1,115,393 bytes
- SHA-256: `53493bf304b639aba6a47400da75c58ce32372b378fd2cd9d16ee987aeb12e4e`

The file was supplied as a local copy of the source corpus and normalized
from CRLF to LF line endings. No text was added, removed, shuffled, or
lower-cased. The training code reads this file locally and never downloads
data at training, evaluation, or generation time.

The dataset is split by character position: the first 90% is training data
and the final 10% is validation data. The split is contiguous and is made
before any random windows are sampled, so validation windows do not leak
across the split boundary.
