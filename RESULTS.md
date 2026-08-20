# Historical v14 versus active v4

The historical v14 checkpoint and its Kaggle artifacts are preserved under
`work/history_v14_*`. The active result is v4, stored separately under
`work/kaggle_expanded_optimize_output_v4`; no checkpoint directory is shared.

## Fixed metrics

| Result | Evaluation corpus | Validation loss | Validation PPL | Test loss | Test PPL | Old validation loss / PPL | Best step | Tokens seen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical v14 | original corpus | 2.378887 | 10.7929 | not evaluated | not evaluated | 2.378887 / 10.7929 | selected by validation | not recorded |
| Active v4 | expanded corpus | 2.350902 | 10.495 | 2.23216 | 9.32 | 1.921622 / 6.832 | 60,000 | 245,760,000 |

The v4 expanded validation loss is `0.027985` lower than the historical v14
old-corpus validation loss. These losses are only directly comparable when the
tokenizer and evaluation split are the same; the table keeps the corpus labels
explicit to avoid treating a cross-corpus loss as a semantic score.

## Fixed-prompt generation

Both results use the same BPE model family, seed `7`, temperature `0.7`,
top-k `40`, top-p `0.9`, repetition penalty `1.05`, and a sentence-boundary
guard. The v4 samples are recorded in the v4 Kaggle output JSON. Across the
four recorded prompts, v4 has no obvious repeated 3-gram loop and no trailing
half-sentence in the saved samples; it still makes semantic jumps and can join
unrelated speakers. Historical v14 has one visibly truncated prompt sample and
more local continuation drift.

The metric and generation evidence support v4 as the active checkpoint, but do
not imply full sentence-level understanding. The next independent experiment is
v5: a lower-learning-rate continuation from the v4 `best.pt` into
`checkpoints/v14_bpe_expanded_optimize_v5`.
