# Historical v14, v4 baseline, and active v5

The historical v14 checkpoint and its Kaggle artifacts are preserved under
`work/history_v14_*`. v4 remains an immutable baseline under
`work/kaggle_expanded_optimize_output_v4`; the current active result is v5 in
the independent `work/kaggle_v5_output` directory. No checkpoint directory is
shared.

## Fixed metrics

| Result | Evaluation corpus | Validation loss | Validation PPL | Test loss | Test PPL | Old validation loss / PPL | Best step | Tokens seen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical v14 | original corpus | 2.378887 | 10.7929 | not evaluated | not evaluated | 2.378887 / 10.7929 | selected by validation | not recorded |
| Baseline v4 | expanded corpus | 2.350902 | 10.495 | 2.23216 | 9.32 | 1.921622 / 6.832 | 60,000 | 245,760,000 |
| Active v5 | expanded corpus | 2.347299 | 10.457 | 1.99964 | 7.386 | 1.913685 / 6.778 | 64,000 | 262,144,000 |

The v5 expanded validation loss is `0.003604` lower than v4 and `0.031588`
lower than historical v14. The v5 expanded test loss is `0.232518` lower than
v4. These are controlled comparisons within the same BPE family and expanded
split; they are not semantic scores. The older v4-versus-historical-v14 loss
comparison remains cross-corpus and is not treated as a controlled metric.

## Fixed-prompt generation

All results use the same BPE model family, seed `7`, temperature `0.7`, top-k
`40`, top-p `0.9`, repetition penalty `1.05`, and a sentence-boundary guard.
The v5 samples are recorded in `v5_result.json` and the v4 samples in
`v4_result.json`. Across the four prompts, v5 has no obvious repeated 3-gram
loop or trailing half-sentence; it still makes semantic jumps and can join
unrelated speakers. Compared with v4, v5 changes the continuation of two
prompts but does not establish full semantic understanding. Historical v14 has
one visibly truncated prompt sample and more local continuation drift.

The metric and generation evidence support v5 as the active checkpoint, while
v4 remains the reproducible comparison baseline. The independent v6 fallback
continues from v4 and may replace v5 only if it beats `2.347299` on expanded
validation.
