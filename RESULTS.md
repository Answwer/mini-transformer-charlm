# Historical v14, v4/v5 baselines, and active v6

The historical v14 checkpoint and its Kaggle artifacts are preserved under
`work/history_v14_*`. v4 remains an immutable baseline under
`work/kaggle_expanded_optimize_output_v4`; v5 is preserved in
`work/kaggle_v5_output`; the current active result is v6 in
`work/kaggle_v6_output`. No checkpoint directory is shared.

## Fixed metrics

| Result | Evaluation corpus | Validation loss | Validation PPL | Test loss | Test PPL | Old validation loss / PPL | Best step | Tokens seen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical v14 | original corpus | 2.378887 | 10.7929 | not evaluated | not evaluated | 2.378887 / 10.7929 | selected by validation | not recorded |
| Baseline v4 | expanded corpus | 2.350902 | 10.495 | 2.23216 | 9.32 | 1.921622 / 6.832 | 60,000 | 245,760,000 |
| Active v5 | expanded corpus | 2.347299 | 10.457 | 1.99964 | 7.386 | 1.913685 / 6.778 | 64,000 | 262,144,000 |
| Active v6 | expanded corpus | 2.344260 | 10.426 | 2.00347 | 7.415 | 1.906484 / 6.729 | 80,000 | 327,680,000 |

The v6 expanded validation loss is the best of the three active-family runs:
`0.006643` lower than v4 and `0.003039` lower than v5. Its old-corpus
validation loss is also lowest (`1.906484`). v5 has the lowest expanded test
loss/PPL (`1.999644` / `7.386`) while v6 is slightly higher (`2.003474` /
`7.415`), so test performance is not uniformly improved. These are controlled
comparisons within the same BPE family and expanded split; they are not
semantic scores. The older v4-versus-historical-v14 loss comparison remains
cross-corpus and is not treated as a controlled metric.

## Fixed-prompt generation

All results use the same BPE model family, seed `7`, temperature `0.7`, top-k
`40`, top-p `0.9`, repetition penalty `1.05`, and a sentence-boundary guard.
The v4, v5, and v6 samples are recorded in `v4_result.json`, `v5_result.json`,
and `v6_result.json`. Across the four prompts, all three avoid an obvious
repeated 3-gram loop. v4 has a sentence-boundary-safe but very short `The king`
continuation; v5 produces a complete-looking `To be` continuation but still
joins unrelated speakers; v6 has the shortest `The king` sample with trailing
blank lines and a compressed `To be` continuation. All three still make
semantic jumps and none establishes full semantic understanding. Historical v14
has one visibly truncated prompt sample and more local continuation drift.

The primary selection criterion is expanded validation loss, so v6 is now the
active checkpoint. v4 and v5 remain reproducible comparison baselines with
independent checkpoints, notebooks, and output directories. Per the requested
pause condition, no further optimization run is being started after v6.
