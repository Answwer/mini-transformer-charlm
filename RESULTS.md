# 最终实验结果：v5

## 结论

v5 是当前正式活动结果。它从 v4 best.pt 继续训练，使用旧的 512 BPE、扩容 Shakespeare 数据和 15% old-data replay。v5 在 expanded test 上优于 v4、v6、v8 和独立 1024 BPE v9。

## v5 正式指标

| 指标 | v5 |
| --- | ---: |
| Expanded validation loss / PPL / BPB | 2.347299 / 10.457 / 2.0009 |
| Development validation loss / PPL / BPB | 1.824066 / 6.197 / 1.5579 |
| Expanded test loss / PPL / BPB | 1.999644 / 7.386 / 1.6784 |
| Old validation loss / PPL | 1.913685 / 6.778 |
| Best step | 64,000 |
| Tokens seen | 262,144,000 |
| 参数量 | 3,422,208 |
| best.pt SHA-256 | 256775abb523fea7d663908431272aba8aaaf0a43f335eb382d4020459f42b2a |
| Kaggle 状态 | COMPLETE |

v5 Kaggle Notebook：[mini-transformer-bpe-lm-expanded-optimize-v5](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-optimize-v5)

## 版本对比

| 版本 | 训练关系 | Expanded val loss / PPL | Expanded test loss / PPL | Old val loss / PPL | Best step | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 历史 v1/v14 | 原始语料基线 | 不适用 | 未测 | 2.378887 / 10.793 | 历史最佳点 | 只读历史 |
| v4 | 扩容数据 512 BPE 基线 | 2.350902 / 10.495 | 2.232163 / 9.320 | 1.921622 / 6.832 | 60,000 | 不可变基线 |
| v5 | 从 v4 续训，replay=0.15 | **2.347299 / 10.457** | **1.999644 / 7.386** | 1.913685 / 6.778 | **64,000** | **正式结果** |
| v6 | 从 v4 独立续训，replay=0.10 | 2.344260 / 10.426 | 2.003474 / 7.415 | **1.906484 / 6.729** | 80,000 | 独立对照 |
| v8 | 从 v5 续训，replay=0.20 | 2.347299 / 10.457 | 1.999644 / 7.386 | 1.913685 / 6.778 | 64,000 | 未超过 v5 |
| v9 | 新 1024 BPE，从零训练 | 2.663134 / 14.341 | 2.452331 / 11.615 | 2.482651 / 11.973 | 22,000 | 负对照 |

## v5 训练配置

- tokenizer：512 BPE；
- 模型：4 层、8 头、256 维，block size=256；
- 参数量：3,422,208；
- batch size：16；
- learning rate：1e-5；
- old-data replay：0.15；
- optimizer：AdamW，恢复 v4 optimizer state；
- early stopping：expanded validation loss；
- GPU：Tesla P100-PCIE-16GB；
- PyTorch：2.5.1+cu124；
- 测试：21/21 passed。

父 checkpoint：

    v4 best.pt
    SHA-256: 369d43e5428f8229fda84a5bcdaee9a20ad11d01ab1a6f64091aed31b9d4874b

扩容数据：

    C:\Users\86151\Desktop\dataset-expand.txt
    SHA-256: 79edbbbd07a86f58cc14e1447c1242ed1e3f645b08bf996fd307cbcbe586d320

## 生成诊断

固定 5 prompts × 3 seeds，使用相同采样设置。raw/guarded 统计：

| 诊断 | v5 | v9 |
| --- | ---: | ---: |
| Sentence termination | 17/30 | 17/30 |
| Half-sentence tails | 12 | 13 |
| Repeated trigrams | 0 | 6 |
| Duplicate lines | 0 | 11 |
| Speaker-label switches | 14 | 16 |

v5 示例：

    Prompt: The king
    The king, my Lord of Somerset, and you
    Sent with him at your sister.

模型仍可能混合角色和作品；guard 只是后处理，不代表完整语义理解。

## 结果选择规则

v5 作为正式结果的原因：

1. expanded test loss/PPL 是当前最好；
2. validation 与 test 之间没有明显冲突；
3. v6 虽然 validation 略低，但 test 略高；
4. v8 没有产生优于 v5 的 checkpoint；
5. v9 的 1024 BPE 训练不足，所有主要 loss/PPL 均低于 v5。

所有历史 checkpoint、Notebook、结果 JSON 和输出目录均独立保存，后续实验不得覆盖 v5。

## 平台状态

主仓库只发布 v5 可复现代码；历史版本只保留结果 JSON 和本报告中的对比证据。历史 Notebook、checkpoint 和独立输出仍保存在本机/Kaggle，不会被删除。

- 主 GitHub 仓库：[Answwer/mini-transformer-charlm](https://github.com/Answwer/mini-transformer-charlm)
- 独立 v9 GitHub 仓库：[Answwer/mini-transformer-bpe1024-dev-v9](https://github.com/Answwer/mini-transformer-bpe1024-dev-v9)
- v9 PR：[PR #1](https://github.com/Answwer/mini-transformer-bpe1024-dev-v9/pull/1)，保留为独立对照记录；
- 上级交付入口应优先查看本仓库 README、SUPERVISOR_REPORT.md 和 v5 Kaggle Notebook。
