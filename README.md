# Mini Transformer：v5 正式训练结果

这是一个从零实现的、可复现的小型 Shakespeare 风格英文因果语言模型。模型逐个预测下一个 BPE token，再根据预测结果生成续写文本。它是教学和实验项目，不是问答模型，也不使用预训练权重。

## 一眼结论

**v5 是当前正式活动结果。** 它从 v4 的 best.pt 继续训练，保持 512 BPE、4 层 Transformer 和原模型结构不变；在扩容测试集上取得当前最佳结果。v1/v14、v4、v6、v8 和 v9 均作为独立历史或对照保留。

| 指标 | v5 结果 |
| --- | ---: |
| Expanded validation loss / PPL | **2.347299 / 10.457** |
| Expanded test loss / PPL | **1.999644 / 7.386** |
| Old validation loss / PPL | **1.913685 / 6.778** |
| Best step | **64,000** |
| Tokens seen | **262.144M** |
| 参数量 | **3,422,208** |
| Kaggle 状态 | **COMPLETE** |
| v5 best.pt SHA-256 | 256775abb523fea7d663908431272aba8aaaf0a43f335eb382d4020459f42b2a |

## 直接查看

- v5 Kaggle Notebook：[Mini Transformer BPE LM Expanded Optimize v5](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-optimize-v5)
- v5 训练代码仓库：[mini-transformer-charlm](https://github.com/Answwer/mini-transformer-charlm)
- 可交付报告：[SUPERVISOR_REPORT.md](SUPERVISOR_REPORT.md)
- 完整指标：[RESULTS.md](RESULTS.md)
- 独立 v9 对照仓库：[mini-transformer-bpe1024-dev-v9](https://github.com/Answwer/mini-transformer-bpe1024-dev-v9)

## v5 做了什么

v5 使用扩容后的完整 Shakespeare 作品数据，并从 v4 best.pt 继续训练：

- tokenizer：512 BPE；
- 模型：4 层、8 个 attention heads、256 维 embedding、256 token 上下文；
- 参数量：3,422,208；
- batch size：16；
- continuation learning rate：1e-5；
- 原始数据 replay：15%，用于降低遗忘；
- optimizer：AdamW，恢复 v4 checkpoint 的 optimizer state；
- early stopping：以 expanded validation loss 选择 best.pt。

v5 不是随机初始化，也不是重新学习 tokenizer；它是同一 512 BPE 体系下从 v4 继续训练的独立实验。

## 数据

扩容数据文件为 C:\Users\86151\Desktop\dataset-expand.txt，由完整作品重建、格式归一化、作品/段落/n-gram 去重后得到。记录值：

- UTF-8 bytes：5,085,615；
- characters：5,024,557；
- SHA-256：79edbbbd07a86f58cc14e1447c1242ed1e3f645b08bf996fd307cbcbe586d320；
- official train/validation/test：30/3/5 部作品。

原始 dataset.txt 从未被修改。训练、验证和测试按作品切分，official test 只在最终评估使用。

## 版本对比

| 版本 | 说明 | Expanded val loss | Expanded test loss | Old val loss | Best step |
| --- | --- | ---: | ---: | ---: | ---: |
| 历史 v1/v14 | 原始语料历史基线 | 不适用 | 未测 | 2.378887 | 历史最佳点 |
| v4 | 扩容数据 512 BPE 基线 | 2.350902 | 2.232163 | 1.921622 | 60,000 |
| **v5** | **从 v4 续训，replay=0.15** | **2.347299** | **1.999644** | 1.913685 | **64,000** |
| v6 | 从 v4 独立续训，replay=0.10 | 2.344260 | 2.003474 | **1.906484** | 80,000 |
| v8 | 从 v5 续训，未超过 v5 | 2.347299 | 1.999644 | 1.913685 | 64,000 |
| v9 | 独立 1024 BPE，从零训练 | 2.663134 | 2.452331 | 2.482651 | 22,000 |

v6 的 validation 略低，但 test 略高于 v5；v8 没有产生优于 v5 的 checkpoint；v9 训练 token 数明显不足。综合 validation、test 和固定 prompt 稳定性，v5 作为正式结果。

## 生成能力边界

固定 prompt 示例：

    Prompt: The king
    v5: The king, my Lord of Somerset, and you
        Sent with him at your sister.

模型可以生成较合理的词边界、标点和舞台文本格式，但仍可能混合角色、作品和场景。sentence-boundary guard 只能减少截断，不能证明完整语义理解。

## 复现 v5

环境要求：Python 3.10+、PyTorch 2.x。项目不在运行时下载数据。

    python -m unittest discover -s tests -t . -v
    python scripts/train_v14.py --config configs/v5_bpe_expanded_optimize.yaml --resume C:\path\to\v4\best.pt --device cuda

正式评估必须使用 best.pt，不要用训练结束时的 last.pt 代替。v5 的本地输出目录为 work/kaggle_v5_output，历史副本为 work/history_v5_preserved。

## 历史保留规则

所有版本的 Notebook、checkpoint、结果 JSON 和输出目录均独立保存。任何后续 v10 或其他实验都必须使用新的 Notebook、新的输出目录和新的 checkpoint，不得覆盖 v5。
