# 上级交付报告：Mini Transformer v5

**项目结论：v5 是当前正式模型。**

## 1. 项目做什么

本项目训练一个小型英文因果 Transformer。输入一段 Shakespeare 风格文本，模型逐个预测下一个 BPE token，最后生成续写文本。

项目目标是：

- 验证从数据、tokenizer、attention、loss 到生成的完整训练流程；
- 在扩容 Shakespeare 语料上降低 validation/test loss；
- 保留每次实验的独立 checkpoint 和可复现证据；
- 用固定指标和固定 prompt 判断新版本是否真的超过旧版本。

它不是问答模型，不使用预训练权重，也不能保证完整语义理解。

## 2. 最终采用 v5 的原因

v5 从 v4 best.pt 继续训练，保留 512 BPE 和原模型结构。v5 的 expanded test loss/PPL 是当前所有同路线版本中最好，且固定生成比 v6 更稳定。

| 指标 | v5 |
| --- | ---: |
| Expanded validation loss / PPL | **2.347299 / 10.457** |
| Expanded test loss / PPL | **1.999644 / 7.386** |
| Old validation loss / PPL | **1.913685 / 6.778** |
| Best step | **64,000** |
| Tokens seen | **262.144M** |
| 参数量 | **3,422,208** |
| Kaggle 状态 | **COMPLETE** |
| best.pt SHA-256 | 256775abb523fea7d663908431272aba8aaaf0a43f335eb382d4020459f42b2a |

## 3. v5 如何训练

### 3.1 数据

扩容文件：

    C:\Users\86151\Desktop\dataset-expand.txt

记录值：

- UTF-8 bytes：5,085,615；
- characters：5,024,557；
- SHA-256：79edbbbd07a86f58cc14e1447c1242ed1e3f645b08bf996fd307cbcbe586d320；
- train/validation/test：30/3/5 部作品。

数据来自完整作品重建，经过格式归一化、作品/段落/n-gram 去重。原始 dataset.txt 未修改。

### 3.2 模型

- tokenizer：512 BPE；
- Transformer layers：4；
- attention heads：8；
- embedding width：256；
- block size：256 BPE tokens；
- dropout：0.1；
- 参数量：3,422,208。

### 3.3 续训配置

- parent：v4 best.pt；
- parent SHA-256：369d43e5428f8229fda84a5bcdaee9a20ad11d01ab1a6f64091aed31b9d4874b；
- learning rate：1e-5；
- batch size：16；
- old-data replay：15%；
- optimizer：AdamW，恢复 parent optimizer state；
- seed：1337；
- accelerator：Tesla P100-PCIE-16GB；
- PyTorch：2.5.1+cu124；
- 本地测试：21/21 passed。

训练每 1,000 step 评估一次 expanded validation，并保存：

- best.pt：验证指标最好的模型，正式使用；
- last.pt：训练结束状态，只用于恢复或过拟合诊断。

v5 在 step 64,000 达到最佳，step 74,000 因连续无改善触发 early stopping。后续 loss 回升，所以不能用 last.pt 替换 best.pt。

## 4. 版本比较

| 版本 | 训练方式 | Expanded val loss | Expanded test loss | Old val loss | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 历史 v1/v14 | 原始语料基线 | 不适用 | 未测 | 2.378887 | 历史保留 |
| v4 | 512 BPE 扩容基线 | 2.350902 | 2.232163 | 1.921622 | 不可变基线 |
| **v5** | **从 v4 续训** | **2.347299** | **1.999644** | 1.913685 | **正式模型** |
| v6 | 从 v4 独立续训 | 2.344260 | 2.003474 | **1.906484** | 独立对照 |
| v8 | 从 v5 继续训练 | 2.347299 | 1.999644 | 1.913685 | 未超过 v5 |
| v9 | 1024 BPE 从零训练 | 2.663134 | 2.452331 | 2.482651 | 负对照 |

v6 的 validation 虽略低，但 test 略高；v8 未产生更好的 best.pt；v9 只训练了 90.112M tokens，明显少于 v5 的 262.144M tokens。因此不能替代 v5。

## 5. 生成结果和限制

固定 prompt 的 v5 示例：

    Prompt: The king
    The king, my Lord of Somerset, and you
    Sent with him at your sister.

v5 可以学习到词边界、标点和舞台文本格式，但仍会出现角色混接、作品混合、半句和语义跳跃。sentence-boundary guard 只能截断明显不完整的尾部，不能当作语义能力。

## 6. Kaggle 交付状态

已用 Kaggle API 只读核查：

| Notebook | 状态 | 用途 |
| --- | --- | --- |
| v5 Notebook | COMPLETE | 正式训练结果 |
| v6 Notebook | COMPLETE | 独立对照 |
| v8 Notebook | COMPLETE | 失败续训对照 |
| v9 Notebook | COMPLETE | 1024 BPE 独立对照 |

v5 Notebook 日志确认 parent hash、数据 hash、21/21 测试通过、P100、best step 和 tokens seen。v5 本地 canonical checkpoint 与历史副本 hash 一致。

## 7. GitHub 交付状态

- 主仓库：Answwer/mini-transformer-charlm；
- 独立 v9 仓库：Answwer/mini-transformer-bpe1024-dev-v9；
- v5 主仓库包含 README、RESULTS 和本报告；
- v1/v14、v4、v5、v6、v8、v9 的代码记录和结果 JSON 按版本保留；
- v9 的独立结果不会覆盖 v5。

提交时优先打开主仓库 README 和本报告，再打开 v5 Kaggle Notebook 查看运行证据。v5 的正式模型文件不提交到 Git 大文件历史中，使用本地/Kaggle checkpoint 路径和 SHA-256 进行核验。

## 8. 交付判断

**推荐交付：v5。**

推荐上级查看顺序：

1. 本报告；
2. GitHub README；
3. GitHub RESULTS；
4. v5 Kaggle Notebook；
5. 需要复核时再查看 v4/v6/v8/v9 对照和历史 v1/v14。

后续如果继续优化，必须建立新的 Notebook、输出目录和 checkpoint；新版本只有在未使用 test 选择 checkpoint 的情况下稳定超过 v5，才可以替换正式结果。
