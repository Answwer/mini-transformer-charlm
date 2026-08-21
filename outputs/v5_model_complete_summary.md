# v5 模型与实验完整记录

更新时间：2026-08-21

本文档记录 v5 的来源、训练配置、数据与 tokenizer、checkpoint 完整性、评估指标、固定生成样例、v7 诊断结果、与 v4/v6 的对比，以及后续优化约束。v5 的历史文件必须独立保留；本文档不替换、不移动、不覆盖任何旧实验文件。

## 1. 结论摘要

v5 是当前 v14 BPE 扩容路线中用于综合比较的活动结果。它从 v4 的 `best.pt` 续训，保持旧 512 BPE tokenizer、4 层 Transformer、扩容数据切分和模型参数形状不变；训练时混入 15% 原始数据 replay。

v5 的最佳 checkpoint 是训练 step 64,000：

- expanded validation loss：`2.3472988367`
- expanded validation PPL：`10.4572847`
- expanded test loss：`1.9996444702`
- expanded test PPL：`7.3864295`
- old validation loss：`1.9136853456`
- old validation PPL：`6.7780222`
- tokens seen：`262,144,000`
- 参数量：`3,422,208`

v5 相比 v4 的主要收益是 expanded test loss 从 `2.23216` 降至 `1.99964`，同时 expanded validation loss 从 `2.35090` 降至 `2.34730`。v6 的 expanded validation 和 old validation 略低，但 expanded test 略高，且 v7 固定多 seed 诊断显示 v5 的 raw generation 稳定性优于 v6。因此 v5 作为综合活动结果，v6 作为独立对照结果保留。

## 2. 版本关系

```text
历史 v1 / 历史 v14
        |
      v4  (从 v4 父模型继续优化，512 BPE)
        |
      v5  (从 v4 best.pt 继续优化，old-data replay=0.15)

v6  (独立从同一个 v4 best.pt 继续优化，old-data replay=0.10)
v7  (独立诊断 Notebook，只读取 v5/v6，不训练、不产生新模型)
```

v5 不是随机初始化，也不是重新训练 tokenizer。v5 从 v4 的 checkpoint 恢复了 Transformer 权重和 optimizer state；其 tokenizer 直接从父 checkpoint 恢复，因此可以称为同 tokenizer 的续训。

## 3. Kaggle 与本地位置

| 项目 | 位置 |
| --- | --- |
| v5 Kaggle Notebook | [mini-transformer-bpe-lm-expanded-optimize-v5](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-optimize-v5) |
| v5 本地输出 | `work/kaggle_v5_output/mini-transformer-charlm` |
| v5 原始 best checkpoint | `work/kaggle_v5_output/mini-transformer-charlm/checkpoints/v14_bpe_expanded_optimize_v5/best.pt` |
| v5 原始 last checkpoint | `work/kaggle_v5_output/mini-transformer-charlm/checkpoints/v14_bpe_expanded_optimize_v5/last.pt` |
| v5 训练 Notebook 源码 | `work/kaggle_expanded_optimize_v5/` |
| v7 诊断 Notebook | [mini-transformer-bpe-lm-expanded-diagnose-v7](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-diagnose-v7) |
| v7 诊断 JSON | `work/kaggle_v7_diagnostic_output/mini-transformer-charlm/v7_diagnostic_result.json` |

## 4. v5 checkpoint 保留与完整性

为了防止后续优化误覆盖，v5 已复制到独立历史目录：

`work/history_v5_preserved/`

该目录包含：

- `checkpoints/v14_bpe_expanded_optimize_v5/best.pt`
- `checkpoints/v14_bpe_expanded_optimize_v5/last.pt`
- `v5_expanded_optimize_result.json`
- `notebook/run_project.py`
- `notebook/kernel-metadata.json`
- `README.txt`

当前文件哈希与 checkpoint 状态：

| 文件 | step | validation 状态 | SHA-256 |
| --- | ---: | --- | --- |
| `best.pt` | 64,000 | 最佳 expanded validation，loss `2.3472988367` | `256775abb523fea7d663908431272aba8aaaf0a43f335eb382d4020459f42b2a` |
| `last.pt` | 74,000 | 末步，expanded validation 已回升到 `2.3494526863` | `b59ca3a20923391979b702cf44c9a4d56162f9c8593524d814ffcf6ba41c6c93` |

注意：旧的 `v5_result.json` 曾把 `b59ca...` 写为 `best_checkpoint_sha256`，但逐文件复核表明该哈希对应 `last.pt`，不是 `best.pt`。后续生成和评估必须使用 `best.pt`；`last.pt` 只用于明确的过拟合对照。

## 5. 训练配置

配置文件：`configs/v5_bpe_expanded_optimize.yaml`

### 模型

| 设置 | 值 |
| --- | ---: |
| tokenizer | BPE |
| vocabulary | 512 |
| block size | 256 BPE tokens |
| layers | 4 |
| attention heads | 8 |
| embedding size | 256 |
| dropout | 0.1 |
| parameters | 3,422,208 |

### 续训

| 设置 | 值 |
| --- | ---: |
| parent checkpoint | v4 `best.pt` |
| parent SHA-256 | `369d43e5428f8229fda84a5bcdaee9a20ad11d01ab1a6f64091aed31b9d4874b` |
| learning rate | `0.00001` |
| batch size | 16 |
| old-data replay ratio | 0.15 |
| max steps | 90,000（实际 best step 64,000，last step 74,000） |
| eval interval | 1,000 steps |
| eval steps | 10 |
| early stopping patience | 10 次评估 |
| early stopping min delta | `0.0002` |
| seed | 1337 |
| device | Tesla P100-PCIE-16GB |
| optimizer | AdamW，恢复父 checkpoint optimizer state |

训练选择依据是 expanded validation loss。训练过程同时记录 old validation loss，但不使用 test 选择 checkpoint。v5 的最佳点在 64,000 step；继续到 74,000 step 后 validation 变差，所以不得用 `last.pt` 代替 `best.pt`。

## 6. 数据与切分

### 扩容数据

- 文件：`C:\Users\86151\Desktop\dataset-expand.txt`
- Kaggle 记录的 SHA-256：`79edbbbd07a86f58cc14e1447c1242ed1e3f645b08bf996fd307cbcbe586d320`
- UTF-8 字节数：`5,085,615`
- 字符数：`5,024,557`
- 512 BPE token 数：`2,991,504`
- `<unk>` 数量：`34,728`
- `<unk>` 比例：`1.1608876%`

扩容文本由完整莎士比亚作品重建、格式归一化、作品/段落/n-gram 去重得到。原始文件 `C:\Users\86151\Desktop\dataset.txt` 不得修改。

### split

官方 `format_report.json` 的作品级切分保持不变：

- train：30 部作品
- validation：3 部作品
- test：5 部作品

训练只使用 train 作品学习 BPE 和更新模型；validation 用于 checkpoint 选择；test 只在实验配置确定后评估一次。v5 额外用原始数据作为 replay，比例为 15%，用于降低扩容训练对旧分布的遗忘。

### 原始 replay 数据

- 文件：`data/tiny_shakespeare.txt`（对应用户原始 `dataset.txt` 的仓库副本）
- 字节数/字符数：`1,115,393`
- 512 BPE token 数：`649,564`
- `<unk>` 比例：`0`

## 7. v4、v5、v6 固定指标对比

| 结果 | expanded validation loss | expanded validation PPL | expanded test loss | expanded test PPL | old validation loss | old validation PPL | best step | tokens seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 | 2.350902 | 10.4950 | 2.232163 | 9.320 | 1.921622 | 6.832 | 60,000 | 245,760,000 |
| **v5** | **2.347299** | **10.4573** | **1.999644** | **7.386** | 1.913685 | 6.778 | 64,000 | 262,144,000 |
| v6 | 2.344260 | 10.4256 | 2.003474 | 7.415 | **1.906484** | **6.729** | 80,000 | 327,680,000 |

相对 v4，v5 的变化为：

- expanded validation loss：`-0.003604`
- expanded test loss：`-0.232518`
- old validation loss：`-0.007936`

相对 v6，v5 的取舍为：

- v5 expanded validation 高 `0.003039`
- v5 expanded test 低 `0.003829`
- v5 old validation 高 `0.007201`
- v5 的 raw generation 在 v7 的 15 个固定样例中更稳定

这些是 token-level loss/PPL，不是语义理解分数。validation 只有 3 部作品，不能只凭 `0.003` 级别的 validation 差异断言模型整体更好；test 与固定生成必须一起看。

## 8. v5 固定 prompt 生成样例

采样设置：seed `7`、temperature `0.7`、top-k `40`、top-p `0.9`、repetition penalty `1.05`、min new tokens `24`、max new tokens `120`，默认启用句末边界 guard。

### `The king`

```text
The king, my Lord of Somerset, and you
Sent with him at your sister.
```

### `Before we proceed`

```text
Before we proceed,
Are we to sleep.

Exit Falstaff and Soldiers.
```

### `To be, or not to be:`

```text
To be, or not to be:
And now the Kings incurable of his love,
And put his business to his evil tongue.
```

### `Friends, Romans, countrymen`

```text
Friends, Romans, countrymen, Albany, Sir Toby,
Art the English fatal foe.
```

这些文本具备 Shakespeare 风格的词汇、标点和舞台格式，但仍存在作品/角色混接、语义跳跃和局部不自然搭配。它们不能证明模型具备完整语义理解。

## 9. v7 对 v5 的固定多 seed 诊断

v7 使用 5 个 prompt × 3 个 seed（`7, 17, 29`），同时保存 raw generation 和 sentence-boundary guarded generation：

- prompt：`The king`
- prompt：`Before we proceed`
- prompt：`To be, or not to be:`
- prompt：`Friends, Romans, countrymen`
- prompt：`All the world's a stage`

raw generation 的汇总：

| 指标（15 个样例） | v5 | v6 |
| --- | ---: | ---: |
| 半句尾部样例数 | 12/15 | 15/15 |
| 重复 trigram 总数 | 0 | 1 |
| 重复行总数 | 0 | 3 |
| speaker label 总数 | 27 | 26 |
| speaker label switch 总数 | 14 | 9 |
| guarded 删除字符总数 | 1,916 | 2,020 |

解释：v5 的 raw generation 比 v6 少半句、无重复 trigram、无重复行，因此生成稳定性较好；但 12/15 仍以半句结束，说明模型自身仍不能稳定生成完整句子。guarded 结果全部变得句末完整，是后处理截断的结果，不能把删除后的文本当成模型语义能力。

v7 的绝对 loss 使用 `eval_steps=100`，与 v5/v6 历史报告使用的 `eval_steps=10` 不同。因此 v7 只用于诊断排序和生成行为，不混用其绝对 loss 数值替代正式报告。

## 10. 已知问题与后续优化方向

1. **512 BPE 的 `<unk>` 比例偏高**：扩容语料约 `1.1609%` token 为 `<unk>`，新作品、人名和罕见拼写覆盖不足。1024 BPE 是合理的独立实验方向，但 tokenizer id 会变化，必须从头初始化，不能伪装成 v5 无损续训。
2. **固定生成仍有半句和角色混接**：降低 temperature 或依赖句末 guard 只能改变输出表面，不能替代训练质量。要实质改善，需要更高质量、格式一致的训练数据或另行采用预训练模型微调路线。
3. **validation/test 排序存在波动**：v6 validation 更低而 test 略高，说明 3 部 validation 作品不足以稳定代表泛化。后续应在不接触官方 test 的前提下建立 development-validation，并用组合验证规则选择 checkpoint。
4. **v5 已出现继续训练收益变小的迹象**：step 64,000 是 best，step 74,000 的 expanded validation 回升；后续续训必须使用独立目录、低学习率、较高 old-data replay，并同时检查 expanded validation、old validation、development-validation 和固定生成。

## 11. 保留规则

- v1/历史 v14、v4、v5、v6 的 Notebook、checkpoint 和输出目录全部保留。
- v5 原始目录 `work/kaggle_v5_output` 不删除、不覆盖。
- v5 备份目录 `work/history_v5_preserved` 不删除、不覆盖。
- 任何 v8 或后续实验必须使用新的 Kaggle Notebook、新的输出目录和新的 checkpoint 目录。
- 训练和评估默认使用 `best.pt`；`last.pt` 只能用于明确的 overfitting comparison。
- 原始 `C:\Users\86151\Desktop\dataset.txt` 不修改；扩容训练使用 `C:\Users\86151\Desktop\dataset-expand.txt`。

## 12. 关联文件

- v5 指标：`v5_result.json`
- v4 指标：`v4_result.json`
- v6 指标：`v6_result.json`
- v7 完整诊断：`work/kaggle_v7_diagnostic_output/mini-transformer-charlm/v7_diagnostic_result.json`
- v5 配置：`configs/v5_bpe_expanded_optimize.yaml`
- 评估脚本：`scripts/evaluate_v14.py`
- 训练脚本：`scripts/train_v14.py`
- 规划文件：`C:\Users\86151\Documents\Codex\2026-08-19\python-torch-transformers-datasets-accelerate-peft-5\outputs\transformer_repo_plan_for_luna_xhigh.md`

## 13. 基于 v7 诊断的 v8 续训结果

为验证“提高 old-data replay、降低学习率”能否修复 v7 暴露的半句和遗忘问题，已创建并完成独立 Notebook：[mini-transformer-bpe-lm-expanded-optimize-v8](https://www.kaggle.com/code/answerr5/mini-transformer-bpe-lm-expanded-optimize-v8)。v8 的父模型是本记录中的 v5 `best.pt`，不是 v4，也不是随机初始化。

v8 配置：旧 512 BPE、相同 4×256 Transformer、old-data replay `0.20`、学习率 `5e-6`、独立输出 `work/kaggle_v8_output`、独立 checkpoint 目录 `v14_bpe_expanded_optimize_v8`。父 checkpoint SHA-256 为 `256775abb523fea7d663908431272aba8aaaf0a43f335eb382d4020459f42b2a`。

结果：

- v8 在 step 72,000 触发 early stopping，未找到低于 v5 `2.3472988367` 的 expanded validation loss。
- v8 `best.pt` 与 v5 父模型指标完全相同：expanded validation `2.3472988367`、expanded test `1.9996444702`、old validation `1.9136853456`。
- v8 `best.pt` SHA-256：`a16c6aedb98095bfd1b02e70d728da5499922ddfadbe9af71ca1e1eaf42b4a01`。
- v8 `last.pt` SHA-256：`fde1c4d492a35c3c95a2e66d41591c860fa5ec098e7710207f70930bb5c0104a`；它在 step 72,000 的 expanded validation loss 为 `2.3504534245`，不应作为正式结果。
- v8 的 5 个固定 prompt（seed 7）输出与 v5 完全一致，因为最佳模型没有离开父 checkpoint。

因此 v8 是一次有明确结论的失败对照：仅靠更高 replay 和更低学习率，无法实质修复 v7 发现的半句、角色混接和语义跳跃。v5 仍是综合活动结果；后续若要继续，应改变可验证的容量/词表或数据划分方案，而不是盲目延长 512 BPE 续训。
