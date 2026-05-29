# Lab3 · 端到端语音识别 (CTC + Bi-LSTM on AN4)

> 复旦 AI 大课 · 人工智能编程基础 · 2026 春

## 实验目标

把一段英文录音 (字母拼写 / 数字 / 简单指令) 自动转写成文字。本实验你会:

1. 把音频波形变成 **Mel 频谱图** (Mel-Spectrogram), 理解声音如何变成神经网络能吃的张量。
2. 用 **双向 LSTM + CTC Loss** 训练一个字符级的语音识别模型。
3. 加 **SpecAugment + Speed Perturbation** 数据增强, 把测试 WER 从 ~47% 降到 ~36%。

---

## 文件清单

``` text
lab3_2026/
├── README.md                       本文件
├── lab3_CTC_ASR_blank.ipynb        ← 主要文件: 你做的就是这个 (6 个 TODO)
├── utils.py                        数据加载 / 增强 / CTC 解码 / WER&CER 评测
├── data/AN4/                       ⚠️ 数据集不在 zip 里, 自己放到这里
└── model/                          训完的 checkpoint 会存到这里 (notebook 会自动建)
```

**数据准备**: 把 AN4 数据集 (~60 MB) 放到 `lab3_2026/data/AN4/`, 目录里应该有 `etc/` 和 `wav/` 两个子目录, 然后再开始跑 notebook。

## 环境

notebook 顶部有一个 `!pip install -q jiwer torchcodec==0.10.0` 的安装 cell, 第一次跑请先执行它。
其他依赖 (`torch`, `torchaudio`, `matplotlib`) 算力平台一般都自带。

GPU 推荐 T4 / P100 / L20 / RTX 30/40 系列; CPU 也能跑但会非常慢。

## 训练时长 (实测, 仅供参考)

| 档 | 内容 | 训练样本数 | 4090 / epoch | 20 ep 总耗时 | ≈T4 | ≈P100 | ≈L20 |
| ---- | ------ | ----------- | -------------- | -------------- | ----- | ------- | ------ |
| **T0** | 纯 AN4, 无增强 | 948 | 4.4 s | ~90 s | ~6 min | ~4 min | ~2 min |
| **T2** | 3× speed perturb + SpecAugment | 2844 | 13 s | ~260 s | ~17 min | ~11 min | ~6.5 min |

Peak VRAM 520~560 MB, 任何一张学生卡都装得下。

## 作业清单 (6 个 TODO, 都在 `lab3_CTC_ASR_blank.ipynb`)

| # | 在哪一步 | 内容 | 对应 slide |
| --- | --------- | ------ | ----------- |
| 1 | Stage 2 (Part A) | `MelSpectrogram` 4 个参数 | p.49-53 |
| 2 | Stage 3 (Part B) | 在词表索引 0 处插入 CTC 的 `<blank>` token | p.60, p.63 |
| 3 | Stage 6 (Part B) | 根据课件 p.79 定义 `CTC_ASR_Model` 的层 | p.79 |
| 4 | Stage 7 (Part B) | 把 logits 转成 CTC Loss 要的形状 | p.65 |
| 5 | Stage 7 (Part B) | 训练循环 3 步 (清梯度 / 反向传播 / 优化器更新) | (lab1/2 强化) |
| 6 | Stage 7 (Part B) | 选合适的 loss 函数和 optimizer | (换 loss) |

Part A 1 个 TODO、Part B 5 个 TODO、Part C (数据增强) 全部给好, 直接 Run 看 WER 改善即可。

## 提交要求

1. 完整运行通的 `lab3_CTC_ASR_blank.ipynb` (含输出)
2. 训出来的 `./model/best.pth` (T0 档) 和 `./model/best_t2.pth` (T2 档)

## 参考

- 课件: 《人工智能编程基础 · RNN 和语音识别》
- AN4 dataset: [https://www.speech.cs.cmu.edu/databases/an4/](https://www.speech.cs.cmu.edu/databases/an4/)
- CTC 论文: Graves et al., "Connectionist Temporal Classification", ICML 2006
- SpecAugment: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition", Interspeech 2019
