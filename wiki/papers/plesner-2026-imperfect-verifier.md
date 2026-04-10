---
title: "An Imperfect Verifier is Good Enough: Learning with Noisy Rewards"
type: paper
tags: [RLVR, noisy-verification, robustness, verifier-accuracy, noise-tolerance, practical-guidelines]
created: 2026-04-10
updated: 2026-04-10
sources: ["https://arxiv.org/abs/2604.07666"]
status: active
---

# An Imperfect Verifier is Good Enough: Learning with Noisy Rewards

## 一句话总结

系统性地证明 RLVR 对 verifier 噪声具有鲁棒性：**15% 噪声率**下峰值性能仅降 2pp，且**适度准确率 + 高精确率**比追求完美验证更重要。

## 基本信息

- **作者**: Andreas Plesner, Francisco Guzmán, Anish Athalye
- **机构**: Handshake AI, ETH Zurich
- **发表日期**: 2026-04-09
- **arXiv**: 2604.07666

## 摘要

RLVR 已成为 LLM 后训练的主流方法，但 verifier 很少是无错的。本文在代码生成和科学推理领域系统研究 verifier 噪声的影响。结果显示：噪声率高达 15% 时，峰值验证准确率仍在 clean baseline 的 2pp 以内。这些发现在受控噪声和模型噪声、三个模型家族（Qwen3、GLM4、Llama 3.1）以及 4B-9B 模型规模上均一致。结论：不完美的验证不构成 RLVR 的根本障碍，实践中应优先追求**适度准确率 + 高精确率**。

## 核心贡献

1. **噪声鲁棒性量化**: 首次系统研究不同噪声类型和噪声率对 RLVR 训练的影响
2. **精确率 > 召回率**: 发现 false positive 比 false negative 更有害，提出 "moderate accuracy + high precision" 原则
3. **噪声作为正则化**: 发现低至中等噪声可作为隐式正则化，防止过拟合
4. **跨域验证**: 从代码生成（MBPP）推广到科学推理（GPQA）

## 方法

### 受控噪声注入

在代码生成场景下，每个 prompt 有 $T$ 个 unit tests，每步生成 $G$ 个 rollout，奖励矩阵为 $G \times T$ 的二值矩阵 $\mathbf{M}$。噪声通过以概率 $p$ 翻转矩阵项注入。

**四种噪声模式**:

| 模式 | 粒度 | 描述 |
|------|------|------|
| Sample × Unit test | 最细 | 每个 cell 独立翻转 |
| Sample × Rollout | 行级 | 整行（单个 rollout 所有测试）翻转 |
| Group × Unit test | 列级 | 整列（所有 rollout 的同一测试）翻转 |
| Group × Rollout | 最粗 | 整个矩阵翻转或不翻转 |

### Model-based Verifier

用不同能力的模型（Qwen3 4B 和 Qwen3 30B-A3B）替代 unit test 执行器，预测每个 test case 是否通过，模拟真实场景中 LLM-as-Judge 的噪声分布。

### GRPO 训练

标准 GRPO 优势估计：

$$\hat{A}_i = \frac{r_i - \mu(\mathbf{r})}{\sigma(\mathbf{r}) + \epsilon}$$

## 实验结果

### 核心发现 1: 15% 噪声是安全阈值

| 噪声率 | 峰值 vs Clean | 状态 |
|--------|--------------|------|
| ≤ 15% | 2pp 以内 | 安全 |
| 20-30% | 轻度退化 | 可接受 |
| 40%+ | 严重退化 | 不可用 |
| 50% | 接近随机 | 失败 |

### 核心发现 2: 噪声类型影响不大

在 $p=0.10$ 噪声率下（MBPP, GLM4 9B）：

| 噪声类型 | Best 验证奖励 |
|----------|-------------|
| Clean Baseline | 0.905 ± 0.002 |
| Group Rollout | 0.900 ± 0.005 |
| Group Unit Test | 0.891 |
| Sample Rollout | 0.866 |
| Unit Test | 0.875 |

Group-level 噪声略优于 sample-level 噪声。

### 核心发现 3: 精确率 > 召回率

Model-based verifier 实验（Qwen3 8B 训练模型, MBPP）：

| Verifier | 验证奖励 | 精确率 | 召回率 |
|----------|---------|--------|--------|
| Qwen3 30B-A3B | 0.871 | 高 | >90% |
| Qwen3 4B | 0.704 | 低 | >90%（更高） |
| Clean baseline | 0.901 | 100% | 100% |

两个 verifier 召回率都 >90%，但 4B verifier 精确率显著更低 → false positive 让模型学会利用错误解（reward hacking），而 false negative 反而迫使模型多样化探索。

### 核心发现 4: 噪声可作为正则化

- Qwen3 8B 在 $p \leq 0.20$ 时，最终 checkpoint 性能略优于无噪声 baseline
- 假说：无噪声时模型过拟合训练数据，轻度噪声作为正则化防止过拟合
- Ackley 函数实验验证：中等噪声（$\sigma=2.0$）帮助 GRPO 逃离局部最优

### 跨域验证（GPQA, Qwen3 8B）

| 设置 | 最佳验证奖励 |
|------|-------------|
| Base model | 0.540 |
| No noise | 0.600 |
| Noise $p=0.05$ | 0.604 |
| Noise $p=0.30$ | 0.603 |

即使 30% 噪声率也不影响 GPQA 上的训练效果。

### 模型规模影响

Qwen3 4B 和 8B 在 $p \leq 0.30$ 时都优雅退化。$p=0.40$ 时 8B 退化更显著（损失 0.13 vs 4B 的 0.065）。

## 与 SPC/URLVR 研究的关系

### 对 Co-Evolving Verifier 的核心启示

1. **噪声容忍度的理论保障**: 本文证明 RLVR 训练对 verifier 噪声有内在鲁棒性（≤15% 几乎无影响）。这直接支持了 Co-Evolving Verifier 的可行性——即使早期阶段 verifier 不够准确，只要噪声率控制在合理范围内，训练仍然有效
2. **精确率优先的设计原则**: Co-Evolving Verifier 的校准策略应优先保证精确率（减少 false positive），即使牺牲部分召回率。因为 false positive 导致 reward hacking，而 false negative 反而促进探索
3. **渐进式 verifier 激活**: 结合本文的正则化效应，Co-Evolving Verifier 可以在训练早期引入较大噪声（作为探索正则化），后期逐渐提升精度
4. **URLVR 中的 verifier 设计**: 对于无监督场景，模型自生成的 verifier 信号天然带噪声，本文表明这不一定是致命问题

### 对 SPC 的影响

- SPC 的 probing-based consistency 信号也不可能完全准确，但只要精确率足够高（噪声率 <15%），就足以提供有效的 step-level 信用分配
- **关键约束**: SPC 信号中应尽量减少 false positive（将错误步骤标记为正确），这比减少 false negative 更重要

## 面试 Q&A

### Q1: 为什么 RLVR 对 verifier 噪声具有鲁棒性？

**A**: 主要有三个原因：(1) GRPO 的 group-relative 归一化机制使得即使奖励被噪声扰动，组内相对排序往往得到保留；(2) Group-level 噪声在触发时完全反转梯度方向，类似于 SAM (Sharpness-Aware Minimization) 中的对抗步，帮助策略逃离尖锐局部最优、偏向平坦最小值；(3) 训练过程中每个 epoch 的噪声独立采样，同一数据点不会被持续错误标注，从而起到类似 dropout 的正则化效果。

### Q2: 为什么 false positive 比 false negative 对 RLVR 更有害？

**A**: 这源于强化学习中的 exploration-exploitation 权衡。False positive（错误代码被标记为正确）让模型学会利用（exploit）这些错误方案，导致 reward hacking——模型在训练 reward 上表现良好但泛化变差。从实验中可以看到 4B verifier 的训练 reward 远高于 30B verifier，这正是过度利用 false positive 的表现。相反，false negative（正确代码被标记为错误）迫使模型尝试更多不同的解法（explore），而多数问题有多种正确解法，所以这种被迫探索并不有害。

### Q3: 这篇工作对 LLM-as-Judge 的 RLVR 实践有什么指导？

**A**: 三点核心建议：(1) 追求 ~85% 的 verifier 准确率和高精确率即可，不必追求完美验证；(2) 构建 verifier 时应优先优化精确率（减少 false positive），即使牺牲部分召回率；(3) 在从 deterministic domain（代码、数学）扩展到 semi-verifiable domain（金融、法律）时，可以接受 LLM-as-Judge 的噪声，只要控制在 15% 以内。
