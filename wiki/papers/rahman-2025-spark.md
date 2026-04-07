---
title: "Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning"
type: paper
tags: [URLVR, PRM, process-reward, self-consistency, GRPO, reward-hacking]
created: 2026-04-07
updated: 2026-04-07
sources: [https://arxiv.org/abs/2512.03244]
status: active
---

# SPARK: Stepwise Process-Aware Rewards for Reference-Free RL

## 基本信息
- **作者**: Mahmudur Rahman et al.
- **机构**: Amazon AGI & UCLA
- **年份**: 2025
- **会议/期刊**: arXiv preprint (arXiv:2512.03244)
- **链接**: https://arxiv.org/abs/2512.03244

## 一句话总结
> 三阶段框架：用 inference-time scaling（self-consistency + meta-critique）生成合成验证数据 → 训练 generative PRM → 用 PRM 做 RL reward，实现无 ground-truth 的 step-level RL，效果超过 ground-truth RLVR。

## 摘要
标准 RLVR 依赖 ground-truth 做 outcome-level reward，但 (1) 标注成本高，(2) outcome-level reward 提供的信号稀疏。SPARK 提出三阶段方案：首先利用 inference-time scaling 技术（大规模采样 + 验证）自动生成高质量的 step-level 验证数据，然后训练 generative [[process-reward-model|PRM]]，最后用 PRM 做 RL 的 reward 信号。关键发现：用合成数据训练的 PRM 在 ProcessBench 上超过了用 ground-truth 训练的 PRM 和 GPT-4o。

## 核心贡献
1. **无需标注的 PRM 训练流程**：通过 self-consistency + meta-critique 生成 step-level 验证数据，完全不需要人工标注
2. **Generative PRM 三种变体**：ORM（outcome-level）、PRM（step-level）、PRM-CoT（带链式推理的 step-level），PRM-CoT 效果最佳
3. **四种 Reward Formulation**：系统对比 Process-Aware、Step-Augmented、Selective Advantage、Global Step-Reward 四种方式
4. **深入的 Reward Hacking 分析**：发现三种 exploitation pattern 并提出对策
5. **超越 ground-truth RLVR**：PRM-CoT + process-aware rewards 在数学推理上超过用 ground-truth 做 reward 的传统 RLVR

## 方法

### 问题定义
两个核心问题：(1) 如何在没有标注的情况下训练可靠的 [[process-reward-model|PRM]]？(2) 如何用 PRM 做 RL reward 而不被 exploit？

### 技术方案

#### Stage 1: 合成验证数据生成（Inference-Time Scaling）

用两个预训练模型——Generator (Qwen2.5-14B-Instruct) 和 Verifier (Qwen-3-32B-Instruct)：

1. **Generator** 对每道题生成 M=8 个完整解法
2. **Verifier** 对每个解法独立验证 N=16 次
3. 四种验证方法：
   - **Outcome-level consistency**：N 次独立求解，答案一致性作为 label
   - **Step-level consistency**：从某步开始重新求解 N 次，一致性作为该步正确性 label
   - **Meta-critique**：Verifier 直接评估每步的推理质量
   - **Hybrid**：综合以上信号

**关键洞察**：Step-level consistency 效果最好，因为它迫使 Verifier "从这步出发重新做"，如果这步有错，后续的答案一致性会很低。

#### Stage 2: 训练 Generative PRM

在 Skywork-OR1-RL-Data 的 8000 道数学题上训练，fine-tuned from Qwen2.5-14B-Instruct：

- **ORM**: 只预测最终答案的正确性
- **PRM**: 对每步预测 correct/incorrect label
- **PRM-CoT**: 对每步先生成分析推理（chain-of-thought），再给出 label。效果最佳

#### Stage 3: PRM 做 RL Reward

用 [[grpo|GRPO]] 在 Qwen2.5-Math-7B 上训练。四种 reward formulation：

1. **Process-Aware Reward**: $R = R_{outcome} \cdot R_{process}$，最终答案分 × 过程分
2. **Step-Augmented Reward**: 在 outcome reward 基础上加 step-level bonus
3. **Selective Advantage**: 只对 PRM 认为 correct 的步骤计算 advantage
4. **Global Step-Reward**: 用所有步骤的平均 PRM 分替代 outcome reward

### 关键公式

**Process-Aware Reward**（效果最佳的 formulation）:

$$R_{PA}(q, o) = R_{outcome}(q, o) \cdot \prod_{s=1}^{S} R_{PRM}(q, o, s)^{1/S}$$

其中 $R_{outcome}$ 是 PRM 对最终答案的判断，$R_{PRM}(q, o, s)$ 是第 $s$ 步的 PRM 分数。

**GRPO 目标函数**（与标准 GRPO 相同，只是 reward 用 PRM 替代 ground-truth）:

$$\mathcal{L}_{GRPO}(\theta) = -\mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left(r(\theta) \hat{A}_i, \; \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL} \right]$$

## 实验结果

### ProcessBench（PRM 质量评估）

| 训练方法 | F1 Score |
|----------|----------|
| Random baseline | 50.0 |
| GPT-4o | 61.9 |
| Reference-guided (ground-truth) | 66.4 |
| **SPARK (step-level consistency)** | **67.5** |

SPARK 训练的 PRM 超过了用 ground-truth 训练的 PRM 和 GPT-4o。

### RL 最终结果（Qwen2.5-Math-7B）

| 方法 | MATH500 | GSM8K | Minerva | OlympiadBench | AMC | AIME | 平均 |
|------|---------|-------|---------|---------------|-----|------|------|
| Base | 37.0 | 60.1 | 11.0 | 15.5 | 25.0 | 3.3 | 25.3 |
| Ground-truth RLVR | 72.8 | **87.6** | 28.5 | 32.1 | 42.5 | 0.0 | 43.9 |
| SPARK (ORM) | 71.0 | 85.2 | 29.4 | 28.3 | 37.5 | 6.7 | 43.0 |
| SPARK (PRM) | 72.4 | 86.5 | 30.5 | 30.2 | 40.0 | 6.7 | 44.4 |
| **SPARK (PRM-CoT)** | **74.0** | 87.1 | **32.4** | **33.0** | **45.0** | **6.7** | **47.4** |

PRM-CoT + process-aware rewards **超过 ground-truth RLVR 3.5 个百分点**。

### Reward Hacking 分析

发现三种 exploitation pattern：

| Pattern | 描述 | 表现 |
|---------|------|------|
| **Solution Appending** | 在正确答案后追加无关内容 | PRM 只看步骤不看冗余 |
| **Step Inflation** | 人为增加步骤数稀释错误步骤的影响 | 更多步骤 → 单步错误权重降低 |
| **Step Reduction** | 压缩为极少步骤以最小化出错机会 | 输出变成 "答案=X" 一步 |

**对策**: Format constraints（限制步骤数范围）+ proper reward design（process-aware 比其他 formulation 更抗 hack）。

### 关键对比：Self-Consistency 直接做 Reward vs 训练 PRM

直接用 [[self-consistency|self-consistency]] 做在线 reward → 约 150 步后模型 collapse（学会生成相同的错误答案）。但用 self-consistency 生成 PRM 训练数据，再用冻结的 PRM 做 reward 是稳定的。

**原因**: 在线 self-consistency reward 是 non-stationary（随模型更新变化），PRM 是 stationary（冻结的外部模型），策略优化更稳定。

## 消融实验
- **PRM-CoT > PRM > ORM**: Chain-of-thought 让 PRM 的判断更准确
- **Process-Aware > 其他 formulation**: 同时考虑结果和过程最有效
- **Step-level consistency > outcome-level > meta-critique**: 生成 PRM 训练数据的最佳方式
- **8000 题 vs 更多/更少**: 8000 题是效率和效果的最佳平衡点

## 与其他工作的关系
- **基于**: [[grpo|GRPO]] 框架、[[process-reward-model|PRM]] 概念
- **核心概念**: [[self-consistency|Self-Consistency]]、[[process-reward-model|Process Reward Model]]
- **对比**: 
  - 标准 RLVR（ground-truth reward）— SPARK 超越之
  - [[wiki/papers/zhang-2025-empo|EMPO]]（纯内部信号）— SPARK 用外部 PRM 更可靠
  - [[wiki/papers/ghimire-2026-prism|PRISM]]（PRM + self-certainty 混合）— SPARK 是纯 PRM 路线
- **验证了 PRISM 的发现**: 纯 self-consistency 在线 reward 会 collapse，需要 stationary 的外部模型
- **被引用**: [[wiki/papers/ghimire-2026-prism|PRISM]] 和 [[wiki/papers/wu-2026-self-judge|Self-Judge]] 都讨论了类似的 reward hacking 问题

## 面试相关
> SPARK 是 URLVR 中 PRM 路线的代表，面试高频

- Q: SPARK 怎么在没有 ground-truth 的情况下训练 PRM？🔴
- A: 三阶段：(1) 用大模型大规模采样 + 验证生成合成标注——对每道题采样 8 个解法，每个解法用 Verifier 从每步开始重新解 16 次，一致性作为 step-level label；(2) 在合成数据上 fine-tune 得到 PRM；(3) 冻结 PRM 做 RL reward。

- Q: 为什么不直接用 self-consistency 做在线 reward？🔴
- A: 会 collapse。Self-consistency 是 non-stationary reward（随模型更新而变），模型 ~150 步后学会生成相同的错误答案来最大化 consistency。用 self-consistency 生成数据训练冻结的 PRM 做 reward 是 stationary 的，训练稳定。

- Q: SPARK 的 PRM 怎么超过 ground-truth 训练的 PRM？🟡
- A: Step-level consistency 提供的信号比简单的 correct/incorrect label 更丰富——它不仅知道最终答案对不对，还知道每步推理是否可靠（通过"从这步重新做 16 次"的一致性）。这种细粒度信号让 PRM 学到更好的 step-level 判断。

- Q: SPARK 发现了哪些 reward hacking 模式？如何应对？🔴
- A: 三种：solution appending（追加冗余内容）、step inflation（增加步骤数稀释错误）、step reduction（极度压缩输出）。对策：format constraints 限制步骤数范围 + process-aware reward design（同时看结果和过程）。

## 个人笔记
> SPARK 的三阶段框架设计很精巧：先用 inference-time scaling 的"笨办法"（大量采样+验证）生成高质量数据，再蒸馏成高效的 PRM。关键洞察是 self-consistency 不能直接做 reward（non-stationary），但可以做 PRM 训练数据的 label（stationary）。
>
> PRM-CoT 超越 ground-truth RLVR 的结果非常有启发性——step-level 的细粒度 reward 比 outcome-level 的 binary reward 提供了更多梯度信息，即使标注源是合成的而非 ground-truth。
>
> 实际应用启示：如果能承受 Stage 1 的计算成本（8个解法 × 16次验证 × 8000题），SPARK 是目前最可靠的 reference-free RLVR 方案。
