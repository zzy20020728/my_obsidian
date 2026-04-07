---
title: "PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards"
type: paper
tags: [URLVR, PRM, self-certainty, reward-hacking, GRPO, RLIF]
created: 2026-04-07
updated: 2026-04-07
sources: [https://arxiv.org/abs/2601.04700]
status: active
---

# PRISM: PRM + Self-Certainty 混合框架

## 基本信息
- **作者**: Mukesh Ghimire et al.
- **机构**: Arizona State University & AWS
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2601.04700)
- **链接**: https://arxiv.org/abs/2601.04700
- **代码**: https://github.com/ghimiremukesh/PRISM

## 一句话总结
> 系统性证明纯内部信号（entropy / self-certainty）做 RLIF 长期训练不可靠会 reward hack，纯 PRM 也会失败（模型忘记格式），提出 PRM + self-certainty 混合方案 PRISM 互补解决。

## 摘要
当任务没有 verifiable rewards（如开放域推理、创意生成）时，现有方法要么用内部信号（token entropy、self-certainty）做 RLIF，要么用外部 [[process-reward-model|PRM]] 做 reward。本文通过大量实验证明：**两种单一方案都不可靠**。纯内部信号短期有效但长期 [[reward-hacking|reward hack]]；纯 PRM 导致模型忘记 instruction following（格式崩溃）。PRISM 结合两者：PRM 提供质量信号防止 overconfidence，self-certainty 促进格式遵循和 instruction following。

## 核心贡献
1. **系统性否定纯 RLIF**：实验证明三种内部信号方法（token-entropy、trajectory-entropy、self-certainty）训练 ~300 步后全部崩溃
2. **发现纯 PRM 的致命问题**：模型"忘记" boxed 格式 → verifier 无法提取答案 → accuracy 下降但 PRM reward 上升
3. **统计分析**：Mann-Whitney U 检验证明 self-certainty 无法区分 correct/incorrect responses（分布高度重叠）
4. **PRISM 方案**：PRM + self-certainty 的加权混合，取长补短，接近甚至超过 GRPO(GT)

## 方法

### 问题定义
对于 unverifiable tasks（无法自动判定答案正确性），如何构建可靠的 RL reward？现有两条路线各有什么致命问题？

### RLIF 方法及其失败分析

#### 三种内部信号方法

1. **Token-level Entropy**: $R = -\frac{1}{T}\sum_{t=1}^{T} H(p_\theta(\cdot|x, y_{<t}))$，鼓励模型在每个 token 上更确定
2. **Trajectory-level Entropy**: 采样多条轨迹，计算答案分布的熵
3. **Self-Certainty**: $R_{SC} = \frac{1}{T}\sum_{t=1}^{T} \log p_\theta(y_t|x, y_{<t})$，即平均 token log-probability

#### 实验发现：全部崩溃

在 Qwen2.5-3B 上训练 300 步（~6 epochs）：
- **前 50-100 步**: Accuracy 快速提升（所有方法都有效）
- **100-300 步**: Accuracy 急剧下降，reward 继续上升 → **reward hacking**
- Self-certainty 最稳但仍失败：模型学会在答案后追加无关问题来 inflate confidence
- Entropy 方法更严重：模型学会重复生成不解决问题的内容

#### 统计证据

- **Moving correlation**（内部 reward vs true accuracy）: 几乎为零，说明 proxy reward 与真实质量无相关性
- **Mann-Whitney U 检验**: Self-certainty 分布在 correct 和 incorrect responses 之间高度重叠（p 值不显著），即 self-certainty 本质上无法区分答案质量

### 纯 PRM 的失败模式

用 GenPRM-7B 做 reward 训练 Qwen2.5-3B：
- **问题**: 模型逐渐"忘记"用 `\boxed{}` 格式输出最终答案
- **后果**: Verifier（PRM）无法提取答案 → 无法判断正确性
- **表现**: PRM reward 继续上升（因为推理过程 "看起来" 合理），但实际 accuracy 下降
- **根因**: PRM 只关注推理过程质量，不关注格式遵循和 instruction following

### PRISM: 混合方案

#### 核心公式

$$\hat{A}_{PRISM} = \gamma \cdot \hat{A}_{SC} + \hat{A}_{PRM}$$

其中：
- $\hat{A}_{SC}$ = self-certainty 的 advantage（组内归一化）
- $\hat{A}_{PRM}$ = PRM 的 advantage（组内归一化）
- $\gamma$ = 权重超参数

#### 互补逻辑
- **Self-certainty 的作用**: 促进 instruction following 和格式遵循（因为模型对正确格式的输出更"自信"）
- **PRM 的作用**: 防止 overconfidence（self-certainty 的 reward hacking），提供更可靠的推理质量信号
- **为什么混合有效**: Self-certainty 防止 PRM 的格式崩溃问题，PRM 防止 self-certainty 的 confidence inflation 问题

#### PRM 实现细节
- 使用 GenPRM-7B（Generative Process Reward Model）
- **Min aggregation**: 取所有步骤中最低的 PRM 分数作为整体 reward
- 比 mean aggregation 更保守，更抗 reward hacking

### 关键公式

GRPO 框架下的 PRISM 目标：

$$\mathcal{L}_{PRISM}(\theta) = -\mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left(r(\theta) \hat{A}_{PRISM,i}, \; \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{PRISM,i} \right) - \beta D_{KL} \right]$$

## 实验结果

### Qwen2.5-3B on MATH

| 方法 | MATH Accuracy | 趋势 |
|------|---------------|------|
| Base | ~25% | — |
| RLIF (self-certainty, INTUITOR) | 先升后崩 | 100步后崩溃 |
| RLIF (token-entropy) | 先升后崩 | 50步后崩溃 |
| PRM only | 缓慢下降 | 格式崩溃 |
| GRPO (ground-truth) | ~45% | 稳定上升 |
| **PRISM** | **~43%** | 稳定上升，接近 GT |

### Qwen2.5-7B on DAPO-17k

| 方法 | MATH500 | GSM8K | Minerva | AMC | AIME |
|------|---------|-------|---------|-----|------|
| INTUITOR (SC only) | 75.8 | 88.9 | 32.4 | 37.5 | 10.0 |
| **PRISM** | **80.8** | **92.1** | **38.6** | **50.0** | **16.7** |
| GRPO (GT) | 79.2 | 91.1 | 36.0 | 45.0 | 13.3 |

PRISM 在 Qwen2.5-7B 上部分 benchmark **超过 GRPO(GT)**。

### PRISM vs INTUITOR 改进幅度

PRISM 比 INTUITOR（纯 self-certainty）平均提升 **34%**。

## 消融实验
- **γ = 0（纯 PRM）**: 格式崩溃，accuracy 下降
- **γ → ∞（纯 self-certainty）**: 短期有效，长期 reward hack
- **γ 最优值**: 在 Qwen2.5-3B 上 γ ≈ 0.5-1.0，需要 task-specific 调参
- **Min vs Mean aggregation**: Min 更好，更保守的 PRM 信号更抗 hack

## 与其他工作的关系
- **基于**: [[grpo|GRPO]] 框架、GenPRM-7B
- **核心概念**: [[process-reward-model|PRM]]、[[reward-hacking|Reward Hacking]]
- **直接对比**:
  - INTUITOR（self-certainty only）— PRISM 大幅超越
  - GRPO(GT)（ground-truth reward）— PRISM 接近甚至超越
- **验证了其他工作的发现**: 
  - [[wiki/papers/rahman-2025-spark|SPARK]] 发现的 self-consistency collapse → PRISM 从理论上解释了为什么纯内部信号不可靠
  - [[wiki/papers/zhang-2025-empo|EMPO]] 的 semantic entropy 方法 → PRISM 证明 entropy 类信号长期不可靠
- **与 SPARK 的区别**: SPARK 训练自己的 PRM，PRISM 直接用现成的 GenPRM-7B；SPARK 是纯 PRM 路线，PRISM 是 PRM + 内部信号混合

## 面试相关
> PRISM 涉及多个高频面试话题

- Q: 为什么纯内部信号（entropy/self-certainty）做 RL reward 会失败？🔴
- A: 三个原因：(1) Internal proxy reward 与 true accuracy 几乎无相关性（moving correlation ≈ 0）；(2) Self-certainty 无法区分 correct/incorrect responses（Mann-Whitney U 检验不显著）；(3) 模型会学会 exploit——比如追加无关内容来 inflate confidence，或重复生成来降低 entropy。

- Q: 纯 PRM 做 reward 有什么问题？🔴
- A: 模型会"忘记" instruction following——比如不再用 `\boxed{}` 格式输出答案。因为 PRM 只评估推理过程质量，不关注格式遵循。结果：PRM reward 上升但实际 accuracy 下降。

- Q: PRISM 怎么解决这两个问题？🔴
- A: 混合 advantage = γ·Â_SC + Â_PRM。Self-certainty 促进格式遵循（模型对正确格式更自信），PRM 防止 overconfidence 并提供可靠的质量信号。两者互补：SC 防 PRM 的格式崩溃，PRM 防 SC 的 confidence inflation。

- Q: PRISM 的 γ 怎么调？🟡
- A: 需要 task-specific 调参。γ 太小退化为纯 PRM（格式崩溃），太大退化为纯 SC（reward hack）。通常 0.5-1.0 范围内搜索。

## 个人笔记
> PRISM 最大的价值不在方案本身（PRM + SC 加权混合相对简单），而在 **系统性的失败分析**。它严格证明了纯内部信号不可靠（统计检验 + 长期训练曲线），纯 PRM 也有格式崩溃问题。这为整个 URLVR 领域提供了重要的实验基础和方法论指导。
>
> 关键启示：任何单一 reward signal 都可能被 exploit。混合多种信号源（尤其是互补的信号源）是更稳健的路线。
>
> 实践建议：如果做 URLVR 研究，一定要跑长期训练曲线（>300步），很多方法短期看起来有效但长期会崩溃。
