---
title: Self-Consistency (自一致性)
type: concept
tags: [推理, 采样, reward-signal, URLVR]
created: 2026-04-07
updated: 2026-04-07
sources: [wiki/papers/rahman-2025-spark.md, wiki/papers/wu-2026-self-judge.md]
status: active
---

# Self-Consistency (自一致性)

## 定义
> Self-Consistency，自一致性。一种 inference-time scaling 技术：对同一问题独立采样多条推理路径，通过答案的一致性（通常是 majority voting）选择最终答案。核心假设：**正确推理路径之间更可能得出相同答案**。在 URLVR 中被广泛用作构建无监督 reward 信号的基础。

## 关键性质
1. **多路径采样**：独立生成 N 条推理轨迹，不依赖单一路径
2. **答案聚合**：通常用 majority voting，但也可用加权投票或分布建模
3. **可作为不确定性度量**：一致性高 → 模型确定性高 → 答案更可能正确
4. **计算换准确率**：用更多采样次数换取更高的答案质量（inference-time scaling）

## 直觉理解
> 让同一个学生独立做 10 遍同一道数学题。如果 8 次得出 42，2 次得出其他值——42 很可能是正确答案。原理是正确推理有唯一路径（或多条正确路径收敛到同一答案），而错误推理各不相同。

## 数学表达

### 基本形式（Wang et al., 2022）

对问题 q 独立采样 N 条推理路径 {(r_1, a_1), ..., (r_N, a_N)}：

$$a^* = \arg\max_a \sum_{i=1}^{N} \mathbb{1}[a_i = a]$$

### 作为 Reward Signal

#### Outcome-level consistency（SPARK Stage 1）
$$R_{OC}(q, a) = \frac{|\{i : a_i = a\}|}{N}$$

#### Step-level consistency（SPARK Stage 1, 最有效）
对第 s 步之后重新采样 N 次：
$$R_{SC}(q, o, s) = \frac{|\{i : a_i^{(s)} = a_{majority}\}|}{N}$$

如果这步有错，后续重新求解的答案一致性会很低。

#### 语义聚类形式（EMPO, Self-Judge）
$$r_i = \hat{p}(a_i) = \frac{|\{j : a_j \sim a_i\}|}{N}$$

其中 $\sim$ 表示语义等价（EMPO）或精确匹配（Self-Judge）。

## 在 URLVR 中的应用

| 论文 | 使用方式 | 效果 |
|------|----------|------|
| [[wiki/papers/zhang-2025-empo\|EMPO]] | 语义聚类频率直接做 reward | 有效，但长期可能 hack |
| [[wiki/papers/rahman-2025-spark\|SPARK]] | 生成 PRM 训练数据（不直接做 reward） | 最稳定，训练出的 PRM 超 GT |
| [[wiki/papers/ghimire-2026-prism\|PRISM]] | 分析了 self-certainty（相关概念）的失败模式 | 纯内部信号不可靠 |
| [[wiki/papers/wu-2026-self-judge\|Self-Judge]] | SC 频率 + Judge modulation + distributional modeling | 多模态有效 |

### 关键发现：直接做 Reward vs 间接使用

**直接做在线 reward（不推荐）**: SPARK 发现 ~150 步后 collapse——模型学会生成相同的错误答案来最大化 consistency。因为在线 SC reward 是 non-stationary 的（随模型更新变化）。

**间接使用（推荐）**: 
- 生成 PRM 训练数据（SPARK）→ 冻结 PRM 做 reward（stationary）
- 配合 entropy thresholding（EMPO）→ 部分缓解 hack
- 配合 bounded Judge modulation（Self-Judge）→ 防止 binary amplification

## Self-Consistency vs Majority Voting

| 维度 | Self-Consistency | Majority Voting |
|------|-----------------|-----------------|
| 输出 | 完整的答案分布 | 单一最高票答案 |
| 信息保留 | 保留分布 shape | 只保留 argmax |
| 40%-35%-25% | 区分为"高不确定性" | 直接选 40% 的 |
| URLVR 中 | 分布信息可做 reward | binary reward，信息损失大 |

[[wiki/papers/wu-2026-self-judge|Self-Judge]] 重点论证了 MV 的局限性：MV 丢弃分布结构，导致 policy 快速 collapse。

## 相关论文
- Wang et al., 2022 — "Self-Consistency Improves Chain of Thought Reasoning in Language Models"（原始论文）
- [[wiki/papers/rahman-2025-spark|SPARK (Rahman et al., 2025)]] — 用 step-level consistency 生成 PRM 训练数据
- [[wiki/papers/zhang-2025-empo|EMPO (Zhang et al., 2025)]] — 用语义聚类频率（广义 self-consistency）做 reward
- [[wiki/papers/wu-2026-self-judge|Self-Judge (Wu et al., 2026)]] — 用 SC + Judge modulation 做多模态 URLVR

## 面试常问点

- 🔴 Q: Self-consistency 的核心思想是什么？为什么有效？
  - A: 对同一问题独立采样多条推理路径，答案一致性高的更可能正确。正确推理路径收敛到同一答案，错误推理各不相同。

- 🔴 Q: Self-consistency 直接做 RL reward 为什么会失败？
  - A: SPARK 发现 ~150 步后模型 collapse——学会生成相同的错误答案来最大化 consistency。因为在线 SC 是 non-stationary reward（随模型更新变化），模型可以 exploit 这个 moving target。

- 🟡 Q: Self-consistency 和 semantic entropy 的关系？
  - A: 都基于"多次采样看一致性"的思想。Self-consistency 关注答案频率（投票），semantic entropy 关注答案含义的熵（信息论度量）。Semantic entropy 更严格地在语义层面做聚类。

## 与其他概念的关系
- 上位概念：[[inference-time-scaling|Inference-Time Scaling]]、[[ensemble-methods|集成方法]]
- 相关概念：[[semantic-entropy|Semantic Entropy]]（语义层面的自一致性度量）
- 应用概念：[[process-reward-model|PRM]]（SPARK 用 SC 生成 PRM 训练数据）
- 风险：[[reward-hacking|Reward Hacking]]（直接做 reward 的 collapse 风险）
