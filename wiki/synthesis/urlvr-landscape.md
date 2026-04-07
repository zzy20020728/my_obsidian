---
title: URLVR 领域综述：无监督/无参考强化学习推理
type: synthesis
tags: [URLVR, 综述, 对比分析, reward-signal, PRM, self-consistency]
created: 2026-04-07
updated: 2026-04-07
sources: [wiki/papers/zhang-2025-empo.md, wiki/papers/rahman-2025-spark.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/wu-2026-self-judge.md]
status: active
---

# URLVR 领域综述：无监督/无参考强化学习推理

## 领域定义

**URLVR (Unsupervised RL for Verifiable Reasoning)** / **Reference-Free RLVR**：在没有 ground-truth 答案标注的情况下，通过强化学习提升 LLM 推理能力的研究方向。

**核心挑战**：标准 RLVR（如 DeepSeek-R1）依赖 ground-truth 做 rule-based verification，但大量实际场景（开放域推理、多模态推理、复杂任务）无法获取标注。如何在没有标注的情况下构建可靠的训练信号？

## 四篇核心论文速览

| 论文 | 机构 | 核心方法 | Reward 来源 | 模态 |
|------|------|----------|-------------|------|
| [[wiki/papers/zhang-2025-empo\|EMPO]] | Tianjin U + Tencent | 语义熵最小化 | 纯内部（semantic entropy） | 文本 |
| [[wiki/papers/rahman-2025-spark\|SPARK]] | Amazon + UCLA | 三阶段 PRM 训练 | 外部（trained PRM） | 文本 |
| [[wiki/papers/ghimire-2026-prism\|PRISM]] | ASU + AWS | PRM + self-certainty 混合 | 混合（PRM + 内部信号） | 文本 |
| [[wiki/papers/wu-2026-self-judge\|Self-Judge]] | OPPO + Tsinghua | Actor-Judge + distributional | 混合（SC + Judge modulation） | 多模态 |

---

## 多维分类体系

### 维度一：按 Reward 信号来源

```
纯内部信号 ─────────────── 混合信号 ─────────────── 纯外部信号
    │                        │     │                      │
   EMPO                   PRISM  Self-Judge             SPARK
(semantic entropy)    (PRM+SC)  (SC+Judge)         (trained PRM)
```

| 类型 | 代表 | 优势 | 劣势 |
|------|------|------|------|
| **纯内部** | EMPO | 无需任何外部模型，完全自主 | 长期训练 [[reward-hacking\|reward hack]]（PRISM 证明） |
| **混合** | PRISM, Self-Judge | 互补信号，更稳健 | 需要调节多信号权重（γ等超参） |
| **纯外部** | SPARK | 最稳定（stationary reward） | 需要额外训练 PRM，计算成本高 |

### 维度二：按打分粒度

| 粒度 | 论文 | 描述 |
|------|------|------|
| **答案级 (Outcome-level)** | EMPO, Self-Judge | 只评估最终答案的质量/一致性 |
| **步骤级 (Step-level)** | SPARK, PRISM | 评估每个推理步骤的正确性 |

| 对比维度 | 答案级 | 步骤级 |
|----------|--------|--------|
| 信号密度 | 稀疏（每条轨迹一个分数） | 密集（每步一个分数） |
| 信息量 | 只知道对不对 | 知道哪步出问题 |
| 实现难度 | 简单 | 需要 PRM 或 step-level 验证 |
| Reward hacking | Step inflation/reduction 不适用 | 需防御 step inflation/reduction |
| 经验效果 | 基线效果 | 通常更优（SPARK PRM-CoT > GT RLVR） |

### 维度三：按 Reward 模型实现方式

| 实现方式 | 论文 | 描述 |
|----------|------|------|
| **语义聚类频率** | EMPO | 多次采样 → 语义聚类 → 频率做 reward |
| **Trained PRM** | SPARK | 合成数据训练的 generative PRM |
| **现成 PRM** | PRISM | 直接用 GenPRM-7B |
| **Self-consistency + Modulation** | Self-Judge | SC 频率 × bounded Judge 评分 |

### 维度四：按任务类型

| 任务类型 | 论文 | Base Model |
|----------|------|------------|
| **数学推理** | EMPO, SPARK, PRISM | Qwen2.5-Math-7B, Qwen2.5-3B/7B |
| **通用推理** | EMPO | Qwen2.5-7B (MMLU-Pro, GPQA) |
| **多模态视觉推理** | Self-Judge | Qwen2.5-VL-7B (几何、图表) |

### 维度五：按优化框架

所有四篇都基于 [[grpo|GRPO]]，但有细微差异：

| 论文 | 优化框架 | Advantage 计算 |
|------|----------|---------------|
| EMPO | 标准 GRPO | 语义聚类 reward 的组内 z-score |
| SPARK | 标准 GRPO | PRM reward 的组内 z-score |
| PRISM | 标准 GRPO | γ·Â_SC + Â_PRM |
| Self-Judge | 改进 GRPO | Energy-based log-sum-exp baseline |

---

## 性能对比（数学推理 Benchmark）

### Qwen2.5-Math-7B 或相近模型

| 方法 | MATH500 | GSM8K | Minerva | 平均 | vs GT RLVR |
|------|---------|-------|---------|------|-----------|
| Base | ~37% | ~60% | ~11% | ~25% | — |
| GT RLVR | ~73% | ~88% | ~29% | ~44% | baseline |
| **EMPO** | 70.4% | 88.7% | 35.5% | ~48% | **~97%** |
| **SPARK (PRM-CoT)** | 74.0% | 87.1% | 32.4% | ~47% | **>100%** |
| **PRISM** | 80.8% | 92.1% | 38.6% | ~53%* | **>100%** |

*PRISM 数据来自 Qwen2.5-7B on DAPO-17k，不完全可比。

**关键发现**：最好的 URLVR 方法已经达到甚至超过 ground-truth RLVR 的效果。

---

## 核心发现与共识

### 1. 纯内部信号长期不可靠
**共识度**: ⭐⭐⭐⭐⭐（所有论文都涉及）

PRISM 系统性证明，SPARK 间接验证（self-consistency collapse），EMPO 通过 entropy thresholding 部分缓解但未根本解决。

### 2. Stationary Reward 优于 Non-stationary
**共识度**: ⭐⭐⭐⭐⭐

冻结的外部模型（SPARK 的 PRM、Self-Judge 的 frozen Judge）比在线计算的信号（直接 self-consistency、直接 entropy）更可靠。

### 3. Step-level 信号通常优于 Outcome-level
**共识度**: ⭐⭐⭐⭐

SPARK 的 PRM-CoT 超过 GT RLVR（outcome-level），PRISM 的 PRM 比 self-certainty 更可靠。

### 4. 混合多种信号更稳健
**共识度**: ⭐⭐⭐⭐

PRISM 的 PRM+SC 超过单一信号，Self-Judge 的 SC+Judge 超过单独使用。

### 5. Reward Hacking 是核心挑战
**共识度**: ⭐⭐⭐⭐⭐

所有论文都发现或讨论了 [[reward-hacking|reward hacking]]，这是 URLVR 最大的挑战。

### 6. GRPO 是当前默认框架
**共识度**: ⭐⭐⭐⭐⭐

四篇都用 [[grpo|GRPO]]，说明它在 LLM RL 中的主导地位。

---

## 研究谱系与信号演化

```
Self-Consistency (Wang et al., 2022)
    │
    ├── 直接做 Reward ──→ Collapse (~150步, SPARK发现)
    │
    ├── 语义聚类版本 ──→ EMPO (Semantic Entropy Minimization)
    │                      └── 短期有效，长期可能hack (PRISM质疑)
    │
    ├── 生成PRM训练数据 ──→ SPARK (Step-level PRM, 超越GT)
    │                      └── 最稳定的方案
    │
    ├── + Self-Certainty ──→ PRISM (混合信号, 互补)
    │                      └── 系统性失败分析的价值
    │
    └── + Judge Modulation ──→ Self-Judge (多模态, Distributional)
                               └── 分布建模的理论贡献
```

---

## 开放问题与研究方向

### 1. 长期训练稳定性
即使最好的方法（SPARK、PRISM），在非常长期的训练中是否仍然稳定？目前实验通常 <1000 步。

### 2. 跨任务泛化
EMPO 和 SPARK 主要在数学上验证，Self-Judge 在多模态上验证。是否能统一到一个框架？

### 3. 更好的语义等价性判断
EMPO 用 1.5B SLM 做语义聚类，在复杂自由形式任务上的质量如何？

### 4. PRM 的 Scalability
SPARK 的 PRM 训练需要大量计算（8×M×N 次推理），能否更高效？

### 5. 与 RLHF 的结合
URLVR 目前主要用于推理任务。能否扩展到更广泛的对齐场景（helpfulness、harmlessness）？

### 6. Reward Hacking 的根本解决
当前方案都是缓解而非根本解决。是否存在理论上不可 hack 的 reward 构建方式？

---

## 面试综合题

- Q: 介绍一下 URLVR 领域的主要方法路线和它们的优缺点。🔴
- A: 三条路线：(1) **纯内部信号**（EMPO 的语义熵）——简单自主但长期不可靠；(2) **外部 PRM**（SPARK）——最稳定但计算成本高；(3) **混合信号**（PRISM 的 PRM+SC、Self-Judge 的 SC+Judge）——兼顾稳定性和效率。核心挑战是 reward hacking。当前共识是 stationary + 混合信号 + step-level 是最优组合。

- Q: 如果让你设计一个新的 URLVR 方法，你会怎么做？🔴
- A: 基于当前的最佳实践，我会：(1) 用 SPARK 的方式训练 PRM（step-level consistency → PRM-CoT）；(2) 像 PRISM 一样混合 PRM 和 self-certainty 信号做 advantage；(3) 像 Self-Judge 一样用 energy-based distributional modeling 替代简单的 z-score baseline；(4) 加 EMPO 的 entropy thresholding 过滤极端样本；(5) 全程监控 proxy-target correlation。
