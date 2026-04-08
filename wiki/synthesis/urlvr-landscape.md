---
title: URLVR 领域综述：无监督/无参考强化学习推理
type: synthesis
tags: [URLVR, 综述, 对比分析, reward-signal, PRM, self-consistency, RAG, sharpening]
created: 2026-04-07
updated: 2026-04-08
sources: [wiki/papers/zuo-2025-ttrl.md, wiki/papers/zhang-2025-empo.md, wiki/papers/rahman-2025-spark.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/wu-2026-self-judge.md, wiki/papers/royer-2026-mcnig.md, wiki/papers/wang-2026-prorag.md, wiki/papers/tan-2026-ctrl-rag.md, wiki/papers/he-2026-urlvr-scale.md]
status: active
---

# URLVR 领域综述：无监督/无参考强化学习推理

## 领域定义

**URLVR (Unsupervised RL for Verifiable Reasoning)** / **Reference-Free RLVR**：在没有 ground-truth 答案标注的情况下，通过强化学习提升 LLM 推理能力的研究方向。

**核心挑战**：标准 RLVR（如 DeepSeek-R1）依赖 ground-truth 做 rule-based verification，但大量实际场景（开放域推理、多模态推理、RAG、复杂任务）无法获取标注。如何在没有标注的情况下构建可靠的训练信号？

## 九篇核心论文速览

| 论文 | 机构 | 核心方法 | Reward 来源 | 任务 | 年份 |
|------|------|----------|-------------|------|------|
| [[wiki/papers/zuo-2025-ttrl\|TTRL]] | Tsinghua + Shanghai AI Lab | Majority voting pseudo-reward + online RL | 纯内部（cross-sample consensus） | 数学/通用推理 | 2025 |
| [[wiki/papers/zhang-2025-empo\|EMPO]] | Tianjin U + Tencent | 语义熵最小化 | 纯内部（semantic entropy） | 数学/通用推理 | 2025 |
| [[wiki/papers/rahman-2025-spark\|SPARK]] | Amazon + UCLA | 三阶段 PRM 训练 | 外部（trained PRM） | 数学推理 | 2025 |
| [[wiki/papers/ghimire-2026-prism\|PRISM]] | ASU + AWS | PRM + self-certainty 混合 | 混合（PRM + 内部信号） | 数学推理 | 2026 |
| [[wiki/papers/wu-2026-self-judge\|Self-Judge]] | OPPO + Tsinghua | Actor-Judge + distributional | 混合（SC + Judge modulation） | 多模态推理 | 2026 |
| [[wiki/papers/royer-2026-mcnig\|MCNIG]] | IBM Research | 信息论 PRM 数据生成 | 外部（trained PRM） | 数学/代码/SQL/医学 | 2026 |
| [[wiki/papers/wang-2026-prorag\|ProRAG]] | Renmin University | 四阶段 RAG RL + MCTS PRM | 混合（PRM + outcome F1） | Multi-hop QA | 2026 |
| [[wiki/papers/tan-2026-ctrl-rag\|CTRL-RAG]] | Ant Group | Contrastive Likelihood Reward | 混合（CLR + accuracy） | RAG Faithfulness | 2026 |
| [[wiki/papers/he-2026-urlvr-scale\|How Far URLVR Scale?]] | Tsinghua | 统一分析 + sharpening 理论 | 综合分析（intrinsic vs external） | 综合 | 2026 (ICLR) |

---

## 多维分类体系

### 维度一：按 Reward 信号来源

```
纯内部信号 ─────────── 混合信号 ──────────── 纯外部信号
    │      │           │    │    │               │      │
  TTRL    EMPO       PRISM  S-J  CTRL-RAG       SPARK  MCNIG
(maj vote)(sem ent) (PRM+SC)(SC+J)(CLR+acc)    (PRM)  (PRM)
                        ProRAG
                     (PRM+F1)
```

| 类型 | 代表 | 优势 | 劣势 |
|------|------|------|------|
| **纯内部** | TTRL, EMPO | 无需任何外部模型，完全自主 | TTRL: 无 step-level credit；EMPO: 长期训练 [[reward-hacking\|reward hack]]（PRISM 证明）；本质是 sharpening（He et al. 证明） |
| **混合** | PRISM, Self-Judge, ProRAG, CTRL-RAG | 互补信号，更稳健 | 需要调节多信号权重（γ, β 等超参） |
| **纯外部** | SPARK, MCNIG | 最稳定（stationary reward） | 需要额外训练 PRM，计算成本高 |

### 维度二：按打分粒度

| 粒度 | 论文 | 描述 |
|------|------|------|
| **答案级 (Outcome-level)** | TTRL, EMPO, Self-Judge, CTRL-RAG | 只评估最终答案的质量/一致性/忠实度 |
| **步骤级 (Step-level)** | SPARK, PRISM, MCNIG, ProRAG | 评估每个推理步骤的正确性 |
| **双粒度 (Dual-granularity)** | ProRAG | 同时使用 outcome + step-level 信号 |

### 维度三：按 PRM 训练数据生成方式

| 方法 | 论文 | 策略 | 复杂度 |
|------|------|------|--------|
| **Step-level Self-consistency** | [[wiki/papers/rahman-2025-spark\|SPARK]] | 从每步重新采样 N 次看答案一致性 | O(M×N) |
| **信息论 (MCNIG)** | [[wiki/papers/royer-2026-mcnig\|MCNIG]] | 对比 correct/incorrect 答案的 log-prob 变化 | O(N) |
| **MCTS + Contrastive Labeling** | [[wiki/papers/wang-2026-prorag\|ProRAG]] | MCTS 探索 + GPT-4o 标注 sibling nodes | O(NlogN) |
| **现成 PRM** | [[wiki/papers/ghimire-2026-prism\|PRISM]] | 直接用 GenPRM-7B | 0（无需训练） |

### 维度四：按任务类型

| 任务类型 | 论文 | Base Model |
|----------|------|------------|
| **数学推理** | TTRL, EMPO, SPARK, PRISM, MCNIG | Qwen2.5-Math-7B, Qwen2.5-3B/7B/32B, Ministral-8B/14B, LLaMA-3.x |
| **通用推理** | TTRL, EMPO | Qwen2.5-7B (MMLU-Pro, GPQA) |
| **多模态视觉推理** | Self-Judge | Qwen2.5-VL-7B (几何、图表) |
| **代码 & SQL** | MCNIG | Ministral-8B/14B |
| **Multi-hop QA (RAG)** | ProRAG, CTRL-RAG | Qwen3-8B, Qwen3-30B-A3B |
| **医学 QA** | MCNIG | Ministral-8B/14B |

### 维度五：按优化框架

所有论文都基于 [[grpo|GRPO]]，但有细微差异：

| 论文 | 优化框架 | Advantage 计算 |
|------|----------|---------------|
| EMPO | 标准 GRPO | 语义聚类 reward 的组内 z-score |
| SPARK | 标准 GRPO | PRM reward 的组内 z-score |
| PRISM | 标准 GRPO | γ·Â_SC + Â_PRM |
| Self-Judge | 改进 GRPO | Energy-based log-sum-exp baseline |
| MCNIG | — (仅 best-of-K) | 未集成到 RL |
| ProRAG | 标准 GRPO | A^out + β·A^proc（dual-granularity） |
| CTRL-RAG | 标准 GRPO | R_CLR × R_acc（multiplicative gating） |

---

## 统一理论框架：Sharpening Mechanism

> 来自 [[wiki/papers/he-2026-urlvr-scale|He et al. 2026 (ICLR 2026)]]

### 核心论点

所有 intrinsic reward methods 本质上都在做同一件事——**sharpening the model's initial distribution**（锐化模型初始分布）。

### Sharpening 的成功条件

$$\text{Sharpening 有效} \iff P_{init}(\text{correct answer}) > P_{init}(\text{any incorrect answer})$$

- **Confidence-Correctness Aligned**: 模型对正确答案最有信心 → sharpening 提升 accuracy
- **Misaligned**: 模型对错误答案更有信心 → sharpening **灾难性失败**

### Rise-then-Fall Pattern

所有 intrinsic 方法的训练曲线呈现相同模式：
1. **Rise phase**: 在 aligned 样本上快速提升
2. **Fall phase**: 在 misaligned 样本上过度 sharpen → accuracy 急剧下降

这解释了 PRISM 观察到的"100 步后崩溃"现象。

### Model Collapse Step (MCS)

- **定义**: Rise → Fall 拐点步数
- **决定因素**: Model prior（预训练质量），而非工程超参数
- **实用价值**: MCS 高 = base model 适合 intrinsic rewards 训练

### 对各论文的理论解释

| 论文 | Sharpening 视角 |
|------|----------------|
| EMPO | Semantic entropy minimization = sharpening；entropy thresholding 是过滤 misaligned 样本的启发式 |
| PRISM | Rise-then-fall = sharpening 的必然结果；PRM 提供非 sharpening 的 external 信号补救 |
| SPARK | Trained PRM = external reward，不受 sharpening 限制，所以最稳定 |
| Self-Judge | Frozen Judge = external 校准，SC 频率 = intrinsic sharpening；混合部分缓解 |
| CTRL-RAG | CLR 利用文档对比 = computational asymmetry，可能属于 external |

---

## 性能对比

### 数学推理 Benchmark（Qwen2.5-Math-7B 或相近模型）

| 方法 | MATH500 | GSM8K | Minerva | 平均 | vs GT RLVR |
|------|---------|-------|---------|------|-----------|
| Base | ~37% | ~60% | ~11% | ~25% | — |
| GT RLVR | ~73% | ~88% | ~29% | ~44% | baseline |
| **EMPO** | 70.4% | 88.7% | 35.5% | ~48% | **~97%** |
| **SPARK (PRM-CoT)** | 74.0% | 87.1% | 32.4% | ~47% | **>100%** |
| **PRISM** | 80.8% | 92.1% | 38.6% | ~53%* | **>100%** |

*PRISM 数据来自 Qwen2.5-7B on DAPO-17k，不完全可比。

### Best-of-K Reranking（MCNIG）

| 方法 | 8 Benchmarks 平均 |
|------|-----------------|
| Majority Voting | 52.2% |
| QwenPRM 7B | 56.8% |
| **MCNIG 8B** | **62.3%** |
| **MCNIG 14B** | **63.4%** |

### RAG Benchmarks (Multi-hop QA)

| 方法 | PopQA | HotpotQA | MuSiQue | 平均 |
|------|-------|----------|---------|------|
| Search-R1 | 56.7 | 54.3 | 47.0 | ~51% |
| **ProRAG** | **60.3** | **57.6** | **55.3** | **~56%** |
| **CTRL-RAG** (think) | — | 84.3* | 82.1* | **~85%*** |

*CTRL-RAG 使用 Qwen3-8B think mode 且训练数据可能不同，与 ProRAG 不完全可比。

### 关键发现
- 最好的 URLVR 方法已经**达到甚至超过 ground-truth RLVR** 的效果（数学推理）
- RAG 任务上，process supervision（ProRAG）和 faithfulness reward（CTRL-RAG）都大幅提升基线
- MCNIG 的 PRM 在 best-of-K 上超过所有基线，且计算成本最低

---

## 核心发现与共识

### 1. 纯内部信号长期不可靠
**共识度**: ⭐⭐⭐⭐⭐（所有论文都涉及）

PRISM 系统性证明，SPARK 间接验证（self-consistency collapse），EMPO 通过 entropy thresholding 部分缓解，He et al. 从理论上解释（sharpening mechanism + confidence-correctness misalignment）。

### 2. Stationary Reward 优于 Non-stationary
**共识度**: ⭐⭐⭐⭐⭐

冻结的外部模型（SPARK 的 PRM、Self-Judge 的 frozen Judge、ProRAG 的 PRM）比在线计算的信号（直接 self-consistency、直接 entropy）更可靠。

### 3. Step-level 信号通常优于 Outcome-level
**共识度**: ⭐⭐⭐⭐

SPARK 的 PRM-CoT 超过 GT RLVR（outcome-level），PRISM 的 PRM 比 self-certainty 更可靠，ProRAG 的 dual-granularity 超过纯 outcome。

### 4. 混合多种信号更稳健
**共识度**: ⭐⭐⭐⭐

PRISM 的 PRM+SC，Self-Judge 的 SC+Judge，ProRAG 的 outcome+process，CTRL-RAG 的 CLR×Acc——所有混合方案都超过单一信号。

### 5. Reward Hacking 是核心挑战
**共识度**: ⭐⭐⭐⭐⭐

所有论文都发现或讨论了 [[reward-hacking|reward hacking]]，包括 verbose collapse（CTRL-RAG）、step inflation/reduction（SPARK）、confidence inflation（PRISM）。

### 6. GRPO 是当前默认框架
**共识度**: ⭐⭐⭐⭐⭐

7 篇（除 MCNIG 仅做 best-of-K）都用 [[grpo|GRPO]]，说明它在 LLM RL 中的主导地位。

### 7. PRM 训练数据可以自动生成
**共识度**: ⭐⭐⭐⭐（SPARK, MCNIG, ProRAG 共同验证）

三种不同的自动标注方案（self-consistency、信息论、MCTS+contrastive）都证明合成 PRM 数据质量足够高，甚至超过 ground-truth 标注。

---

## 研究谱系与信号演化

```
Self-Consistency (Wang et al., 2022)
    │
    ├── 直接做 Reward ──→ Collapse (~150步, SPARK发现)
    │
    ├── 语义聚类版本 ──→ EMPO (Semantic Entropy Minimization)
    │                      └── 短期有效，本质是 sharpening (He et al.)
    │
    ├── 生成PRM训练数据 ──→ SPARK (Step-level PRM, 超越GT)
    │                      └── 最稳定的方案
    │
    ├── + Self-Certainty ──→ PRISM (混合信号, 互补)
    │                      └── 系统性失败分析的价值
    │
    ├── + Judge Modulation ──→ Self-Judge (多模态, Distributional)
    │                          └── 分布建模的理论贡献
    │
    └── 统一理论 ──→ He et al. (ICLR 2026): Sharpening + MCS

Information Theory
    │
    └── Information Gain ──→ MCNIG (O(N) PRM 训练数据, 跨任务)
                              └── 信息论替代重采样方法

RAG + RL
    │
    ├── MCTS + PRM Pipeline ──→ ProRAG (四阶段, dual-granularity)
    │                            └── 最完整的 RAG RL 框架
    │
    └── Contrastive Likelihood ──→ CTRL-RAG (轻量 faithfulness reward)
                                    └── 无需额外模型的 RAG reward
```

---

## 扩展分类维度

### 按 Reward 模型实现方式（完整版）

| 实现方式 | 论文 | 描述 |
|----------|------|------|
| 语义聚类频率 | EMPO | 多次采样 → 语义聚类 → 频率做 reward |
| Trained PRM (self-consistency) | SPARK | 合成数据训练的 generative PRM |
| Trained PRM (信息论) | MCNIG | [[information-gain\|MCNIG]] 自动标注训练的 PRM |
| Trained PRM (MCTS) | ProRAG | [[mcts\|MCTS]] 探索 + GPT-4o 标注训练的 PRM |
| 现成 PRM | PRISM | 直接用 GenPRM-7B |
| Self-consistency + Modulation | Self-Judge | SC 频率 × bounded Judge 评分 |
| [[contrastive-likelihood\|Contrastive Likelihood]] | CTRL-RAG | 有/无文档的 log-likelihood 对比 |

### 按 Length Bias 处理方式

| 论文 | Length Bias 处理 |
|------|----------------|
| EMPO | Token-level loss normalization (1/\|o_i\|) |
| SPARK | PRM-CoT 本身对长度不敏感 |
| PRISM | GRPO 标准归一化 |
| Self-Judge | Group-wise distributional baseline |
| MCNIG | N/A (best-of-K) |
| ProRAG | Z-score normalization |
| CTRL-RAG | **√T 归一化**（关键发现：/T 太强，无归一化导致 verbose collapse） |

---

## 开放问题与研究方向

### 1. 长期训练稳定性
即使最好的方法（SPARK、PRISM），在非常长期的训练中是否仍然稳定？He et al. 的 MCS 提供了预测工具，但尚未在大规模训练中验证。

### 2. 跨任务泛化
EMPO 和 SPARK 主要在数学上验证，Self-Judge 在多模态上验证，ProRAG/CTRL-RAG 在 RAG 上验证。MCNIG 展示了跨任务的 PRM 训练（数学+代码+SQL+医学），但未集成到 RL。是否能统一到一个框架？

### 3. PRM 训练数据方法的统一
三种自动生成方案（SPARK self-consistency、MCNIG 信息论、ProRAG MCTS）各有优劣。能否组合或选择最优？MCNIG 效率最高但需要 ground-truth 分组，SPARK 完全无监督但 O(M×N)，ProRAG 质量最高但成本最大。

### 4. RAG + Process Supervision 的深化
ProRAG 和 CTRL-RAG 开辟了 RAG RL 的新方向。能否将 ProRAG 的 step-level PRM 和 CTRL-RAG 的 faithfulness reward 结合？

### 5. Sharpening 的根本突破
He et al. 证明 intrinsic rewards 受限于 confidence-correctness alignment。如何突破？Computational asymmetries（验证比生成容易）是一条路，MCNIG 的信息论方法是另一条路。

### 6. MCNIG 集成到 RL Pipeline
MCNIG 目前仅做 best-of-K reranking。如果用 MCNIG 训练的 PRM 做 RL reward（类似 SPARK 的 Stage 3），效果会如何？

### 7. Reward Hacking 的根本解决
当前方案都是缓解而非根本解决。是否存在理论上不可 hack 的 reward 构建方式？

---

## 面试综合题

- Q: 介绍一下 URLVR 领域的主要方法路线和它们的优缺点。🔴
- A: 三条路线：(1) **纯内部信号**（EMPO 的语义熵）——简单自主但长期不可靠，本质是 sharpening；(2) **外部 PRM**（SPARK、MCNIG）——最稳定但计算成本高；(3) **混合信号**（PRISM 的 PRM+SC、Self-Judge 的 SC+Judge、ProRAG 的 PRM+F1、CTRL-RAG 的 CLR+Acc）——兼顾稳定性和效率。He et al. (ICLR 2026) 统一理论：所有 intrinsic 方法都在做 sharpening，成功取决于 confidence-correctness alignment。

- Q: 对比三种 PRM 训练数据自动生成方法。🔴
- A: (1) SPARK：从每步重新采样 N 次看答案一致性，完全无监督但 O(M×N)；(2) MCNIG：用信息论（log-prob 变化）标注，O(N) 最高效，但需要知道正确/错误答案分组；(3) ProRAG MCTS：MCTS 探索 + GPT-4o contrastive labeling，96% 一致性最高质量，但成本最大。各有适用场景。

- Q: 如果让你设计一个新的 URLVR 方法，你会怎么做？🔴
- A: 基于当前最佳实践：(1) 用 MCNIG 高效生成 PRM 训练数据（O(N) 复杂度）；(2) 像 PRISM 一样混合 PRM 和 self-certainty 信号做 advantage；(3) 像 Self-Judge 一样用 energy-based distributional modeling 替代简单的 z-score baseline；(4) 加 EMPO 的 entropy thresholding 过滤极端样本；(5) 借鉴 He et al. 的 MCS 来决定何时停止训练或切换到 external rewards；(6) 如果是 RAG 任务，加入 CTRL-RAG 的 CLR 做 faithfulness 信号。

- Q: RAG 任务上的 RL 有哪些方法？各自优缺点？🔴
- A: 两种代表方案：(1) **ProRAG**——四阶段 pipeline（SFT → MCTS PRM → RFT → Process-Supervised RL），dual-granularity advantage，效果最好但最重；(2) **CTRL-RAG**——contrastive likelihood reward（有/无文档的 log-likelihood 对比），轻量无需额外模型，重点关注 faithfulness。ProRAG 提供 step-level 精细信号，CTRL-RAG 提供 outcome-level 的 faithfulness 信号。资源充足选 ProRAG，轻量部署选 CTRL-RAG，两者可以结合。
