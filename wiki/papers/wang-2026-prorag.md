---
title: "ProRAG: Process-Supervised Reinforcement Learning for Retrieval-Augmented Generation"
type: paper
tags: [URLVR, RAG, PRM, MCTS, process-supervision, multi-hop-QA, dual-granularity]
created: 2026-04-08
updated: 2026-04-08
sources: [https://arxiv.org/abs/2601.21912]
status: active
---

# ProRAG: Process-Supervised RL for RAG

## 基本信息
- **作者**: Linwei Wang et al.
- **机构**: Renmin University of China
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2601.21912)
- **链接**: https://arxiv.org/abs/2601.21912
- **代码**: https://github.com/lilinwz/ProRAG

## 一句话总结
> 四阶段 RAG RL 框架：SFT warmup → MCTS-based [[process-reward-model|PRM]] 训练 → PRM-Guided Reasoning Refinement → Process-Supervised RL with dual-granularity advantage，在 multi-hop QA 上大幅超越 Search-R1 等基线。

## 摘要
现有 RAG 系统通过 RL 训练（如 Search-R1）仅使用 outcome-level reward（最终答案是否正确），信号稀疏且无法定位推理链中的具体错误。ProRAG 提出四阶段 process-supervised RL 框架：(1) Format-aware SFT warmup 建立结构化推理格式；(2) 用 MCTS 探索多样推理路径并训练 PRM；(3) PRM-Guided Refinement 解决 cold-start；(4) 用 dual-granularity advantage（outcome + process）做 RL。结构化推理 schema：`<step>→<subquery>→<retrieval>→<subanswer>→<answer>`。

## 核心贡献
1. **四阶段框架**：首次在 RAG 任务上实现完整的 process-supervised RL pipeline
2. **MCTS-based PRM 训练**：用 MCTS 探索多样推理路径，GPT-4o 对 sibling nodes 做 contrastive labeling（96% 与人工一致）
3. **Dual-Granularity Advantage**：outcome advantage + process advantage（PRM）的加权组合，比单一信号更有效
4. **PRM-Guided Refinement**：RL 前先用 PRM 过滤 SFT policy 生成的轨迹做 RFT warmup，解决 cold-start 问题
5. **高数据效率**：仅 1K training queries 接近 peak 性能

## 方法

### 问题定义
Multi-hop QA 需要多步检索和推理。现有 RL-based RAG（如 Search-R1）只用最终答案正确性做 reward，信号太稀疏。能否提供 step-level 的训练信号来指导每个检索和推理步骤？

### 技术方案

#### Stage 1: Supervised Policy Warmup
- 用少量人工标注的结构化推理轨迹做 SFT
- **结构化推理 schema**: `<step>` → `<subquery>` → `<retrieval>` → `<subanswer>` → ... → `<answer>`
- **Format-aware SFT**: 用 λ > 1 的权重强化 control tokens（`<step>`, `<subquery>` 等）的生成概率

$$\mathcal{L}_{SFT} = -\sum_{t} w_t \log P_\theta(y_t | y_{<t}, q, D)$$

其中 $w_t = \lambda$ 如果 $y_t$ 是 control token，否则 $w_t = 1$。

#### Stage 2: MCTS-based PRM Training

**MCTS 探索**：
- 从 SFT policy 出发，用 MCTS 探索多样推理路径
- **PUCT selection**: 平衡 exploration 和 exploitation 选择扩展节点
- **Q-value backpropagation**: $Q(s) = \text{reward} \times \gamma^{T-t}$，decay factor γ 确保靠前步骤的 Q 值反映下游贡献

**Contrastive Labeling**：
- 对 MCTS 树中的 sibling nodes（同一父节点的不同子节点），用 GPT-4o 做对比标注
- 标注一致性：96% 与人工标注一致
- 生成 (good step, bad step) 对比对，训练 PRM

**PRM 训练**：
- 基于 contrastive data 训练 step-level PRM
- PRM 输出每步的质量分数 $R_{step}(q, D, s_t) \in [0, 1]$

#### Stage 3: PRM-Guided Reasoning Refinement (RFT)
RL 前的 cold-start 解决方案：
- 用 SFT policy 生成大量推理轨迹
- **Dual-criterion 过滤**: 只保留 (1) outcome correct 且 (2) 每步 PRM score > 0 的轨迹
- 在过滤后的高质量轨迹上做 RFT（Rejection Fine-Tuning）

#### Stage 4: Process-Supervised RL

**Dual-Granularity Advantage**：

$$A_{i,t,k} = A^{out}_{i,t,k} + \beta \cdot A^{proc}_{i,t,k}$$

其中：
- $A^{out}$ = outcome advantage（基于 F1 score 的组内 z-score 归一化）
- $A^{proc}$ = process advantage（基于 PRM step-level score 的组内 z-score 归一化）
- $\beta$ = 权重超参数

### 关键公式

**RL 目标**（基于 [[grpo|GRPO]]）：

$$\mathcal{L}(\theta) = -\mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i} \min\left(r(\theta) A_{i,t,k}, \; \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A_{i,t,k}\right) \right]$$

**Outcome Reward**:

$$R^{out}(y, y^*) = F1(y, y^*)$$

使用 F1 score 而非 EM（Exact Match），提供更平滑的 reward 信号。

## 实验结果

### Multi-hop QA Benchmarks

| 方法 | PopQA | HotpotQA | 2WikiMQA | MuSiQue | Bamboogle | 平均 |
|------|-------|----------|----------|---------|-----------|------|
| Search-R1 | 56.7 | 54.3 | 49.2 | 47.0 | 48.0 | 51.0 |
| HiPRAG | 58.1 | 55.8 | 51.0 | 49.6 | 50.4 | 53.0 |
| **ProRAG** | **60.3** | **57.6** | **53.8** | **55.3** | **52.8** | **56.0** |

ProRAG 在最复杂的 MuSiQue 上 EM=55.3，显著超过 Search-R1 (47.0) 和 HiPRAG (49.6)。

### 消融实验

| 消融 | 平均变化 |
|------|---------|
| 去掉 PRM stage (Stage 2) | -3.2 |
| 去掉 Refinement stage (Stage 3) | -2.1 |
| 去掉 dual-granularity (只用 outcome) | -1.8 |
| 去掉 format-aware SFT | -1.2 |

PRM stage 贡献最大，验证了 process supervision 在 RAG 中的价值。

### 鲁棒性测试
- **Irrelevant documents**: 加到 30 docs 性能几乎不降，说明 ProRAG 学会了有效过滤无关信息
- **数据效率**: 仅 1K training queries 达到接近 peak 性能（vs 5K-10K for baselines）

### 训练设置
- **Base Model**: Qwen3-8B
- **Retriever**: E5-base
- **Document Corpus**: 2018 Wikipedia dump
- **Retrieval Setting**: top-3 documents

## 关键发现
1. **Process supervision 在 RAG 中特别重要**：multi-hop 推理链中任何一步检索或推理错误都会级联放大，step-level 信号能精确定位问题
2. **MCTS + contrastive labeling 是高效的 PRM 数据生成方法**：96% 一致性说明 GPT-4o 做 sibling comparison 非常可靠
3. **Cold-start 问题不可忽视**：没有 Stage 3 的 RFT warmup，直接 RL 效果差 2.1 points
4. **F1 vs EM 作为 reward**：F1 提供更平滑的信号，比 EM 的 0/1 reward 更利于学习

## 与其他工作的关系
- **基于**: [[grpo|GRPO]] 框架做 RL 优化
- **核心概念**: [[process-reward-model|PRM]]、MCTS
- **对比**:
  - Search-R1 — 仅 outcome-level reward 的 RAG RL，ProRAG 大幅超越
  - HiPRAG — 分层 RAG，ProRAG 通过 process supervision 超越
- **与 [[wiki/papers/rahman-2025-spark|SPARK]] 的关系**：都用合成数据训练 PRM。SPARK 用 step-level self-consistency，ProRAG 用 MCTS + contrastive labeling。SPARK 面向数学推理，ProRAG 面向 multi-hop QA with retrieval
- **与 [[wiki/papers/ghimire-2026-prism|PRISM]] 的关系**：dual-granularity advantage（outcome + process）与 PRISM 的混合 advantage（SC + PRM）思路类似，都是多信号组合
- **与 [[wiki/papers/tan-2026-ctrl-rag|CTRL-RAG]] 的关系**：都是 RAG 任务的 RL 方法。CTRL-RAG 用 contrastive likelihood reward（无需 PRM），ProRAG 用完整的 PRM pipeline。ProRAG 更重但更精细
- **与 [[wiki/papers/royer-2026-mcnig|MCNIG]] 的关系**：ProRAG 的 MCTS-based PRM 训练和 MCNIG 的信息论 PRM 训练是两种不同的自动标注方案，可以互相替换

## 面试相关

- Q: ProRAG 的四个阶段分别做什么？🔴
- A: (1) Format-aware SFT warmup：建立结构化推理格式（<step>→<subquery>→<retrieval>→<subanswer>→<answer>），加权强化 control tokens；(2) MCTS-based PRM 训练：用 MCTS 探索 + GPT-4o contrastive labeling 生成训练数据训练 step-level PRM；(3) PRM-Guided Refinement：用 PRM 过滤高质量轨迹做 RFT，解决 cold-start；(4) Process-Supervised RL：用 dual-granularity advantage（outcome + process）做 GRPO。

- Q: ProRAG 的 dual-granularity advantage 是什么？和 PRISM 的混合 advantage 有什么区别？🔴
- A: A = A^out + β·A^proc，其中 A^out 是 F1-based outcome advantage，A^proc 是 PRM step-level advantage，都做组内 z-score 归一化。PRISM 是 A = γ·Â_SC + Â_PRM（self-certainty + PRM）。区别：ProRAG 的 outcome 是 task-specific F1 metric，PRISM 的 self-certainty 是 internal signal。ProRAG 在 RAG 场景有 ground-truth F1 可用，所以 outcome 信号更可靠。

- Q: 为什么 ProRAG 需要 MCTS 来训练 PRM？不能直接用 SPARK 的方法吗？🟡
- A: SPARK 的方法是从每步重新采样 N 次续写看答案一致性。在 RAG 场景下推理链更长且包含检索步骤，直接重新采样的搜索空间更大。MCTS 通过 PUCT selection 和 Q-value backpropagation 更有效地探索推理空间。且 contrastive labeling（对比 sibling nodes）比 consistency-based labeling 提供更精确的对比信号。

- Q: ProRAG 的 cold-start 问题是什么？怎么解决的？🟡
- A: 直接用 SFT policy 做 RL 时，大部分生成的轨迹质量很低，RL 很难从纯噪声中学习。Stage 3 用 PRM 过滤出"outcome 正确 + 每步 PRM score > 0"的高质量轨迹做 RFT，让 policy 先到达一个较好的起点再做 RL。去掉这步性能下降 2.1 points。

## 个人笔记
> ProRAG 是目前看到的最完整的 RAG RL 框架。四阶段设计很工程化但每一步都有明确作用：SFT 建立格式 → PRM 提供精细信号 → RFT 解决冷启动 → RL 持续优化。
>
> 值得注意的技巧：(1) Format-aware SFT 中加权 control tokens（λ > 1），确保模型严格遵循推理格式；(2) 用 F1 而非 EM 做 outcome reward，提供更平滑的梯度信号；(3) MCTS 中的 Q-value decay factor γ^(T-t) 确保靠前步骤被正确评估。
>
> 与 CTRL-RAG 的对比：ProRAG 是"重炮"方案（需要训练 PRM、MCTS 探索），CTRL-RAG 是"轻量"方案（只需计算 log-likelihood 对比）。在资源充足时 ProRAG 更好，资源有限时 CTRL-RAG 更实用。
