---
title: "Step-Level Semantic Entropy: 无监督步骤级评分研究方案"
type: synthesis
tags: [semantic-entropy, step-level, URLVR, SPAE, SEP, TTRL, process-reward, research-proposal, reward-hacking]
created: 2026-04-08
updated: 2026-04-08
sources: [wiki/papers/wu-2026-spae.md, wiki/papers/zhang-2025-empo.md, wiki/papers/zhang-2026-grad2reward.md, wiki/papers/royer-2026-mcnig.md, wiki/papers/ghimire-2026-prism.md, wiki/concepts/semantic-entropy.md, wiki/concepts/process-reward-model.md, wiki/concepts/reward-hacking.md]
status: active
---

# Step-Level Semantic Entropy: 无监督步骤级评分研究方案

## 一句话总结

> 将 [[wiki/papers/wu-2026-spae|SPAE]] 的 training-free step-level probing 框架与 [[wiki/concepts/semantic-entropy|Semantic Entropy]] 结合，用语义熵替代需要 GT 的 Correctness 信号，实现**完全无监督的步骤级 credit assignment**。

## 研究动机

### 现有方法的局限

| 方法 | Step-Level? | 无监督? | 问题 |
|------|:---------:|:------:|------|
| [[wiki/papers/zhang-2025-empo|EMPO]] | 否（outcome-level） | 是 | 无法区分关键步骤与冗余步骤 |
| [[wiki/papers/wu-2026-spae|SPAE]] | 是 | **否**（需要 GT 算 Correctness） | 不适用于 URLVR 场景 |
| [[wiki/papers/rahman-2025-spark|SPARK]] | 是（PRM） | 部分（self-consistency 生成标注） | 需要训练 PRM |
| [[wiki/papers/royer-2026-mcnig|MCNIG]] | 是（PRM） | 否（需要 GT） | 需要训练 PRM |
| [[wiki/papers/zhang-2026-grad2reward|Grad2Reward]] | 是（token-level） | 是（self-judging） | 面向 open-ended 任务，数学推理场景未验证 |

**核心空白**: 没有一个方法同时满足「step-level」+「training-free」+「完全无监督」。

### 文献空白确认

穷举搜索 arXiv / Semantic Scholar / Google Scholar（2026-04-08），**没有找到任何论文将 Kuhn et al. 的 semantic entropy 应用到每个推理步骤并作为 process reward 信号**。

最接近的工作：
- Wang et al. (2025, 2511.06168)：step-level SE matrix 用于**评估** CoT 质量，非 reward
- ROSE (Zhao et al. 2026, 2601.05053)：SE 选择 MCTS branching 点，非 reward
- CoFiCot (Zhang et al. 2026, 2603.08251)：SE 做 query-level 分流 + PRM 做 step-level，两者分离
- EDU-PRM (Cao et al. 2025, 2503.22233)：token-level **predictive** entropy（非语义熵）做步骤分割
- SEED-GRPO (Chen et al. 2025, 2505.12346)：question-level SE 调节 GRPO 更新幅度

## 核心Idea

### SPAE 的有监督瓶颈

SPAE 的 Step Potential 由两个信号合成：

$$\Phi(\tau_i^k) = 1.5 \cdot \text{Acc}(\tau_i^k) \cdot \text{Conf}(\tau_i^k) + 0.5 \cdot \text{Acc}(\tau_i^k) - \text{Conf}(\tau_i^k)$$

| 信号 | 计算方式 | 需要 GT? |
|------|---------|---------|
| **Confidence** | probe 续写的 token entropy 负指数均值 | 否 |
| **Correctness (Acc)** | force-feed GT tokens 的条件概率均值 | **是** — 有监督瓶颈 |

### 用 Semantic Certainty 替代 Correctness

**核心观察**：Correctness 衡量"离正确答案多近"，Semantic Entropy 衡量"对最终答案多确定"。两者不等价但高度相关——模型收敛到正确答案时 SE↓ Acc↑；模型走偏时 SE↑ Acc↓。

定义 **Semantic Certainty (SC)**：

$$SC(\tau_i^k) = 1 - \frac{SE(\tau_i^k)}{SE_{\max}}$$

其中 $SE(\tau_i^k)$ 为该步 probe 续写的语义熵，$SE_{\max}$ 为归一化常数。SC ∈ [0, 1]。

**无监督 Step Potential**：

$$\Phi_{\text{unsup}}(\tau_i^k) = 1.5 \cdot SC(\tau_i^k) \cdot \text{Conf}(\tau_i^k) + 0.5 \cdot SC(\tau_i^k) - \text{Conf}(\tau_i^k)$$

三种推理状态完美保留：

| 状态 | SC | Conf | Phi_unsup | 含义 |
|------|-----|------|-----------|------|
| 探索中 | 低（续写语义分散） | 低 | ≈ 0 | 尚未收敛 |
| 正确收敛 | 高（续写语义一致） | 高 | → +1 | 高概率正确 |
| 自信幻觉 | 低（续写语义分散） | 高 | → -1 | 自信但答案不确定 |

## 关键设计决策

### SE 计算：复用 SPAE 的 probe 续写

**SPAE 已经在每步采样 N=5 条短续写（各 10 tokens）**。不需要额外采样：

1. 取这 5 条 probe 续写
2. 提取每条续写的 semantic embedding（用模型自身的 last hidden state 或外部小模型）
3. 基于 embedding cosine similarity 做语义聚类（阈值聚类或 NLI）
4. 计算离散 semantic entropy：$SE = -\sum_c \frac{|C_c|}{N} \log \frac{|C_c|}{N}$

**额外成本 ≈ 零**：仅多一次 embedding 提取 + 聚类计算。

### 续写长度的 trade-off

| probe 续写长度 | SE 估计质量 | 计算成本 | 建议 |
|--------------|:----------:|:-------:|------|
| 10 tokens（SPAE 默认） | 可能不够区分语义 | 最低 | 快速验证 |
| 50 tokens | 中等 | 5x probe 成本 | 推荐起步 |
| 100 tokens | 较好 | 10x probe 成本 | 精度需求高时 |
| 完整续写 | 最好 | O(T×K×L)，爆炸 | 不推荐 |

### 语义等价判断方式

| 方法 | 优点 | 缺点 | 推荐度 |
|------|------|------|:------:|
| **Embedding cosine similarity** | 快速，无需额外模型 | 阈值敏感 | 推荐 |
| NLI 模型（MNLI） | Kuhn et al. 原始方案，理论最准 | 需要额外模型推理 | 备选 |
| 模型自身 hidden state 距离 | 零额外开销 | 可能不够准 | 快速实验 |

## 计算成本对比

| 方案 | 每步额外成本 | 总额外成本（T步） | 相对于基线 RLVR |
|------|-----------|:----------------:|:---------------:|
| SPAE（原始） | N=5 × 10 tokens + 1 GT force-feed | T × 55 tokens | +~15% |
| **SPAE-SE（方案A）** | N=5 × 10 tokens + embedding | T × 50 tokens + 聚类 | +~12% |
| **SPAE-SE-Long（方案B）** | N=5 × 50 tokens + embedding | T × 250 tokens + 聚类 | +~50% |
| **SPAE-SEP（方案C）** | 1 次线性预测 | T 次线性预测 | +~0.1% |
| 全量 step-level SE（朴素） | K=10 × 完整续写 | T × K × L tokens | +1000%+ |

方案 A 的成本甚至**低于原始 SPAE**（省去了 GT force-feed 步骤）。

## SEP 探针方案（进阶优化）

### Semantic Entropy Probes (SEPs) 原理

来源：Kossen et al. 2024 (arXiv:2406.15927)

核心思想：模型的 hidden state 中已经编码了语义不确定性信息。训练一个线性探针直接从 hidden state 预测 SE，**完全避免采样**。

### 三阶段部署

**Phase 1: 数据收集**
- 选一批问题（~5K-10K）
- 跑 SPAE-SE 方案（N=5 × 50 tokens）
- 记录每步边界的 hidden state 和真实 SE
- 得到 $(h_k, SE_k)$ pairs

**Phase 2: 训练 SEP**
- 训练线性探针：$\hat{SE}_k = w^\top h_k + b$
- 简单 logistic regression / 小 MLP（<100 参数）
- 验证 AUROC / 相关系数

**Phase 3: RL 训练部署**
- 每步边界读取 hidden state
- SEP 预测 $\hat{SE}_k$ → 计算 $SC_k = 1 - \hat{SE}_k / SE_{\max}$
- Confidence 用 token entropy（也无需采样）
- **总额外成本 ≈ 每步一次矩阵乘法**

### 需验证的假设
1. Step-level hidden state 是否包含足够的语义不确定性信息（vs answer-level）
2. 线性探针在不同问题类型上的迁移性
3. SEP 预测误差是否导致 reward 信号退化

## 实验设计建议

### Phase 1: 概念验证（~1周）

**目标**: 验证 SC 能否替代 Correctness

1. 选 SPAE 论文的 base model（DeepSeek-R1-Distill-Qwen-7B）
2. 跑 1000 个数学问题的 probe
3. 同时计算 Acc（原始 SPAE）和 SC（我们的方案）
4. 分析 Acc vs SC 的 Pearson/Spearman 相关系数
5. **关键指标**: r > 0.7 说明替代可行

### Phase 2: RL 训练（~1-2周）

**目标**: 完整 RLVR 训练对比

| 实验组 | Correctness 来源 | 需要 GT? |
|--------|-----------------|---------|
| SPAE（原始） | Force-feed GT | 是 |
| SPAE-SE（方案A） | 10 tokens probe SE | 否 |
| SPAE-SE-Long（方案B） | 50 tokens probe SE | 否 |
| Baseline (DAPO/RF-B) | N/A（outcome-level） | 是 |
| EMPO | N/A（outcome-level SE） | 否 |

评估：AIME2024/2025, AMC23, MATH500 的 Acc 和 Avg Len

### Phase 3: SEP 优化（~1周）

**目标**: 验证探针加速可行性

1. 用 Phase 2 的数据训练 SEP
2. 对比 SEP 预测 vs 真实 SE 的相关性
3. 用 SEP 替代真实 SE 做 RL 训练
4. 对比 SPAE-SE vs SPAE-SEP 的性能差距

## 论文 Story（Motivation 逻辑链）

### 为什么这么设计？——五步逻辑链

**Step 1**: RLVR 的 sparse outcome reward 是根本瓶颈。100 步推理链中所有 token 共享 +1/-1，模型无法区分第 37 步的关键突破和第 85 步的冗余验证。SPAE 量化了后果：8% R2W 失败率。

**Step 2**: 现有 step-level 方案都需要外部监督。PRM（SPARK/MCNIG/ProRAG）需要 step-level 标注或 GT；SPAE 的 Correctness 需要 force-feed GT。无法扩展到无标注场景。

**Step 3（关键洞察）**: SPAE 的 probe 机制已经藏着一个被忽视的无监督信号。5 条 probe 续写除了算 Confidence（token entropy），还包含**语义一致性信息**。如果 5 条续写都说同一件事 → SE 低 → 模型已收敛。这是一个 **免费的、被丢弃的** 无监督 correctness proxy。

**Step 4**: Semantic Certainty 在理论上是合理的 step-level reward。EMPO 证明 answer-level SE 与 accuracy 强相关（|r| > 0.8）。信息论角度：好步骤应该降低后续答案的语义不确定性（Information Gain）。Ng 1999 的 potential-based reward shaping 保证不改变最优策略。

**Step 5**: 纯 step-level 信号不够，需要 outcome anchor。TTRL 的 majority voting 提供完全无监督的 outcome signal。双层架构：TTRL 判断对错，SPAE-SE 分配功劳。全程只需 {q}。

### 一句话 Story

> SPAE 的 probing 机制中隐藏着一个被忽视的无监督信号——probe 续写的语义一致性。我们将其形式化为 Semantic Certainty，结合 TTRL 的 majority voting 做 outcome anchor，首次实现完全无监督的双层 step-level credit assignment。

### Positioning

- vs EMPO: 我们是 step-level（更精细），EMPO 是 outcome-level
- vs SPAE: 我们不需要 GT（URLVR），SPAE 需要
- vs SPARK/MCNIG: 我们不需要训练 PRM（training-free）
- vs Grad2Reward: 我们面向可验证推理任务，Grad2Reward 面向 open-ended
- vs 纯 TTRL: 我们提供 step-level credit assignment，TTRL 只有 outcome-level
- vs TTRL follow-ups (SPINE/DARE): 我们用语义层面的信号（SE），它们用 token-level 信号

### 潜在 Title

- *"SPAE-SE: Unsupervised Step-Level Credit Assignment via Semantic Entropy Probing"*
- *"From Outcome to Process: Bridging TTRL and Step-Level Rewards without Supervision"*

## 实验设计

### Phase 1: 概念验证（~1 周）

**目标**: 验证 SC 能否替代 Correctness

1. 选 SPAE 的 base model（DeepSeek-R1-Distill-Qwen-7B）
2. 跑 1000 个数学问题的 probe（N=5, 50 tokens）
3. 同时计算 Acc（原始 SPAE，需要 GT）和 SC（我们的方案）
4. 分析 Acc vs SC 的 Pearson/Spearman 相关系数
5. **关键指标**: r > 0.7 说明替代可行

### Phase 2: RL 训练对比（~1-2 周）

**目标**: 完整 RLVR 训练对比

| 实验组 | Outcome Signal | Step Signal | 需要 GT? |
|--------|---------------|-------------|---------|
| SPAE（上界） | GT binary | Correctness probe | 是 |
| **TTRL + SPAE-SE（我们的）** | Majority voting | SC probe | **否** |
| 纯 TTRL | Majority voting | 无 | 否 |
| EMPO | Semantic entropy (outcome) | 无 | 否 |
| DAPO/RF-B（基线） | GT binary | 无 | 是 |

评估：AIME2024/2025, AMC23, MATH500 的 Acc@16 和 Avg Len@16

### Phase 3: 稳定性验证（关键）

**目标**: 证明不会 reward hack

1. 跑 >500 步训练曲线
2. 监控：SC vs real accuracy 的 moving correlation（PRISM 的方法）
3. 监控：输出长度变化（是否 premature convergence）
4. 监控：R2W 率变化
5. **成功标准**: moving correlation 持续 > 0.3；accuracy 不出现 rise-then-fall

### Phase 4: SEP 加速（~1 周）

1. 用 Phase 2 数据训练线性探针
2. 对比 SEP 预测 vs 真实 SE 的 AUROC
3. 用 SEP 替代真实 SE 做 RL 训练
4. 对比 SPAE-SE vs SPAE-SEP 性能差距

### 实验需要证明的五件事

1. **SC 和 Correctness 的相关性** → 合理性
2. **SPAE-SE vs SPAE 的 accuracy 差距** → 无监督 vs 有监督 upper bound
3. **TTRL + SPAE-SE vs 纯 TTRL** → step-level credit assignment 的增量价值
4. **长期训练不崩溃** → reward hacking 防御有效
5. **(bonus) SEP 近似效果** → 实用性

## 风险与缓解

### Reward Hacking 详细分析

SE 作为内在奖励**一定会面临 reward hacking**。核心问题：SC 高不代表答案正确。

#### Hack 1: "一致地错"（Confident but Wrong Consensus）

模型在某步后 5 条续写都收敛到同一个错误答案 → SC 很高但答案错。

- **严重度**: 中高
- **PRISM 证据**: 纯内部信号 300 步后崩溃（但 PRISM 测试的是 token entropy / self-certainty，**未测试 SE**）
- **缓解**: TTRL 的 outcome anchor（下节详述）；Entropy Thresholding；KL constraint

#### Hack 2: "提前终止"（Premature Convergence）

模型学会在早期步骤就让 SE 快速下降（跳到简单答案），Saturation Penalty 过早触发，推理链被截断。

- **严重度**: 中
- **对比**: 类似 LC-R1 的无差别截断导致 accuracy 下降
- **缓解**: ε_sat 不能太松；outcome reward 作为 anchor 惩罚错误答案

#### Hack 3: "语义坍缩"（Semantic Collapse）

模型学会一种"万能续写模式"，无论哪步后面都生成相似续写 → SE 在所有步骤都很低，SC 失去区分度。

- **严重度**: 低-中
- **对比**: 类似 self-consistency collapse（SPARK 发现）
- **缓解**: KL 散度约束限制 policy 偏离；Dynamic Sampling 过滤 reward 方差为零的样本

#### SE vs PRISM 测试的三种信号——为什么 SE 可能更鲁棒？

| 信号 | PRISM 测试结果 | SE 的区别 |
|------|:-------------:|---------|
| Token entropy | 崩溃 | SE 在**语义空间**操作，不是单 token 分布 |
| Trajectory entropy | 崩溃 | SE 用**语义聚类**而非字符串匹配 |
| Self-certainty | 崩溃 | SE 是**多条独立采样的 consensus**，不是单条路径的自评 |

**SE 的本质是 cross-sample consensus**——不是模型自己说"我觉得对"，而是"从不同角度算都得到同一个意思"。PRISM 未测试此类信号。但这不意味着 SE 不会被 hack——只是 hack 的方式不同，需要实验验证。

### 综合缓解策略

1. **双层 reward 结构**（TTRL outcome + SPAE-SE step）— 最重要
2. **Entropy Thresholding**（EMPO 方案）：过滤 SE 极端的问题
3. **KL constraint**：限制 policy 偏离 reference model
4. **长期训练曲线监控**：>300 步，持续检查 SC vs real accuracy 的 moving correlation
5. **Bounded signals**：SC ∈ [0,1]，Step Potential ∈ [-1,1]，天然 bounded

## 双层无监督架构（TTRL + SPAE-SE）

### 为什么纯 step-level SC 不够？

Step-level SC 只告诉"这一步让模型更确定了"，不告诉"最终答案是对的"。一条链可能每步 SC 都高（每步都让模型更确定），但第一步就走偏了，后面都在"确定地"沿错误方向推进。

**需要 outcome-level anchor 做最终对错判断。Step-level SC 只做 credit assignment（分配功劳），不替代 outcome signal。**

### TTRL (Test-Time Reinforcement Learning) 作为 outcome anchor

- **TTRL**（Zuo et al. 2025, arXiv:2504.16084）：用 majority voting 做无监督 pseudo-reward
- 对每个问题采样 G 条回答，majority answer 为 pseudo-label
- $R_i^{TTRL} = \mathbb{I}[\text{answer}_i = \text{majority answer}]$

### 双层架构

```
┌──────────────────────────────────────┐
│  Layer 1: Response-Level (TTRL)      │
│  majority voting → R_i^{TTRL}       │
│  → Group Advantage Â_i^{Group}      │
│  作用：判断最终答案对错              │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│  Layer 2: Step-Level (SPAE-SE)       │
│  probe → SE → SC → Step Potential Φ  │
│  Saturation Penalty f(Φ)             │
│  Difference Shaping g(ΔΦ)            │
│  作用：在"正确"链中区分关键/冗余步骤 │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│  Token-Level Advantage               │
│  Â_ij = Â_i^{Group}·f(Φ) + ξ·g(ΔΦ) │
│  → GRPO/DAPO policy update           │
│  全程无监督：仅需 {q}               │
└──────────────────────────────────────┘
```

### Outcome anchor 如何解决 reward hacking

| Hack 类型 | 纯 SC 的问题 | TTRL anchor 如何缓解 |
|----------|------------|---------------------|
| 一致地错 | SC 高但答案错 | R_i^{TTRL}=0 → Â_i^{Group} 为负 → 整条链被惩罚 |
| 提前终止 | 推理链太短导致准确率下降 | Majority voting 的准确率下降 → 自动惩罚 |
| 语义坍缩 | SC 失去区分度 | 不影响 outcome reward，退化为纯 TTRL（保底） |

### 与 TTRL follow-ups 的关系

TTRL 已有 16+ 篇 follow-up（截至 2026-03），都在解决 TTRL 的各种问题：

| 方法 | 改进点 | 与我们的关系 |
|------|-------|------------|
| SPINE (2511.17938) | token-selective updates + entropy regularization | 思路类似但用 token entropy，我们用 step-level SE |
| DARE (2601.21804) | distribution-aware reward | 互补——DARE 改进 outcome reward，我们改进 credit assignment |
| SCRL (2603.19880) | negative pseudo-labeling + entropy gating | 互补——SCRL 解决 hard 问题上的 failure |
| ETTRL (2508.11356) | entropy-based exploration-exploitation | question-level entropy，我们是 step-level |
| Self-Harmony (2511.01191, ICLR 2026) | paraphrase-based stability | 缓解 majority voting 偏差 |

**我们的定位**: 不是 TTRL 的竞争者，而是 TTRL 的 **step-level credit assignment 补充模块**。TTRL 给 outcome signal，我们给 step-level signal。

## 理论合理性论证

### 论证 1: Information-Theoretic Perspective

推理过程是一个信息通道。给定问题 q，推理链 $\tau^1, \tau^2, ..., \tau^K$ 逐步提供信息：

$$H_{semantic}(A | q, \tau^{1:k}) \leq H_{semantic}(A | q, \tau^{1:k-1})$$

"好"步骤应**显著降低最终答案的语义不确定性**。SC 正是度量这一点。

与 [[wiki/papers/royer-2026-mcnig|MCNIG]] 的 Information Gain 在思想上一致，但 SE 在语义层面操作（对 paraphrase 鲁棒），MCNIG 在 token 层面。

### 论证 2: Potential-Based Reward Shaping (Ng et al., 1999)

SC 满足势能函数性质：
- 是状态（推理前缀）的函数
- 单调反映"离目标的距离"（SE 越低 = 离确定答案越近）
- Difference Shaping $g(\Delta\Phi)$ 直接利用势能差做 reward

Ng 1999 证明：基于势能函数的 reward shaping 不改变最优策略 → SC-based shaping 理论上安全。

### 论证 3: EMPO 的经验证据推广

EMPO 在 answer-level 证明 SE 与 accuracy 的 |Pearson r| > 0.8。我们假设 step-level 也成立——**这正是 Phase 1 实验要验证的**。如果 step-level r > 0.7，直接支撑 SC 作为 Acc proxy。

### SE vs 其他内部信号——为什么 SE 更合适？

| 属性 | Token Entropy | Self-Certainty | **Semantic Entropy** |
|------|:------------:|:-------------:|:-------------------:|
| 操作空间 | Token distribution | Single output log-prob | **Semantic equivalence classes** |
| Cross-sample? | 否 | 否 | **是（N 条独立采样）** |
| 对 paraphrase 鲁棒? | 否 | 否 | **是** |
| PRISM 测试过? | 是（崩溃） | 是（崩溃） | **否（未测试）** |
| 计算方式 | 单次前向传播 | 单次前向传播 | N 次采样 + 聚类 |

## 相关论文全景

### 核心参考
- [[wiki/papers/wu-2026-spae|SPAE (Wu et al., 2026)]] — 我们的基础框架
- [[wiki/papers/zhang-2025-empo|EMPO (Zhang et al., 2025)]] — outcome-level semantic entropy
- Kuhn et al. (2023, 2302.09664) — Semantic Entropy 原始论文
- Kossen et al. (2024, 2406.15927) — Semantic Entropy Probes (SEPs)

### 计算效率参考
- Sun et al. (2026, 2603.22812) — 自适应贝叶斯 SE 估计，~50% 样本减少
- McCabe et al. (2025, 2509.14478) — 少样本 SE 估计校正

### 步骤级评分参考
- [[wiki/papers/zhang-2026-grad2reward|Grad2Reward (Zhang et al., 2026)]] — gradient attribution，token-level
- [[wiki/papers/royer-2026-mcnig|MCNIG (Royer et al., 2026)]] — information gain，step-level
- [[wiki/papers/rahman-2025-spark|SPARK (Rahman et al., 2025)]] — self-consistency → PRM

### 相关但不同的工作
- Wang et al. (2025, 2511.06168) — step-level SE 用于评估（非 reward）
- ROSE (Zhao et al. 2026, 2601.05053) — SE 用于 MCTS branching（非 reward）
- EDU-PRM (Cao et al. 2025, 2503.22233) — token entropy 做步骤分割（非语义熵）
