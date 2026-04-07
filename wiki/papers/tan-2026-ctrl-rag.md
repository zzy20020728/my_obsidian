---
title: "CTRL-RAG: Contrastive Likelihood Reward Based Reinforcement Learning for Context-Faithful RAG Models"
type: paper
tags: [URLVR, RAG, contrastive-likelihood, faithfulness, GRPO, internal-reward]
created: 2026-04-08
updated: 2026-04-08
sources: [https://arxiv.org/abs/2603.04406]
status: active
---

# CTRL-RAG: Contrastive Likelihood Reward for RAG

## 基本信息
- **作者**: Jianing Tan et al.
- **机构**: Ant Group
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.04406)
- **链接**: https://arxiv.org/abs/2603.04406

## 一句话总结
> 提出 Contrastive Likelihood Reward (CLR)——对比有/无 supporting documents 时 answer 的 log-likelihood 差异作为 reward，衡量模型对文档的依赖程度，鼓励 context-faithful 推理，无需额外 PRM 训练。

## 摘要
现有 RAG RL 方法主要用 accuracy-based reward（答案正确性），忽略了模型是否真正依赖检索到的文档进行推理。即使答案正确，模型可能依赖参数记忆而非文档证据，导致 faithfulness 不足。CTRL-RAG 提出 Contrastive Likelihood Reward (CLR)：对比有完整文档 vs 移除关键文档后 answer 的 log-likelihood 差异，量化模型对文档的 **evidential contribution**。CLR 鼓励模型生成更依赖文档证据的答案，提升 context-faithfulness。结合 accuracy reward 的 hybrid reward 方案在多个 benchmark 上超越包括 Qwen3-235B-A22B-Instruct 在内的大模型。

## 核心贡献
1. **Contrastive Likelihood Reward (CLR)**：无需额外模型训练的 reward 信号，直接利用 policy model 的 log-likelihood 对比
2. **Evidential Contribution 度量**：leave-one-out 方法量化每个 supporting document 对答案的贡献
3. **Token-level 可解释性**：可视化每个 token 的 evidential contribution，揭示模型学到的推理模式
4. **Reference Reliance 分析框架**：RR_θ = Acc_θ(Q,D) - Acc_θ(Q)，量化模型对文档的依赖度变化
5. **Perplexity 动态分析**：发现训练过程中 PPL(y|D) 稳定但 PPL⁻(y|D) 持续上升的有趣现象

## 方法

### 问题定义
RAG 系统不仅需要答案正确，还需要答案 **faithful to context**——即基于检索文档的证据推理，而非依赖参数记忆。如何构建一个 reward 信号来同时鼓励正确性和忠实性？

### 技术方案

#### 1. Evidential Contribution (E)

定义"有完整文档时的 scoring"：

$$S(y|D) = \sum_{t} \log P_\theta(y_t | y_{<t}, q, D)$$

定义"移除最关键文档后的 scoring"（Leave-One-Out）：

$$S^-(y|D) = \min_{d_i \in D^+} \sum_{t} \log P_\theta(y_t | y_{<t}, q, D \setminus \{d_i\})$$

其中 $D^+$ 是 supporting documents 集合，$\min$ 取最差情况（移除哪个文档影响最大）。

**Evidential Contribution**：

$$E(y) = S(y|D) - S^-(y|D)$$

直觉：如果答案真正基于文档推理，那么移除关键文档后 log-likelihood 应该显著下降。E(y) 越大说明模型越依赖文档。

#### 2. CLR Reward

$$R_{CLR}(y) = \frac{E(y) \cdot \mathbb{1}[E(y) > \tau]}{\sqrt{T}}$$

- **Threshold τ**: 过滤 E(y) 太小的样本（噪声）
- **√T 归一化**: 解决 length bias（长答案天然有更大的 log-likelihood 差异）

#### 3. Hybrid Reward

$$R_{hybrid} = R'_{CLR} \cdot R_{acc}$$

**Multiplicative gating**: 确保错误答案的 reward 为 0（无论 CLR 多高）。比 additive fusion ($R_{CLR} + R_{acc}$) 更好，因为：
- 答案错误 → $R_{acc} = 0$ → $R_{hybrid} = 0$，不会因为高 faithfulness 奖励错误答案
- 答案正确 → $R_{acc} > 0$ → CLR 提供额外的 faithfulness 区分

### 关键公式

基于 [[grpo|GRPO]] 框架，用 $R_{hybrid}$ 作为 reward 计算 advantage：

$$\hat{A}_i = \frac{R_{hybrid}(y_i) - \mu_R}{\sigma_R}$$

其中 μ_R, σ_R 是 group 内的均值和标准差。

**Reference Reliance 度量**：

$$RR_\theta = Acc_\theta(Q, D) - Acc_\theta(Q)$$

$Acc_\theta(Q)$ 是不给任何文档时的准确率，$Acc_\theta(Q, D)$ 是给文档时的准确率。RR 越高说明模型越依赖文档。

## 实验结果

### Multi-hop QA Benchmarks (Qwen3-8B think mode)

| Reward | HotpotQA | 2WikiMQA | MuSiQue | Bamboogle | AVG |
|--------|----------|----------|---------|-----------|-----|
| R_acc | 82.5 | 85.1 | 80.3 | 86.4 | 83.6 |
| R_total (acc + format) | 83.2 | 85.6 | 80.8 | 86.0 | 83.9 |
| **R_hybrid (CLR)** | **84.3** | **86.8** | **82.1** | **86.8** | **85.0** |

### Faithfulness (PRGB Benchmark)

CLR 在 faithfulness 评估上**显著领先**，比所有基线高 >3 points margin。说明 CLR 确实让模型学会了更好地利用文档证据。

### 大模型对比 (Qwen3-30B-A3B MoE)

| 方法 | AVG |
|------|-----|
| Qwen3-235B-A22B-Instruct-2507 | 78.0% |
| **CTRL-RAG (Qwen3-30B-A3B + R_CLR)** | **85.0%** |

30B MoE 模型通过 CTRL-RAG 训练**超过 235B 的 instruct 版本**。

### Reference Reliance 变化
训练后 RR 提升 6%，证明模型确实学会了更好地利用文档。

### Perplexity 动态
- PPL(y|D)：有文档时的困惑度，训练后稳定
- PPL⁻(y|D)：移除关键文档后的困惑度，训练后**持续上升**
- 解读：模型从"依赖参数知识"转向"依赖文档证据"，没有文档就真的不确定了

## 消融实验

### Length Normalization

| 归一化方式 | 效果 |
|-----------|------|
| 无归一化 | verbose collapse（生成越来越长） |
| /T | 学习停滞 |
| **√T** | **最优** |

### LOO 策略

| 策略 | 效果 |
|------|------|
| LOO_avg | 较差 |
| **LOO_min** | **更好** |

LOO_min（移除影响最大的文档）比 LOO_avg（平均所有文档的影响）更有区分力。

### Fusion 方式

| 方式 | 效果 |
|------|------|
| Additive (R_CLR + R_acc) | 较差（可能奖励错误但高 faithfulness 的答案） |
| **Multiplicative (R_CLR × R_acc)** | **更好**（错误答案 reward = 0） |

### Token-level 可解释性分析
CLR 鼓励三类 token：
1. **文档引用标记**：明确引用文档中的事实
2. **跨文档推理连接词**：连接不同文档的证据进行推理
3. **新信息而非重复内容**：避免简单复述文档

## 关键发现
1. **Accuracy ≠ Faithfulness**：答案正确但不依赖文档的情况很常见，需要额外的 faithfulness 信号
2. **CLR 是轻量但有效的 faithfulness reward**：无需训练额外模型，直接用 policy model 的 log-likelihood 对比
3. **Multiplicative gating > Additive fusion**：确保错误答案不被 faithfulness 奖励
4. **√T 归一化是解决 length bias 的关键**：线性归一化过强导致学习停滞，无归一化导致 verbose collapse
5. **Perplexity dynamics 揭示学习机制**：模型确实在从参数记忆转向文档依赖

## 与其他工作的关系
- **基于**: [[grpo|GRPO]] 框架
- **核心概念**: Contrastive Likelihood、Context Faithfulness
- **与 [[wiki/papers/zhang-2025-empo|EMPO]] 的关系**：CLR 和 [[semantic-entropy|semantic entropy]] 都是 internal signals（不需要额外模型），但 CLR 利用外部文档做对比（有/无文档），不是纯内部信号。可以视为 "internal-external hybrid reward"
- **与 [[wiki/papers/ghimire-2026-prism|PRISM]] 的关系**：PRISM 混合 PRM + self-certainty，CTRL-RAG 混合 CLR + accuracy。都是多信号混合思路，但信号类型和任务场景不同
- **与 [[wiki/papers/wang-2026-prorag|ProRAG]] 的关系**：都是 RAG 任务的 RL 方法。ProRAG 用完整的 PRM pipeline（重炮方案），CTRL-RAG 只需 log-likelihood 对比（轻量方案）。ProRAG 提供 step-level 信号，CTRL-RAG 提供 outcome-level 的 faithfulness 信号。两者可以结合使用
- **与 [[wiki/papers/rahman-2025-spark|SPARK]] 的区别**：SPARK 训练 PRM 做 reward，CTRL-RAG 不需要训练额外模型。SPARK 面向数学推理，CTRL-RAG 面向 RAG faithfulness
- **新的研究方向**：CLR 开辟了"利用文档对比做 reward"的新思路，可以扩展到其他有外部证据的任务

## 面试相关

- Q: CTRL-RAG 的 CLR reward 是怎么计算的？🔴
- A: 对比有/无关键文档时 answer 的 log-likelihood 差异。Evidential Contribution E(y) = S(y|D) - S⁻(y|D)，其中 S⁻ 是 leave-one-out（移除影响最大的 supporting document）后的 log-likelihood。CLR = E(y)·1[E>τ]/√T，用 threshold 过滤噪声，√T 归一化解决 length bias。

- Q: 为什么用 multiplicative fusion 而不是 additive？🔴
- A: Multiplicative: R_hybrid = R_CLR × R_acc。如果答案错误（R_acc = 0），无论 CLR 多高 reward 都是 0。Additive 可能奖励"错误但 faithful"的答案，这不合理。Multiplicative 确保正确性是前提，faithfulness 是加分。

- Q: CTRL-RAG 和 ProRAG 的区别？各自适用场景？🔴
- A: CTRL-RAG 轻量——只需计算 log-likelihood 对比，不需要训练额外模型，关注 faithfulness。ProRAG 重量——需要 MCTS 探索 + PRM 训练，四阶段 pipeline，关注推理过程质量。资源充足且需要 step-level supervision 选 ProRAG，资源有限或重点关注 faithfulness 选 CTRL-RAG。两者可以结合。

- Q: CLR 为什么用 √T 归一化而不是其他方式？🟡
- A: 实验发现：无归一化 → verbose collapse（长答案天然 E 值更大，模型学会变长）；/T → 过度惩罚长答案导致学习停滞；√T 是折中，既避免 length bias 又不过度抑制。

- Q: Reference Reliance 怎么衡量？训练后有什么变化？🟡
- A: RR = Acc(Q,D) - Acc(Q)，即有文档 vs 无文档的准确率差。训练后 RR 提升 6%，说明模型更依赖文档。配合 Perplexity 动态（PPL⁻持续上升）进一步验证。

## 个人笔记
> CTRL-RAG 最大的亮点是 **轻量级 faithfulness reward**。不需要训练 PRM，不需要 MCTS，只需要跑两遍 forward（有/无文档），就能获得高质量的 faithfulness 信号。这在工程上非常友好。
>
> 三个有趣的分析值得深入理解：(1) Token-level 可解释性——可以看到 CLR 具体鼓励了哪些 token；(2) Perplexity dynamics——PPL⁻ 持续上升意味着模型真的在"忘记"参数知识、"学习"文档依赖；(3) Reference Reliance 变化——量化了模型行为的转变。
>
> 与 EMPO 的有趣对比：EMPO 的 semantic entropy 是"模型内部一致性"信号，CLR 是"模型对外部证据依赖度"信号。两者都不需要额外模型，但捕捉的是完全不同的维度。可以设想把两者结合：用 semantic entropy 衡量推理一致性 + CLR 衡量文档依赖度。
>
> LOO_min vs LOO_avg 的选择很有意思：min 是"最坏情况"思维，找模型最依赖的那个文档来评估。这比平均更有区分力，因为不是所有文档都是关键文档。
