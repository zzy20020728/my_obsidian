---
title: Contrastive Likelihood (对比似然)
type: concept
tags: [RAG, faithfulness, reward-signal, contrastive, URLVR]
created: 2026-04-08
updated: 2026-04-08
sources: [wiki/papers/tan-2026-ctrl-rag.md]
status: active
---

# Contrastive Likelihood (对比似然)

## 定义
> Contrastive Likelihood，对比似然。通过对比不同条件下模型生成某个输出的 log-likelihood 差异来衡量特定条件的"贡献度"。在 URLVR 中，[[wiki/papers/tan-2026-ctrl-rag|CTRL-RAG]] 将其用于 RAG 场景：对比有/无 supporting documents 时 answer 的 log-likelihood，量化模型对文档的依赖程度（evidential contribution），构建 context-faithfulness reward。

## 关键性质
1. **无需额外模型训练**：直接利用 policy model 自身的 log-likelihood，计算成本低
2. **衡量"依赖度"而非"正确性"**：不判断答案对不对，而是判断答案是否基于文档推理
3. **Leave-One-Out 策略**：移除最关键的文档后观察 log-likelihood 变化，更有区分力
4. **需要 √T 归一化**：长答案天然有更大的 log-likelihood 差异，需要长度归一化

## 直觉理解
> 你写了一篇论文引用了三篇参考文献。如何判断你是否"真的参考了"这些文献？把某篇参考文献拿走——如果你的论文内容大幅改变，说明你确实依赖这篇文献（高 evidential contribution）；如果论文几乎不变，说明你只是形式上引用，实际靠自己的知识写的（低 evidential contribution）。CTRL-RAG 对 LLM 做同样的事：拿走关键文档后看答案的 log-likelihood 变化。

## 数学表达

### Evidential Contribution (E)

$$S(y|D) = \sum_{t} \log P_\theta(y_t | y_{<t}, q, D)$$

$$S^-(y|D) = \min_{d_i \in D^+} \sum_{t} \log P_\theta(y_t | y_{<t}, q, D \setminus \{d_i\})$$

$$E(y) = S(y|D) - S^-(y|D)$$

- $D$ = 全部检索文档
- $D^+$ = supporting documents 集合
- $\min$ = leave-one-out 中取影响最大的文档

### CLR Reward

$$R_{CLR}(y) = \frac{E(y) \cdot \mathbb{1}[E(y) > \tau]}{\sqrt{T}}$$

- $\tau$: threshold，过滤 noise
- $\sqrt{T}$: 长度归一化

### Hybrid Reward

$$R_{hybrid} = R'_{CLR} \cdot R_{acc}$$

Multiplicative gating：错误答案 reward = 0。

## CLR vs 其他 Reward Signals

| 维度 | CLR | Semantic Entropy | PRM |
|------|-----|-----------------|-----|
| 类型 | Internal-External Hybrid | Pure Internal | External |
| 需要额外模型 | 否 | 否 | 是 |
| 衡量维度 | Faithfulness | Consistency | Step Quality |
| 适用任务 | RAG | 通用推理 | 通用推理 |
| 计算成本 | 2× forward | G× forward | 1× PRM forward |
| Length Bias | 需要 √T 归一化 | 无（基于聚类频率） | 无 |

## 相关论文
- [[wiki/papers/tan-2026-ctrl-rag|CTRL-RAG (Tan et al., 2026)]] — 提出 CLR 用于 RAG faithfulness
- [[wiki/papers/wang-2026-prorag|ProRAG (Wang et al., 2026)]] — RAG RL 的 PRM-based 替代方案
- [[wiki/papers/zhang-2025-empo|EMPO (Zhang et al., 2025)]] — 另一种无需额外模型的 reward（semantic entropy）

## 面试常问点

- 🔴 Q: CLR 是怎么衡量 faithfulness 的？
  - A: 对比有/无关键文档时 answer 的 log-likelihood 差异。用 Leave-One-Out (LOO_min) 移除影响最大的 supporting document，E(y) = S(y|D) - S⁻(y|D)。E 越大说明答案越依赖文档。配合 √T 归一化和 threshold 过滤噪声。

- 🟡 Q: CLR 为什么用 multiplicative fusion 而不是 additive？
  - A: Multiplicative: R = CLR × Acc。答案错误时 Acc=0 → R=0，不会因为高 faithfulness 奖励错误答案。Additive 会出现"错误但 faithful"的答案得到正 reward 的问题。

- 🟡 Q: CLR 和 PRM 各自什么时候更适用？
  - A: CLR 适合 RAG 场景（有外部文档做对比），轻量且关注 faithfulness。PRM 适合通用推理（提供 step-level quality 信号），更重但信号更精细。两者可以结合。

## 与其他概念的关系
- 上位概念：Reward Signal、Contrastive Learning
- 对比概念：[[semantic-entropy|Semantic Entropy]]（另一种无需额外模型的 reward）、[[process-reward-model|PRM]]（外部模型 reward）
- 应用概念：[[grpo|GRPO]]（CLR 做 reward 的 RL 框架）
- 风险：[[reward-hacking|Reward Hacking]]（verbose collapse if no length normalization）
