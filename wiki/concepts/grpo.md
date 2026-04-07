---
title: GRPO (Group Relative Policy Optimization)
type: concept
tags: [RL, policy-gradient, LLM, DeepSeek, URLVR]
created: 2026-04-07
updated: 2026-04-07
sources: [wiki/papers/zhang-2025-empo.md, wiki/papers/rahman-2025-spark.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/wu-2026-self-judge.md]
status: active
---

# GRPO (Group Relative Policy Optimization)

## 定义
> Group Relative Policy Optimization，组相对策略优化。DeepSeek 团队提出的 RL 算法，是 [[ppo|PPO]] 的简化替代方案。核心改进：**去掉 value network（critic），用组内相对 reward 排名替代 advantage 估计**。在所有四篇 URLVR 论文（EMPO、SPARK、PRISM、Self-Judge）中都作为基础优化框架使用。

## 关键性质
1. **无需 Value Network**: PPO 需要训练一个 critic 网络来估计 V(s)，GRPO 完全不需要
2. **组内相对排名**: 对每个 prompt 采样一组输出，reward 在组内做归一化（零均值单位方差）
3. **计算更高效**: 少了一个 critic 网络，内存和计算成本降低约 40%
4. **天然适合 LLM**: LLM 场景中状态空间巨大，训练准确的 value function 极其困难

## 直觉理解
> PPO 像考试+排名：先有一个"预期分数"（value function），然后看你是超出预期还是不及预期。训练这个"预期"本身就很难。GRPO 说：不需要预期分数，直接让同一道题的多个答案互相比较——谁比组内平均分高就奖励谁。同学之间横向对比，不需要绝对标准。

## 数学表达

### GRPO 目标函数

$$\mathcal{L}_{GRPO}(\theta) = -\mathbb{E}_{q \sim D, \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left(r_t^i(\theta) \hat{A}_i, \; \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right) \right]$$

其中：
- $r_t^i(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}$ 是概率比
- $\hat{A}_i$ 是组内归一化的 advantage
- $\beta$ 是 KL 惩罚系数
- $G$ 是每个 prompt 的采样组大小

### Advantage 计算（核心区别）

**PPO**: $\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$（需要 value function 估计 TD error）

**GRPO**: 
$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^{G})}{\text{std}(\{R_j\}_{j=1}^{G})}$$

直接用组内 reward 的 z-score 归一化，无需 value function。

### KL 惩罚（Per-Token 形式）

$$D_{KL} = \frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - \log\frac{\pi_{ref}(o_{i,t}|q, o_{i,<t})}{\pi_\theta(o_{i,t}|q, o_{i,<t})} - 1$$

## GRPO vs PPO

| 维度 | PPO | GRPO |
|------|-----|------|
| Value Network | 需要（critic） | **不需要** |
| Advantage 估计 | GAE (需要 V(s)) | **组内 z-score** |
| 内存开销 | 高（policy + critic） | **低（只有 policy）** |
| 适用场景 | 通用 RL | **特别适合 LLM** |
| 超参数 | 较多（GAE λ, critic lr 等） | **较少** |
| 训练稳定性 | 依赖 critic 质量 | **更稳定（无 critic bias）** |

## 在 URLVR 中的使用

所有四篇论文都基于 GRPO 框架，只是 reward 信号不同：

| 论文 | Reward 来源 | 组大小 G |
|------|-------------|----------|
| [[wiki/papers/zhang-2025-empo\|EMPO]] | 语义聚类频率 | — |
| [[wiki/papers/rahman-2025-spark\|SPARK]] | PRM-CoT 评分 | — |
| [[wiki/papers/ghimire-2026-prism\|PRISM]] | γ·SC + PRM | — |
| [[wiki/papers/wu-2026-self-judge\|Self-Judge]] | SC·Judge modulation | n |

GRPO 的优势在 URLVR 中特别明显：
1. 不需要 value function → 少一个需要训练的模型 → 更简单
2. 组内相对排名 → 天然适配 "采样多个输出比较" 的 URLVR 范式
3. Reward 可以是任意信号（不需要标准化的绝对分数）→ 适配各种无监督 reward

## Self-Judge 的改进：Energy-Based Baseline

标准 GRPO 用组内均值做 baseline，[[wiki/papers/wu-2026-self-judge|Self-Judge]] 提出用 log-sum-exp baseline：

$$A_k = \log q_\alpha(\tau_k|x) = \alpha R_k - \log\sum_{j=1}^{n}\exp(\alpha R_j)$$

理论上更好地匹配 energy-based target distribution。

## 相关论文
- Shao et al., 2024 — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"（GRPO 首次提出）
- [[wiki/papers/zhang-2025-empo|EMPO (Zhang et al., 2025)]] — 基于 GRPO
- [[wiki/papers/rahman-2025-spark|SPARK (Rahman et al., 2025)]] — 基于 GRPO
- [[wiki/papers/ghimire-2026-prism|PRISM (Ghimire et al., 2026)]] — 基于 GRPO
- [[wiki/papers/wu-2026-self-judge|Self-Judge (Wu et al., 2026)]] — 改进 GRPO 的 baseline

## 面试常问点

- 🔴 Q: GRPO 和 PPO 的核心区别是什么？
  - A: GRPO 去掉了 value network（critic），用组内 reward 的 z-score 归一化替代 advantage 估计。PPO 需要训练 critic 来估计 V(s)（昂贵且困难），GRPO 直接用同一 prompt 下多个输出的 reward 互相比较。

- 🔴 Q: 为什么 GRPO 特别适合 LLM 场景？
  - A: 三个原因：(1) LLM 的状态空间巨大，训练准确的 value function 极难；(2) LLM 场景天然需要采样多个输出（用于质量对比），GRPO 的组内排名天然利用这些采样；(3) 少一个 critic 网络，内存和计算成本降低约 40%。

- 🟡 Q: GRPO 的组大小 G 怎么选？
  - A: 通常 4-16。太小 → advantage 估计方差大（排名不稳定）；太大 → 计算成本线性增加。需要在信号质量和计算成本之间平衡。

- 🟡 Q: GRPO 为什么需要 KL 惩罚？
  - A: 防止策略偏离参考模型（通常是 SFT/pretrained model）太远。没有 KL 惩罚的话，模型可能 overfit 到 reward signal 而丢失通用能力（reward hacking 的一种形式）。

## 与其他概念的关系
- 上位概念：[[reinforcement-learning|强化学习]]、[[policy-gradient|策略梯度]]
- 前身：[[ppo|PPO]]（GRPO 是 PPO 的简化）
- 对比：[[dpo|DPO]]（离线方法，不需要在线采样）
- 常配合：[[process-reward-model|PRM]]、[[self-consistency|Self-Consistency]]、[[semantic-entropy|Semantic Entropy]]
