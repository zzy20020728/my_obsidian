---
title: "Discounted Beta-Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards"
type: paper
tags: [RLVR, GRPO, reward-estimation, Beta-Bernoulli, variance-reduction, sample-efficiency, Bayesian]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2603.18444]
status: active
---

# DBB: Discounted Beta-Bernoulli Reward Estimation

## 基本信息
- **作者**: Haechan Kim, Soohyun Ryu, Gyouk Chu, Doohyuk Jang, Eunho Yang
- **机构**: 未明确标注（推测为 KAIST，基于 Eunho Yang 的 affiliation）
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.18444v1 [cs.LG])
- **链接**: https://arxiv.org/abs/2603.18444

## 一句话总结
> 从统计估计视角重新审视 RLVR，将 reward 建模为 Bernoulli 分布，提出 **Discounted Beta-Bernoulli (DBB)** 估计器：利用历史 reward 统计量的折扣 Beta 后验替代点估计，以小量 bias 换取大幅 variance reduction，理论上避免 variance collapse，在 Qwen3-8B 上 OOD benchmark 平均 +6.92%，**无额外计算/内存开销**。

## 摘要
现有 group-based RLVR 方法存在严重的 **sample inefficiency**，根源在于：
1. **点估计的高方差**: 少量 rollout 的 empirical mean 作为 reward 估计，方差大
2. **Variance collapse**: 当所有 rollout 奖励相同时，advantage 归零，浪费计算
3. **on-policy 信息丢失**: 历史 rollout 在单次梯度更新后被丢弃

DBB 将 RLVR 重新形式化为 **分布估计** 问题：reward 是策略诱导的 Bernoulli 分布的采样，advantage 计算等价于从有限数据估计 reward 分布。通过 Beta-Bernoulli 共轭先验，折扣历史观测以适应非平稳分布，实现 bias-variance trade-off。

## 核心贡献
1. **统计估计视角**: 首次将 RLVR 中的 reward/advantage 计算重新框定为 Bernoulli 参数的分布估计问题，而非简单的点估计
2. **DBB 估计器**: 设计折扣 Beta-Bernoulli 模型，通过指数折扣历史后验参数适应策略的非平稳性
3. **理论保证**: 证明 DBB 的 variance 严格低于点估计，且永远不会 collapse 到零；MSE 分析表明在 reward 平缓变化时 DBB 优于点估计
4. **零额外成本**: 与 replay-based 方法不同，DBB 仅维护每个 prompt 的两个标量 (α, β)，无需存储 token probability 或额外 forward pass

## 方法详解

### 1. Reward 的统计建模
给定 prompt q，策略 π_θ 生成 response o，binary reward 建模为：
$$X_{\tau,i} \sim \mathrm{Bernoulli}(p_\tau)$$

其中 $p_\tau = \mathbb{P}(X_{\tau,i}=1 | q, \pi_{\theta_{\eta,\mathrm{old}}})$ 表示当前策略在 epoch τ 对 prompt q 的正确率。

**点估计的问题**:
$$\hat{p}_\tau^{\mathrm{pt}} = \frac{1}{N}\sum_{i=1}^N X_{\tau,i}, \quad \mathrm{Var}(\hat{p}_\tau^{\mathrm{pt}} | p_\tau) = \frac{p_\tau(1-p_\tau)}{N}$$

N 小时方差大，且当 $\hat{p} \in \{0, 1\}$（所有 rollout 全对或全错）时，estimated variance collapse 为 0。

### 2. Beta-Bernoulli 共轭模型
利用 Beta 分布作为 Bernoulli 参数的共轭先验：
$$p_\tau \sim \mathrm{Beta}(\alpha_\tau, \beta_\tau)$$

**朴素版本**（假设平稳）：
$$\alpha_\tau = \alpha_{\tau-1} + S_\tau, \quad \beta_\tau = \beta_{\tau-1} + N - S_\tau$$

问题：RLVR 中策略持续演化，reward 分布非平稳，历史观测不应与最新观测等权。

### 3. Discounted Beta-Bernoulli (DBB)
引入折扣因子 λ ∈ (0,1] 对历史后验进行指数衰减：

$$\alpha_\tau = \lambda \alpha_{\tau-1} + S_\tau, \quad \beta_\tau = \lambda \beta_{\tau-1} + N - S_\tau$$

**DBB 估计器**:
$$\hat{p}_\tau^{\mathrm{dbb}} = \frac{\alpha_\tau}{\alpha_\tau + \beta_\tau}$$

**Variance 估计**:
$$\widehat{\mathrm{Var}}^{\mathrm{dbb}}(X_\tau | \alpha_\tau, \beta_\tau) = \frac{\alpha_\tau \beta_\tau}{(\alpha_\tau + \beta_\tau)^2}$$

### 4. 统计性质分析

**期望**（引入 bias，shrinkage toward historical mean）:
$$\mathbb{E}[\hat{p}_\tau^{\mathrm{dbb}} | p_\tau, \alpha_{\tau-1}, \beta_{\tau-1}] = w \mu_{\tau-1} + (1-w) p_\tau$$

其中 $w = \frac{\lambda(\alpha_{\tau-1}+\beta_{\tau-1})}{\lambda(\alpha_{\tau-1}+\beta_{\tau-1})+N}$ 控制历史信息权重。

**Variance**（严格低于点估计）:
$$\mathrm{Var}(\hat{p}_\tau^{\mathrm{dbb}}) = (1-w)^2 \frac{p_\tau(1-p_\tau)}{N}$$

由于 $(1-w)^2 < 1$，DBB 的 variance 严格小于点估计的 $p_\tau(1-p_\tau)/N$。

**MSE 分析**:
$$\mathrm{MSE}(\hat{p}_\tau^{\mathrm{dbb}}) = \underbrace{\left(\sum_{k=0}^\tau c_k p_k - p_\tau\right)^2}_{\mathrm{Bias}^2} + \underbrace{\frac{\sum_{k=1}^\tau \lambda^{2(\tau-k)} N p_k(1-p_k)}{H_\tau^2}}_{\mathrm{Variance}}$$

当 reward 概率缓慢变化时（$p_k \approx p_\tau$），bias 项小，而 variance 大幅降低 → MSE 整体低于点估计。

### 5. 防止 Variance Collapse
点估计下，当 $\hat{p} = 0$ 或 $\hat{p} = 1$ 时 variance 为 0 → advantage 为 0 → 该 prompt 的 rollout 全部浪费。

DBB 下，由于历史累积，$\alpha_\tau$ 和 $\beta_\tau$ 不会同时为 0，因此 $\widehat{\mathrm{Var}}^{\mathrm{dbb}} > 0$ **始终成立**，保证每个 prompt 都能产生非零 training signal。

## 实验结果

### 主要结果

| Method | MATH500 | Minerva | AIME24 | AIME25 | AMC24 | Olympiad | ID Avg. | MMLU-Pro | GPQA-D | BBH | OOD Avg. |
|--------|---------|---------|--------|--------|-------|----------|---------|----------|--------|-----|----------|
| **Qwen3-8B-Base** | | | | | | | | | | | |
| GRPO | 88.05 | 39.25 | 30.00 | 26.67 | 56.39 | 54.82 | 49.20 | 50.02 | 42.93 | 59.18 | 50.71 |
| RePO | 86.20 | 35.71 | 27.92 | 22.92 | 56.11 | 52.61 | 46.91 | 53.15 | 43.93 | 42.96 | 46.68 |
| **GRPO-DBB** | **88.92** | **39.48** | **34.17** | **30.83** | **60.00** | **56.34** | **51.62** | **63.12** | **46.46** | **63.32** | **57.63** |
| Δ vs GRPO | +0.87 | +0.23 | **+4.17** | **+4.16** | **+3.61** | +1.52 | **+2.42** | **+13.10** | +3.53 | +4.14 | **+6.92** |
| **Qwen3-1.7B-Base** | | | | | | | | | | | |
| GRPO | 68.37 | 25.74 | 8.75 | 4.58 | 23.61 | 29.41 | 26.74 | 33.72 | 20.70 | 12.71 | 22.38 |
| **GRPO-DBB** | **71.95** | **26.31** | **14.17** | **7.08** | **26.94** | **33.29** | **29.96** | **40.83** | **29.80** | **33.97** | **34.87** |
| Δ vs GRPO | +3.58 | +0.57 | **+5.42** | +2.50 | +3.33 | +3.88 | **+3.22** | +7.11 | **+9.10** | **+21.26** | **+12.49** |

**关键发现**:
- **OOD 泛化极为突出**: 1.7B 模型 OOD 平均 +12.49%，BBH 上 +21.26%，说明 Bayesian 不确定性建模显著改善泛化
- **难题提升更明显**: AIME24/25 提升最大（+4~5%），这些 benchmark 的 rollout 正确率低，正是点估计高方差/variance collapse 的重灾区
- **零额外成本**: 与 RePO（需额外 GPU 内存）相比，DBB 仅维护 (α, β) 标量

### 折扣因子 λ 的影响
| λ | 8B ID Avg. | 1.7B ID Avg. |
|---|------------|--------------|
| 1.0 | 50.20 | 26.29 |
| 0.75 | **51.62** | 29.28 |
| 0.5 | 50.61 | **29.96** |
| 0.25 | 50.44 | 29.00 |
| GRPO | 49.20 | 26.74 |

- λ=1.0（无折扣）退化为朴素 Beta-Bernoulli，因非平稳导致 bias 过大
- 小模型偏好更小 λ（0.5），大模型偏好较大 λ（0.75）→ 小模型策略变化更剧烈
- 所有 λ 值均优于 GRPO → DBB 对 λ 不敏感

### 训练动态
- GRPO-DBB 维持更长的 response length 和更高的 entropy → 更好的探索
- 训练过程中 validation accuracy 持续上升，而 GRPO 更早饱和
- Reward 曲线更平滑，避免了 GRPO 的震荡

## 与 SPC/URLVR 研究的关系

### 直接关联
1. **Historical reward statistics 与 SPC 的 probing mechanism**: DBB 通过历史 reward 统计量提供更稳定的 outcome-level signal。SPC 的 probing 在 step-level 提取 semantic consistency signal。两者可层级化组合——DBB 提供 stable outcome advantage baseline，SPC 在此基础上做 step-level redistribution
2. **Variance collapse 与无监督设定**: 在 URLVR 中，pseudo-label 基于 majority voting，variance collapse 更频繁（因 pseudo-label 更不可靠）。DBB 的 anti-collapse 性质对 URLVR 尤其重要——即使所有 rollout 得到相同 pseudo-reward，历史信息仍保证非零 advantage
3. **Bayesian 不确定性估计**: DBB 的后验参数 (α, β) 隐式编码了模型对每个 prompt 的"熟悉程度"。这与 Co-Evolving Verifier 的校准目标一致——verifier 需要知道何时信任自己的判断
4. **折扣机制与策略非平稳性**: DBB 的 λ 折扣是对策略演化的适应。在 SPC 中，probing signal 的可靠性也随训练变化，可借鉴类似的历史折扣策略

### 对 Co-Evolving Verifier 的启示
- **Verifier 校准**: DBB 的 Beta 后验可直接作为 verifier 的 uncertainty estimate——$\alpha/(\alpha+\beta)$ 越接近 0 或 1，verifier 越有把握
- **冷启动问题**: DBB 用 (α₀=1, β₀=1) 初始化（uniform prior），在 URLVR 的 verifier cold-start 阶段提供无信息先验是合理的
- **信号稳定性**: Co-Evolving Verifier 产生的 reward 本身不可靠，DBB 的 Bayesian smoothing 可作为 verifier reward 的后处理，避免 noisy verifier signal 导致训练崩溃

## 面试 Q&A

### Q1: 为什么 DBB 用 bias 换 variance 是值得的？这在 ML 中的理论基础是什么？
**A**: 这是经典的 **bias-variance trade-off** 在强化学习中的具体实例化。在 RLVR 中，由于计算约束每个 prompt 仅采样 N=8 个 rollout，点估计的 variance 为 $p(1-p)/8$，在 p 接近 0.5 时高达 0.03。DBB 通过 shrinkage（向历史均值收缩）将 variance 降低 $(1-w)^2$ 倍。关键洞察是：(1) RLVR 中策略变化通常是渐进的（p_τ ≈ p_{τ-1}），所以 bias 小；(2) variance 直接影响 advantage 的信噪比，而 advantage 的方向（正/负）比精确数值更重要——小 bias 不改变方向，但大 variance 会。这在统计学中对应 **James-Stein estimator** 的思想——在高维/小样本场景下，shrinkage estimator 在 MSE 意义下胜过 MLE。

### Q2: DBB 与 experience replay 的区别和优势是什么？
**A**: Experience replay（如 RePO）**完整重用**历史 rollout 的 token/probability/reward，需要 importance sampling 修正 off-policy bias，引入大量 GPU 内存（存储 token-level probability）和计算（额外 forward pass）。DBB 只 **压缩重用**历史信息——每个 prompt 仅维护 2 个标量 (α, β)，是极致的信息压缩。效果上，DBB 在 1.7B 模型上 ID +3.22/OOD +12.49，而 RePO 仅 ID -0.77/OOD +8.52（甚至 ID 下降）。这说明 lightweight Bayesian smoothing 比 heavy replay 更有效，因为 replay 的 importance weight 在策略快速变化时可能不准确。

### Q3: 为什么 OOD 提升比 ID 提升大得多？
**A**: 两个原因：(1) **不确定性保留**: DBB 的 Bayesian 后验天然保留了模型对 prompt 的不确定性信息（通过 α+β 的大小），训练时不会因 variance collapse 而过早放弃困难/不熟悉的 prompt。这些"困难 prompt"的特征迁移到 OOD 任务；(2) **更稳定的探索**: 点估计的 variance collapse 导致模型在简单题上过拟合，而 DBB 保证所有 prompt 都有非零 training signal，维持了更广泛的能力覆盖。特别在 BBH（+21.26%）上，这是一个包含多种推理类型的 benchmark，受益于更均匀的能力发展。
