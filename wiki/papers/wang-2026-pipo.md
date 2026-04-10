---
title: "PIPO: Policy Improvement Policy Optimization"
type: paper
tags: [RLVR, GRPO-flaw, gradient-explosion, mode-collapse, policy-improvement, dual-stage, explore-verify, Beihang, PKU]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2604.00860]
status: active
---

# PIPO: Policy Improvement Policy Optimization

## 基本信息
- **作者**: Wang et al.
- **机构**: Beihang University & Peking University (PKU)
- **年份**: 2026 (arXiv: April 2026)
- **会议/期刊**: arXiv preprint (arXiv:2604.00860)
- **链接**: https://arxiv.org/abs/2604.00860
- **Base Model**: 4B / 7B 级别模型 (GRPO/GSPO/DAPO baselines)

## 一句话总结
> 揭露 GRPO 的数学缺陷——group-relative normalization 引入 state-dependent gradient scaling factor η(p)，当 p→0 或 p→1 时 η(p)→∞，导致梯度爆炸和 mode collapse；提出 PIRL 框架（最大化累积跨迭代改进）和 PIPO 算法（双阶段 explore-verify），在 GSPO/DAPO 等多种 baseline 上均获得稳定提升。

## 摘要
GRPO 是当前 RLVR 领域最主流的优化算法，被 DeepSeek-R1、DAPO 等广泛采用。然而本文从数学角度严格分析发现，GRPO 的 group-relative advantage normalization 引入了一个隐含的 state-dependent gradient scaling factor η(p)，该因子在 policy 概率趋近 0 或 1 时趋向无穷大，造成梯度爆炸，最终导致 mode collapse。为解决此问题，论文提出 Policy Improvement RL (PIRL) 框架，将优化目标从单步 reward 最大化转变为**累积跨迭代改进最大化**，并设计 PIPO 算法实现双阶段训练流程：(1) Explore 阶段用多样化 rollouts 探索解空间；(2) Verify 阶段通过对比当前迭代与上一迭代的 rollouts 计算 Policy Improvement Reward。实验表明 PIPO 可作为即插即用模块，在 GRPO、GSPO、DAPO 等多种 RL 算法上均实现一致提升。

## 核心贡献
1. **GRPO 梯度爆炸的严格数学证明**: 推导出 group-relative normalization 引入的 gradient scaling factor η(p) = √(G-1)/[G·p(1-p)]，证明当 p→0 或 p→1 时 η(p)→∞，解释了 GRPO 训练中常见的 mode collapse 现象
2. **PIRL 框架**: 提出以累积 inter-iteration improvement 作为优化目标的新范式，跳出传统单步 reward 最大化的限制
3. **PIPO 双阶段算法**: 设计 explore + verify 的训练流程，通过 retrospective verification（对比相邻迭代的 rollout 质量）构建 Policy Improvement Reward
4. **即插即用兼容性**: PIPO 可与 GRPO、GSPO、DAPO 等多种 RL 算法组合使用，一致提升性能

## 方法

### 问题定义
在标准 RLVR 框架下，给定 query q 和 policy π_θ，GRPO 使用 group-relative advantage normalization：

对 G 个 rollouts {y_1, ..., y_G}，advantage 计算为：

$$\hat{A}_i = \frac{R(y_i) - \text{mean}(\{R(y_j)\})}{\text{std}(\{R(y_j)\})}$$

本文分析发现，std 归一化引入了隐含的 gradient scaling factor，使得训练在 boundary 处不稳定。

### 技术方案

#### 1. GRPO 梯度爆炸分析

Group-relative normalization 引入的 scaling factor：

$$\eta(p) \sim \frac{\sqrt{G-1}}{G \cdot p(1-p)}$$

当 p → 0 或 p → 1 时，η(p) → ∞。这意味着：
- **简单问题**（模型几乎全对，p→1）：梯度被无限放大，过度强化已确信的答案
- **困难问题**（模型几乎全错，p→0）：梯度同样被无限放大，导致 noisy gradients
- 两种情况都推动 policy 快速收敛到极端值 → **mode collapse**

这解释了为什么 GRPO 训练中常见 entropy 快速下降和 reward hacking 现象。

#### 2. PIRL: Policy Improvement RL 框架

核心思想：不是最大化单步 reward，而是最大化**跨迭代的累积改进**：

$$\mathcal{J}_{PIRL}(\theta) = \sum_{t=1}^{T} \mathbb{E}_{q \sim \mathcal{D}} \left[ \text{Improvement}(\pi_{\theta_t}, \pi_{\theta_{t-1}} | q) \right]$$

通过将优化目标从 absolute performance 转变为 relative improvement，避免了 η(p) 在极端值处的爆炸。

#### 3. PIPO: Dual-Stage Explore-Verify 算法

**Stage 1 - Explore（探索阶段）**:
- 用当前 policy π_θ_t 对 query 集合进行多样化 rollouts
- 保存本次迭代的 rollout 结果 {y^t_1, ..., y^t_G}

**Stage 2 - Verify（验证阶段）**:
- 对比当前迭代 rollouts 与上一迭代 rollouts
- 计算 Policy Improvement Reward：

$$R_{PI} = [\text{local attribution}] \cdot \varphi(\text{global verification})$$

其中 local attribution 衡量单条 rollout 的质量变化，global verification 通过全局统计验证改进的显著性。

**Closed-loop vs Open-loop**:
- **Closed-loop**: 同时使用当前迭代和上一迭代的数据进行训练，利用两次迭代的对比信息
- **Open-loop**: 仅使用当前迭代的数据，但以上一迭代的统计量作为 baseline

### 关键公式

**GRPO gradient scaling divergence**:
$$\eta(p) \sim \frac{\sqrt{G-1}}{G \cdot p(1-p)} \to \infty \quad \text{when } p \to 0 \text{ or } p \to 1$$

**Policy Improvement Reward**:
$$R_{PI}(y_i) = \text{LocalAttribution}(y_i) \cdot \varphi(\text{GlobalVerification})$$

**PIPO dual-stage optimization**:
- Explore: $\{y^t_1, ..., y^t_G\} \sim \pi_{\theta_t}(\cdot|q)$
- Verify: 对比 $\{y^t\}$ vs $\{y^{t-1}\}$，计算 improvement signal

### 实现细节
| 参数 | 值 |
|------|-----|
| 基础 RL 算法 | GRPO / GSPO / DAPO |
| Explore 阶段 rollouts | 多样化采样（higher temperature） |
| Verify 阶段 | 对比相邻迭代 rollouts |
| 额外开销 | 每步 12-19% overhead |
| 模型规模 | 4B / 7B |
| Closed-loop 模式 | 使用两次迭代数据 |
| Open-loop 模式 | 仅使用当前迭代数据 |

## 实验结果

### 主要结果

| 方法 | AIME25 | 4B Avg | 说明 |
|------|:------:|:------:|------|
| GSPO (baseline) | X | - | 标准训练 |
| GSPO + PIPO | X + 7.4% | - | **+7.4% absolute improvement** |
| DAPO (baseline) | - | ~48 | 标准训练 |
| DAPO + PIPO | - | **51.9** | 显著提升 |

### 跨算法兼容性

| RL 算法 | Without PIPO | With PIPO | 提升 |
|---------|:----------:|:---------:|:----:|
| GRPO | baseline | ✓ 提升 | 一致正向 |
| GSPO | baseline | ✓ 提升 (+7.4% AIME25) | 一致正向 |
| DAPO | baseline | ✓ 提升 (4B Avg 51.9) | 一致正向 |

### 计算开销 vs 收益

| 指标 | 值 |
|------|-----|
| 每步额外 overhead | 12-19% |
| 样本效率 | 优于 baseline（更少总步数达到相同性能） |
| 综合效率 | 尽管单步更贵，但 better sample efficiency 补偿了 overhead |

## 与其他工作的关系

### GRPO 系列改进对比
| 维度 | PIPO | [[wiki/papers/xie-2025-capo|CAPO]] | [[wiki/papers/liu-2025-vppo|VPPO]] | [[wiki/papers/yang-2025-trapo|TraPO]] |
|------|------|------|------|-------|
| 解决的问题 | 梯度爆炸 / mode collapse | 对比学习增强 | 价值函数辅助 | Trajectory-level 优化 |
| 修改位置 | Reward + 优化目标 | Advantage 计算 | Critic 网络 | Trajectory 选择 |
| 即插即用 | ✓ | ✓ | 需额外 value head | ✓ |
| 理论保证 | ✓（η(p) 分析） | 部分 | ✓（PPO 理论） | 部分 |

### 与 [[wiki/papers/he-2026-urlvr-scale|He et al. URLVR Survey]] 的关系
- He et al. 发现所有 intrinsic 方法的 rise-then-fall pattern
- PIPO 从另一个角度解释了 collapse 原因：η(p) gradient explosion
- 两者互补：He et al. 证明 sharpening 是宏观机制，PIPO 揭示 GRPO 的微观梯度不稳定性

### 与 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的关系
- TTRL 使用 GRPO 作为默认优化器，受 η(p) divergence 影响
- PIPO 发现的 p→0/p→1 梯度爆炸解释了 TTRL 在 hard/easy queries 上的不稳定性
- TTRL + PIPO 的组合可能进一步提升 test-time training 的稳定性

### 与 [[wiki/concepts/grpo|GRPO]] 概念页的关系
- PIPO 是对 GRPO 最深入的数学分析之一
- η(p) divergence 是 GRPO group-relative normalization 的固有缺陷
- 建议 GRPO 概念页增加 PIPO 发现的缺陷分析

## 局限性与开放问题
1. **Overhead 非零**: 12-19% 的额外计算开销在大规模训练中可能不容忽视
2. **需要保存历史 rollouts**: Verify 阶段需要上一迭代的数据，增加内存/存储需求
3. **η(p) 分析基于简化假设**: 实际训练中 gradient scaling 的行为可能更复杂
4. **未覆盖 step-level 信号**: PIPO 主要改进 outcome-level 的 RL 优化，未涉及 step-level credit assignment
5. **与 DAPO 的 Dynamic Sampling 交互**: DAPO 已部分缓解了 group 内方差问题（Dynamic Sampling 过滤全对/全错 groups），PIPO 与之的交互效应需进一步分析

## 面试相关
> 这篇论文可能被问到的面试问题

- **Q: GRPO 的 group-relative normalization 有什么数学缺陷？** 🔴
- A: GRPO 对 group 内 reward 做标准差归一化，引入了隐含的 gradient scaling factor η(p) ∝ 1/[p(1-p)]。当 p→0（全错）或 p→1（全对）时 η(p)→∞，导致梯度爆炸。这解释了 GRPO 训练中常见的 mode collapse——模型在简单/困难 query 上过度更新，entropy 快速下降。

- **Q: PIPO 的 dual-stage explore-verify 流程是怎么设计的？** 🟡
- A: Stage 1 (Explore)：对当前 policy 进行多样化 rollouts 采样；Stage 2 (Verify)：将当前迭代的 rollouts 与上一迭代的 rollouts 对比，计算 Policy Improvement Reward = local attribution × φ(global verification)。通过回顾性验证（retrospective verification）衡量策略改进程度，而非绝对 reward 值。

- **Q: PIPO 为什么能解决 GRPO 的梯度爆炸问题？** 🟡
- A: PIRL 框架将优化目标从 absolute reward 转变为 relative improvement。当 p→0 或 p→1 时，improvement signal 不会像 GRPO 的 η(p) 那样 diverge，因为对比的是两次迭代之间的相对变化而非绝对值。这从根本上绕开了 group-relative normalization 的 scaling 问题。

- **Q: PIPO 的 closed-loop 和 open-loop 模式有什么区别？** 🟢
- A: Closed-loop 同时使用当前和上一迭代的 rollout 数据进行训练，利用更丰富的对比信息；Open-loop 仅用当前迭代数据，但以上一迭代统计量为 baseline。Closed-loop 通常效果更好但数据利用更复杂。

- **Q: PIPO 发现的 η(p) 问题对 TTRL 有什么影响？** 🟡
- A: TTRL 默认使用 GRPO，受 η(p) divergence 影响。在 TTRL 场景中：hard queries（模型初始正确率低，p→0）和 easy queries（正确率高，p→1）的梯度都被过度放大，导致训练不稳定。TTRL 可以考虑采用 PIPO 或至少切换到 DAPO（已部分缓解 group variance 问题）来提高稳定性。

## 个人笔记
### 与 SPC 研究方案的关系
PIPO 的发现对 SPC 实验方案有直接影响：

1. **GRPO boundary 梯度爆炸影响 SPC 实验**: SPC 方案中的 RL 训练如果使用 GRPO，hard/easy queries 的梯度信号会不稳定。对于 SPC 的 step-level 信号，如果某些 step 的正确率接近 0 或 1，η(p) divergence 会放大噪声。
2. **建议 SPC 实验使用 DAPO 替代 raw GRPO**: DAPO 的 Dynamic Sampling（过滤全对/全错 groups）和 token-level normalization 已部分缓解了 η(p) 问题，是更安全的 baseline。
3. **PIPO-style retrospective verification 的启示**: SPC 的三层架构可以借鉴 PIPO 的 "对比相邻迭代" 思路——在 Co-Evolving PRM 的训练中，对比前后迭代的 PRM 质量变化作为额外信号。
4. **理论工具**: η(p) 分析为理解 SPC 实验中的训练不稳定性提供了定量工具。如果训练出现异常，可以检查是否是 boundary gradient explosion 导致的。
