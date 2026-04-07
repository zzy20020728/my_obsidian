---
title: "Proximal Policy Optimization Algorithms"
type: paper
tags: [RL, policy-gradient, PPO, OpenAI]
created: 2026-04-07
updated: 2026-04-07
sources: [raw/papers/schulman-2017-ppo.pdf]
status: draft
---

# Proximal Policy Optimization Algorithms

## 基本信息
- **作者**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- **机构**: OpenAI
- **年份**: 2017
- **会议/期刊**: arXiv preprint (arXiv:1707.06347)
- **链接**: https://arxiv.org/abs/1707.06347

## 一句话总结
> 提出 PPO 算法，用简单的 clip 机制替代 TRPO 的复杂约束优化，在保持训练稳定性的同时大幅简化实现。

## 摘要
策略梯度方法在 RL 中效果好但步长难调——太大训练崩溃，太小收敛太慢。本文提出 PPO，一种新的策略梯度方法，通过 clip surrogate objective 实现 "多次小步更新" 的效果，同时避免了 TRPO 的计算复杂性。

## 核心贡献
1. 提出 PPO-Clip 目标函数，用 clip 代替 KL 约束
2. 证明 PPO 在多个连续控制任务上达到或超过 TRPO 的效果
3. 实现极其简单，几行代码就能实现核心逻辑

## 方法

### 问题定义
标准策略梯度的问题：更新步长难以控制。TRPO 用 KL 约束解决但计算复杂（需要共轭梯度法）。能否用更简单的方法达到同样效果？

### 技术方案
定义概率比 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，然后 clip 这个比率：

### 关键公式

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

直觉：取 clipped 和 unclipped 目标的较小值，形成一个悲观的（下界的）估计。

## 实验结果

| 环境 | PPO | TRPO | A2C |
|------|-----|------|-----|
| HalfCheetah | **最优** | 接近 | 较差 |
| Hopper | **最优** | 接近 | 较差 |
| Walker2d | **最优** | 接近 | 较差 |
| Atari 平均 | **最优** | — | A2C 接近 |

PPO 在几乎所有任务上都达到或超过 TRPO，同时训练速度更快。

## 与其他工作的关系
- 基于：[[wiki/papers/schulman-2015-trpo|TRPO (Schulman et al., 2015)]]
- 对比：A2C/A3C, ACER
- 被引用：被后续所有 [[rlhf|RLHF]] 工作使用（InstructGPT, ChatGPT 等）
- 后续改进：[[wiki/concepts/grpo|GRPO]]、[[wiki/concepts/dpo|DPO]] 可以视为替代方案

## 面试相关
> 这篇论文可能被问到的面试问题

- Q: 为什么 PPO 比 TRPO 好？
- A: 效果相当但实现简单得多。TRPO 需要共轭梯度法（二阶优化），PPO 只需要一阶梯度 + clip。

- Q: PPO 的 ε 一般取多少？
- A: 通常 0.2。太大训练不稳定，太小收敛慢。

- Q: PPO 在 LLM 对齐中是怎么用的？
- A: [[rlhf|RLHF]] 第三阶段，用 reward model 的打分做 reward，PPO 优化策略使 LLM 输出更符合人类偏好。加 KL 惩罚防止偏离 SFT 模型太远。

## 个人笔记
> *读完后在这里记录感想、疑问、创新启发*

