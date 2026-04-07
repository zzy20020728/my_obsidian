---
title: PPO (Proximal Policy Optimization)
type: concept
tags: [RL, policy-gradient, 对齐, RLHF]
created: 2026-04-07
updated: 2026-04-07
sources: [raw/papers/schulman-2017-ppo.pdf]
status: draft
---

# PPO (Proximal Policy Optimization)

## 定义
> Proximal Policy Optimization，近端策略优化。一种 on-policy 的策略梯度算法，通过 clip 机制限制策略更新幅度，在保证训练稳定性的同时实现高效优化。是当前 [[rlhf|RLHF]] 中最常用的 RL 算法。

## 关键性质
1. On-policy 算法：用当前策略采样的数据来更新当前策略
2. Clip 机制：限制新旧策略的比率在 `[1-ε, 1+ε]` 范围内，防止更新过大
3. 不需要二阶优化（对比 [[trpo|TRPO]]），实现简单、计算高效

## 直觉理解
> 想象你在调整一个旋钮。TRPO 说「每次只能转一小步，而且要精确计算这一步有多大」（需要二阶导数）。PPO 说「随便转，但如果转太多了我就截断」（clip）。效果差不多，但 PPO 简单太多了。

## 数学表达

### 目标函数（PPO-Clip）

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的概率比
- $\hat{A}_t$ 是优势函数估计（通常用 GAE）
- $\epsilon$ 通常取 0.2

### 完整损失

$$L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta]$$

- $L^{VF}$：Value function 损失
- $S$：策略熵（鼓励探索）

## 相关论文
- [[wiki/papers/schulman-2017-ppo|Schulman et al., 2017 — PPO 原论文]]
- [[wiki/papers/schulman-2015-trpo|Schulman et al., 2015 — TRPO]]

## 在 RLHF 中的角色

在 [[rlhf|RLHF]] 的第三阶段，PPO 用于：
1. 用 reward model 的打分作为 reward signal
2. 优化 LLM 的策略，使其生成更符合人类偏好的回复
3. 加入 KL 惩罚防止偏离 SFT 模型太远

对比其他方法：
- [[dpo|DPO]]：直接从偏好数据优化，不需要 reward model，也不需要 PPO
- [[grpo|GRPO]]：DeepSeek 提出的 group relative 方法，是 PPO 的替代方案

详细对比见 → [[wiki/synthesis/rl-alignment-methods|RL 对齐方法对比]]

## 面试常问点

- 🔴 Q: PPO 的 clip 机制是怎么工作的？为什么要 clip？
  - A: clip 限制了策略更新的幅度。当 $r_t(\theta)$ 偏离 1 太多时（新旧策略差异太大），clip 会截断梯度。这保证了训练稳定性。

- 🔴 Q: PPO 和 TRPO 的区别？
  - A: TRPO 用 KL 散度约束 + 共轭梯度法（二阶优化），PPO 用 clip（一阶优化）。PPO 更简单、更快、效果相当。

- 🟡 Q: PPO 中 GAE 是什么？
  - A: Generalized Advantage Estimation，用 λ 参数在偏差和方差之间做权衡来估计优势函数。

## 与其他概念的关系
- 上位概念：[[reinforcement-learning|强化学习]]、[[policy-gradient|策略梯度]]
- 下位概念：[[ppo-clip|PPO-Clip]]、[[ppo-penalty|PPO-Penalty]]
- 相关概念：[[trpo|TRPO]]、[[gae|GAE]]、[[rlhf|RLHF]]、[[grpo|GRPO]]、[[dpo|DPO]]
