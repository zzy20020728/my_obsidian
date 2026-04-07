---
title: RLHF (Reinforcement Learning from Human Feedback)
type: concept
tags: [RL, LLM, 对齐, RLHF]
created: 2026-04-07
updated: 2026-04-07
sources: []
status: draft
---

# RLHF (Reinforcement Learning from Human Feedback)

## 定义
> 通过人类反馈信号训练 reward model，再用 RL 算法（通常是 [[ppo|PPO]]）优化 LLM，使其输出更符合人类偏好。是 ChatGPT、GPT-4 等模型对齐的核心技术。

## 三阶段流程

```
Stage 1: SFT（监督微调）
  预训练模型 → 用高质量对话数据微调 → SFT 模型

Stage 2: Reward Model 训练
  收集人类偏好数据（对同一 prompt 的两个回复排序）
  → 训练一个 reward model 来预测人类偏好

Stage 3: RL 优化（PPO）
  用 reward model 给 LLM 输出打分
  → PPO 优化 LLM 策略
  → KL 惩罚防止偏离 SFT 模型太远
```

## 关键性质
1. 解决了「人类偏好难以用 loss function 直接表达」的问题
2. 三阶段串行，pipeline 复杂，训练成本高
3. Reward model 的质量直接决定最终效果

## 面试常问点

- 🔴 Q: RLHF 三个阶段分别是什么？
  - A: SFT → Reward Model → PPO 优化

- 🔴 Q: 为什么需要 KL 惩罚？
  - A: 防止 reward hacking——模型学会骗过 reward model 但实际输出质量下降。KL 惩罚让模型不偏离 SFT 太远。

- 🔴 Q: RLHF 的缺点是什么？[[dpo|DPO]] 怎么解决的？
  - A: 缺点：pipeline 复杂、reward model 不稳定、PPO 训练昂贵。DPO 跳过 reward model，直接从偏好数据优化策略。

## 与其他概念的关系
- 核心组件：[[ppo|PPO]]、[[reward-model|Reward Model]]
- 替代方案：[[dpo|DPO]]、[[grpo|GRPO]]
- 应用于：[[wiki/entities/chatgpt|ChatGPT]]、[[wiki/entities/gpt-4|GPT-4]]

详细对比 → [[wiki/synthesis/rl-alignment-methods|RL 对齐方法对比]]
