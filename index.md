---
title: Wiki 索引
type: index
updated: 2026-04-07
---

# Wiki 索引

## 论文 (wiki/papers/)
> 论文摘要与深度分析

### RL 基础
- [[wiki/papers/schulman-2017-ppo|PPO 原论文 (Schulman et al., 2017)]] — 策略梯度经典，RLHF 核心算法

### URLVR（无监督/无参考 RL 推理）
- [[wiki/papers/zhang-2025-empo|EMPO (Zhang et al., 2025)]] — 语义熵最小化，完全无监督，仅需 {q} 不需要 {a}
- [[wiki/papers/rahman-2025-spark|SPARK (Rahman et al., 2025)]] — 三阶段 PRM 训练，step-level reward 超越 GT RLVR
- [[wiki/papers/ghimire-2026-prism|PRISM (Ghimire et al., 2026)]] — 系统性证明纯内部信号不可靠，PRM + self-certainty 混合方案
- [[wiki/papers/wu-2026-self-judge|Self-Judge (Wu et al., 2026)]] — Actor-Judge 多模态无监督自进化，distributional reward modeling

## 概念 (wiki/concepts/)
> 核心概念与知识点

### RL 算法
- [[wiki/concepts/ppo|PPO]] — 近端策略优化，RLHF 中最常用的 RL 算法
- [[wiki/concepts/grpo|GRPO]] — 组相对策略优化，URLVR 四篇论文的共同优化框架

### 对齐技术
- [[wiki/concepts/rlhf|RLHF]] — 基于人类反馈的强化学习，LLM 对齐核心技术

### URLVR 核心概念
- [[wiki/concepts/semantic-entropy|Semantic Entropy (语义熵)]] — EMPO 的核心 reward 信号
- [[wiki/concepts/self-consistency|Self-Consistency (自一致性)]] — 多路径采样一致性，URLVR 基础信号
- [[wiki/concepts/process-reward-model|Process Reward Model (PRM)]] — 步骤级奖励模型，SPARK/PRISM 核心
- [[wiki/concepts/reward-hacking|Reward Hacking (奖励攻击)]] — URLVR 最大挑战，所有论文都涉及

## 实体 (wiki/entities/)
> 模型、团队、公司、人物

*尚无条目*

## 面试 (wiki/interview/)
> 面试知识点整理

- [[wiki/interview/llm-and-rl|大模型 & 强化学习面试题]] — 算法岗核心战场
- [[wiki/interview/ml-basics|机器学习基础面试题]] — ML 八股
- [[wiki/interview/java-and-programming|Java & 编程面试题]] — 应用开发岗需要

## 项目 (wiki/projects/)
> 项目经验与总结

*尚无条目*

## 综合分析 (wiki/synthesis/)
> 跨论文对比与领域综述

- [[wiki/synthesis/urlvr-landscape|URLVR 领域综述]] — 四篇论文多维分类对比，信号来源/打分粒度/Reward 模型/任务类型

## 学习计划 (plans/)
> 秋招备战计划与进度

- [[plans/master-plan|秋招备战总计划]]
- [[plans/weekly-tracker|每周进度追踪]]
- [[plans/obsidian-tutorial|Obsidian & Wiki 使用教程]]

---
*本索引由 LLM 自动维护，每次 ingest/query 操作后更新*
