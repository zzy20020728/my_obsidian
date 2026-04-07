---
title: 操作日志
type: log
---

# 操作日志

## [2026-04-07] init | Wiki 知识库初始化
- 创建目录结构：raw/, wiki/, plans/, templates/
- 创建 Schema (AGENTS.md)、索引 (index.md)、模板
- 创建秋招备战总计划
- 初始化 Git 仓库并推送到 GitHub

## [2026-04-07] update | 学习计划重构
- 修正个人背景：实习内容就是 URLVR 研究（10:00-19:00），算法岗积累在白天自然完成
- 重构时间安排：晚上 3h 全部给 Java，周末做论文 Wiki 整理 + 八股
- 移除 LeetCode 安排（暂不需要）
- 更新 master-plan.md 和 weekly-tracker.md

## [2026-04-07] content | 教程与示例页面
- 创建 Obsidian & Wiki 使用教程 (plans/obsidian-tutorial.md)
- 创建示例概念页：PPO (wiki/concepts/ppo.md)
- 创建示例概念页：RLHF (wiki/concepts/rlhf.md)
- 创建示例论文页：Schulman 2017 PPO (wiki/papers/schulman-2017-ppo.md)
- 更新 index.md 索引
- 优化 Obsidian 配置（shortest link format）

## [2026-04-07] ingest | URLVR 四篇论文批量摄入
- 摄入论文：EMPO (Zhang et al., 2025) — 语义熵最小化，完全无监督 RLVR
- 摄入论文：SPARK (Rahman et al., 2025) — 三阶段 PRM 训练，step-level reward
- 摄入论文：PRISM (Ghimire et al., 2026) — PRM + self-certainty 混合，纯内部信号失败分析
- 摄入论文：Self-Judge (Wu et al., 2026) — Actor-Judge 多模态无监督自进化
- 创建论文页：wiki/papers/zhang-2025-empo.md
- 创建论文页：wiki/papers/rahman-2025-spark.md
- 创建论文页：wiki/papers/ghimire-2026-prism.md
- 创建论文页：wiki/papers/wu-2026-self-judge.md
- 创建概念页：wiki/concepts/semantic-entropy.md（语义熵）
- 创建概念页：wiki/concepts/self-consistency.md（自一致性）
- 创建概念页：wiki/concepts/process-reward-model.md（过程奖励模型）
- 创建概念页：wiki/concepts/grpo.md（组相对策略优化）
- 创建概念页：wiki/concepts/reward-hacking.md（奖励攻击）
- 创建综合分析：wiki/synthesis/urlvr-landscape.md（URLVR 领域综述，含多维分类对比表格）
- 更新 index.md 索引（按类别组织所有新条目）

## [2026-04-08] ingest | URLVR 第二批四篇论文摄入
- 摄入论文：MCNIG (Royer et al., 2026) — 信息论自动生成 PRM 训练数据，O(N) 复杂度
- 摄入论文：ProRAG (Wang et al., 2026) — 四阶段 RAG RL，MCTS-based PRM，dual-granularity advantage
- 摄入论文：CTRL-RAG (Tan et al., 2026) — Contrastive Likelihood Reward，轻量 RAG faithfulness 信号
- 摄入论文：How Far Can URLVR Scale? (He et al., 2026, ICLR) — 统一理论框架 [DRAFT: 仅 abstract]
- 创建论文页：wiki/papers/royer-2026-mcnig.md
- 创建论文页：wiki/papers/wang-2026-prorag.md
- 创建论文页：wiki/papers/tan-2026-ctrl-rag.md
- 创建论文页：wiki/papers/he-2026-urlvr-scale.md [draft]
- 创建概念页：wiki/concepts/information-gain.md（信息增益）
- 创建概念页：wiki/concepts/mcts.md（蒙特卡洛树搜索）
- 创建概念页：wiki/concepts/contrastive-likelihood.md（对比似然）
- 更新概念页：wiki/concepts/process-reward-model.md（添加 MCNIG/ProRAG 的 PRM 训练方法 + 对比表格）
- 大幅更新综合分析：wiki/synthesis/urlvr-landscape.md（从4篇扩展到8篇，添加 RAG 分类维度、Sharpening 理论、PRM 训练数据生成方法对比）
- 更新 index.md 索引（添加新论文、新概念页、RAG+RL 分类）
