---
title: 大模型 & 强化学习面试题
type: interview
tags: [LLM, RL, RLHF, 面试]
created: 2026-04-07
updated: 2026-04-07
sources: []
status: draft
---

# 大模型 & 强化学习面试题

> 这是算法岗面试的核心战场。你的研究方向就是 URLVR，这部分必须做到专家级。

## 🔴 高频考点

### 1. Transformer 架构
- **定义**：基于 self-attention 的序列到序列架构
- **关键点**：Multi-Head Attention、位置编码、Layer Norm、残差连接
- **常见追问**：为什么用 Layer Norm 而不是 Batch Norm？Attention 复杂度多少？如何优化？
- **参考回答**：*待填充*
- **关联概念**：[[wiki/concepts/transformer|Transformer]]

### 2. RLHF 全流程
- **定义**：通过人类反馈的强化学习，三阶段：SFT → Reward Model → PPO
- **关键点**：为什么需要 RM？PPO 中的 clip 机制？KL 惩罚的作用？
- **常见追问**：RLHF 的问题有哪些？DPO 怎么解决的？
- **参考回答**：*待填充*
- **关联概念**：[[wiki/concepts/rlhf|RLHF]], [[wiki/concepts/ppo|PPO]]

### 3. PPO 算法
- **定义**：Proximal Policy Optimization，近端策略优化
- **关键点**：Clip 目标函数、GAE、Value function
- **常见追问**：PPO 和 TRPO 的区别？Clip 范围怎么选？
- **参考回答**：*待填充*

### 4. DPO vs PPO vs GRPO
- **定义**：三种不同的对齐/优化方法
- **关键点**：DPO 不需要 RM、GRPO 是 group relative 的方式
- **常见追问**：各自优缺点？什么场景用什么方法？
- **参考回答**：*待填充*
- **关联概念**：[[wiki/synthesis/rl-alignment-methods|RL对齐方法对比]]

### 5. 分布式训练
- **关键点**：数据并行、模型并行、Pipeline 并行、ZeRO
- **常见追问**：DeepSpeed 的 ZeRO Stage 1/2/3 区别？FSDP？
- **参考回答**：*待填充*

### 6. 推理优化
- **关键点**：KV-Cache、量化（INT8/INT4）、vLLM、PagedAttention
- **常见追问**：量化对精度的影响？Speculative Decoding？
- **参考回答**：*待填充*

## 🟡 中频考点

### 7. Tokenization
- **关键点**：BPE、SentencePiece、WordPiece
- **参考回答**：*待填充*

### 8. 位置编码
- **关键点**：绝对位置编码、RoPE、ALiBi
- **参考回答**：*待填充*

### 9. Loss Function 设计
- **关键点**：Cross-entropy、Label Smoothing、Focal Loss
- **参考回答**：*待填充*

### 10. SFT 数据工程
- **关键点**：数据质量 vs 数量、数据配比、数据去重
- **参考回答**：*待填充*

## 🟢 低频考点

### 11. MoE（Mixture of Experts）
- **参考回答**：*待填充*

### 12. 多模态大模型
- **参考回答**：*待填充*

---

## 复习记录
| 日期 | 掌握程度 | 备注 |
|------|----------|------|
| 2026-04-07 | 框架创建 | 开始填充内容 |
