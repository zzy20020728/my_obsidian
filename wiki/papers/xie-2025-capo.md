---
title: "CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment"
type: paper
tags: [credit-assignment, GenPRM, LLM-as-judge, step-level-reward, token-level-reward, RLVR, asymmetric-reward, voting-mechanism]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2508.02298, https://github.com/andyclsr/CAPO]
status: active
---

# CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment

## 基本信息
- **作者**: Guofu Xie*, Yunsheng Shi*, Hongtao Tian, Ting Yao, Xiao Zhang, Jun Xu (*equal contribution)
- **机构**: Renmin University of China (人民大学高瓴AI学院), Tencent WeChat Search (腾讯微信搜索)
- **年份**: 2025 (arXiv: 2508.02298)
- **代码**: https://github.com/andyclsr/CAPO
- **Base Model**: Llama-3-1B/3B-Instruct, Qwen2.5-1.5B/7B-Base

## 一句话总结
> 用 off-the-shelf 大模型作为生成式 PRM（LLM-as-GenPRM），单次推理生成所有步骤的正误判定，以可验证的 token-level 奖励替代 RLVR 中粗粒度的 outcome-only 信号，并提出非对称奖励塑形平衡 outcome 与 process 信号。

## 核心贡献

1. **识别 RLVR credit assignment 问题**：GRPO 等方法给所有 token 分配相同 outcome reward，无法区分"答对但过程有错" vs "答对且过程全对"
2. **LLM-as-GenPRM**：用 70B 级别的通用 LLM 单次推理（single pass）生成所有步骤的确定性二值判定，无需训练辅助 PRM
3. **多种 Voting 机制**：Intersection（保守高精度）/ Majority（平衡）/ Union（高召回）/ Average / Greedy，多次 critique 投票降噪
4. **非对称奖励塑形**：分析 outcome-oriented 与 process-oriented 信号的冲突，提出 $W_{\text{whole}} > W_{\text{process}}$ 的层次化奖励

## 方法

### Token-level Reward 公式（核心）

$$R_t^i = r_v \cdot W_{\text{whole}} - \mathbb{I}(t \in T_{\text{err}}^i) \cdot W_{\text{process}}$$

其中：
- $r_v$: rule-based verifier 的 outcome reward (0/1)
- $W_{\text{whole}}$: outcome 权重（默认 2）
- $W_{\text{process}}$: process penalty 权重（默认 1）
- $T_{\text{err}}^i$: rollout $o_i$ 中所有错误步骤的 token 索引集合

### 奖励分层效果

| 情况 | Token Reward |
|------|-------------|
| 答案正确 + 过程正确 | +2 |
| 答案正确 + 过程有错步 | +1 |
| 答案错误 + 过程正确 | 0 |
| 答案错误 + 过程有错步 | -1 |

### Pipeline

1. Policy model 采样 n=16 个 rollout
2. Rule-based verifier 获取 outcome reward
3. LLM-as-GenPRM（如 Qwen2.5-72B-Instruct）生成 k=4 个 critique
4. Voting 机制聚合得到最终错误步骤集合
5. 映射错误步骤到 token 索引
6. 计算 per-token reward → per-token advantage → PPO-clip 更新

### Voting 策略

- **Intersection**: $S_i^{\cap} = \bigcap_{j=1}^k S_{i,j}$ — 所有 critique 一致认为有错才标记
- **Majority**: $S_i^{\text{maj}} = \{s \mid \text{count}(s) \geq k/2\}$
- **Union**: $S_i^{\cup} = \bigcup_{j=1}^k S_{i,j}$ — 任一 critique 认为有错就标记

**关键发现**：小模型（1B/1.5B）用 Intersection 最优（需要高精度信号），大模型（7B）用 Majority/Union 最优（允许更多探索）。

## 实验结果

### 主要结果（vs GRPO-Rule baseline）

| 模型 | Math Mean | OOD Mean | All Mean | vs GRPO-Rule |
|------|-----------|----------|----------|-------------|
| Qwen2.5-7B | **34.8** | 41.4 | **37.0** | +2.3 |
| Llama-3-3B | **23.7** | 41.4 | **30.4** | +1.8 |
| Llama-3-1B | 17.6 | **21.8** | **18.8** | +2.9 |
| Qwen2.5-1.5B | **30.5** | 35.5 | **32.3** | +1.0 |

亮点：Qwen2.5-7B AIME24 从 3.6 提升到 **10.8**（+200%），Llama-3-3B AMC23 从 16.9 到 **26.8**（+59%）。

### Ablation: 非对称权重

| 配置 | All Mean |
|------|----------|
| W=2, P=1 (非对称) | **29.4** |
| W=2, P=0.1 (弱 process) | 28.8 |
| W=2, P=2 (对称) | 28.5 |
| W=2, P=5 (process 主导) | 27.4 |

验证了 $W_{\text{whole}} > W_{\text{process}}$ 最优。Process 权重过大会导致 outcome 和 process 信号冲突。

### Voting Scaling

更多 critique (N=2→4→8) 带来持续提升，体现 test-time scaling 特性。

## 与 URLVR/SPC 研究的关系

### Ground-truth 依赖分析
- **当前实现依赖 GT answer**：GenPRM prompt 包含 ground-truth answer，将任务从"独立判断"简化为"给定答案的错误定位"
- **但 Table 9 显示 w/o GT 和 w/ GT 性能接近**：GenPRM 有一定独立判断能力
- **可改造为无监督版本**：不提供 GT，让 GenPRM 独立判断逻辑一致性和计算正确性

### 对 SPC 方案的启发
1. **确定性二值判断 > 概率估计**：与 SPC 的 semantic equivalence 判断理念一致
2. **非对称奖励塑形可直接迁移**：SPC 的 step reward 也需要与 TTRL 的 outcome reward 平衡
3. **Voting 降噪策略**：SPC 的多次 probing 也可通过 voting 聚合来降低 estimation noise
4. **小模型用严格阈值、大模型用宽松阈值**：对 SPC 的 confidence threshold 设置有指导意义

### 关键区别
- CAPO 用外部强模型（70B）做 judge，SPC 用 policy model 自身
- CAPO 需要 outcome-level GT，SPC 目标是完全无监督

## 局限性

1. **依赖 70B+ 外部模型**：计算开销大，每个 rollout 都需要调用 GenPRM
2. **依赖 GT answer**：限制了在纯无监督场景的应用
3. **Step 分割依赖 `\n\n` 格式**：对非标准格式的模型不适用
4. **仅在数学推理上验证**
5. **未与 VinePPO 直接对比**

## 面试关联

- 🔴 **Credit assignment**：RL 中的核心问题，CAPO 的 token-level 方案
- 🟡 **LLM-as-judge**：用大模型评估小模型的方法论
- 🟡 **Reward shaping**：outcome vs process reward 的平衡

## 相关页面
- [[wiki/concepts/ppo|PPO/GRPO]] — credit assignment 基础
- [[wu-2026-spae|SPAE]] — 另一种 step-level reward 构造方法
- [[wiki/synthesis/spc-experiment-plan|SPC 实验设计方案]] — CAPO 的非对称奖励设计可迁移
