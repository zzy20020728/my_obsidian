---
title: "PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training"
type: paper
tags: [RLVR, prefix-rollout, step-reward, agentic-search, multi-hop-QA, PPO, rollout-reuse, credit-assignment]
created: 2026-04-11
updated: 2026-04-11
sources: ["https://arxiv.org/abs/2604.03675"]
status: active
---

# PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training

## 一句话总结

从 agentic search 轨迹中提取 prefix states，在每个 prefix 处让模型生成 intermediate answer，用**相邻 prefix 的 answer quality 差异**作为 step-level reward，同时将 prefix-answer 对作为额外训练样本复用 rollout，单一共享模型同时做 search policy 和 prefix evaluation。

## 基本信息

- **作者**: Erhan Zhang†, Yiqun Chen†, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao‡
- **机构**: Renmin University of China, Xiaohongshu Inc., USC
- **发表日期**: 2026-04-04
- **arXiv**: 2604.03675

## 核心贡献

1. **Prefix-based rollout reuse**: 一条 T-turn 搜索轨迹 → T+1 个 prefix-answer 训练样本，大幅提升数据效率
2. **Adjacent prefix gains 作为 process reward**: $r_t^{proc} = \alpha(v_t - v_{t-1})$，将稀疏终端监督转化为 step-aligned feedback
3. **共享模型 joint optimization**: 同一个模型同时做 search policy 和 prefix answer evaluation，搜索能力和评估能力共同进化
4. **在 multi-hop QA 上 consistent improvement**: 5 个 benchmark 上全面超越 Search-R1、R1-Searcher、StepSearch、TIPS 等

## 方法

### 关键框架

```
Main Search Rollout → 提取 Prefix States {s_0, s_1, ..., s_T}
                           ↓
                    Prefix Answering: ỹ_t ~ π_θ^ans(·|s_t)
                           ↓
                    Answer Scoring: v_t = R(ỹ_t, y)
                           ↓
              Process Reward: r_t^proc = α(v_t - v_{t-1})
              Outcome Reward: r^out = R(ŷ, y)
                           ↓
                    Joint PPO Optimization
```

### 核心公式

**Prefix state 定义**：
$$s_t = (q, \{(c_{t'}, a_{t'}, o_{t'})\}_{t'=1}^{t})$$

其中 $c_t$ = reasoning, $a_t$ = search action, $o_t$ = search result。

**Adjacent prefix gain（核心 step reward）**：
$$r_t^{proc} = \alpha(v_t - v_{t-1}), \quad t = 1, \dots, T$$

$v_t = R(\tilde{y}_t, y)$ 是从 prefix $s_t$ 生成的 answer 与 GT 的匹配分数。

**Token-level reward assignment**：
$$\rho_k = \begin{cases} r_t^{proc}, & k = m_t \text{ (search turn end)} \\ r^{out}, & k = m_{ans} \text{ (final answer)} \\ 0, & \text{otherwise} \end{cases}$$

**PPO objective**（标准 clipped PPO）。

### 两种模式

- **Search mode**: $\pi_\theta^{search}(\cdot|q)$ — 多轮搜索
- **Answer mode**: $\pi_\theta^{ans}(\cdot|s_t)$ — 从 prefix 生成答案

同一个模型 $\pi_\theta$，不同 prompt template。

## 实验结果

### 主实验 (Qwen2.5-7B, Table 1)

| 方法 | NQ F1 | HotpotQA F1 | 2Wiki F1 | Bamboogle F1 | MuSiQue F1 | Avg F1 |
|------|-------|-------------|----------|--------------|------------|--------|
| Search-R1 | 40.98 | 51.24 | 47.10 | 48.43 | 22.50 | 42.05 |
| R1-Searcher | 41.15 | 55.55 | 50.64 | 54.62 | 27.74 | 45.94 |
| StepSearch | 39.79 | 50.02 | 44.35 | 50.32 | 27.50 | 42.40 |
| TIPS | 40.38 | 45.98 | 42.95 | 46.20 | 18.59 | 38.82 |
| **PRAISE** | **43.62** | **60.62** | **58.14** | **56.99** | **30.73** | **50.02** |

PRAISE 在所有 5 个 benchmark 上 F1 和 EM 都是最优。

### Ablation (Table 2)

| 变体 | HotpotQA F1 | Δ | 2Wiki F1 | Δ |
|------|------------|---|----------|---|
| PRAISE (full) | 60.62 | — | 58.14 | — |
| w/o joint optimization (policy model) | 57.16 | -3.46 | 57.81 | -0.33 |
| w/o joint optimization (frozen 7B) | 57.65 | -2.97 | 57.93 | -0.21 |
| w/o joint optimization (frozen 14B) | 57.57 | -3.05 | 58.85 | +0.71 |
| w/o process reward (α=0) | 58.80 | -1.82 | 58.64 | +0.50 |
| w/o prefix evaluator | 58.42 | -2.20 | 58.00 | -0.14 |

关键发现：
1. **Joint optimization >> frozen evaluator**：即使 14B frozen 也不如 7B joint，因为 evaluator 需要跟 policy 同步进化
2. **Process reward 有用**：去掉 α 后平均降 1-3 个点
3. **Prefix evaluator 有用**：去掉后也降 1-2 个点

### 为什么用 PPO 而非 GRPO

GRPO 需要对 same state 做 grouped sampling。在 multi-turn search 中，不同 turn 对应不同 state，做严格 same-state grouping 成本爆炸。PPO 更自然。

### 为什么不用 likelihood-based prefix scoring

TIPS 和 IGPO 用 GT answer 的 log-likelihood 做 prefix 评分。但直接优化 log-likelihood → 变成 SFT on reference answer → 鼓励 answer memorization 而非 incremental evidence value。PRAISE 用 rollout-based generation + task reward 评分，保留了 process reward 的"边际贡献"语义。

## 🔴 与 SPC 研究的关系

### PRAISE 是 SPC 最接近的竞品

| 维度 | PRAISE | SPC |
|------|--------|-----|
| **框架** | 从 prefix 做 rollout → 得到 answer → 算 step reward | 从 prefix 做 short probe → 得到 answer → 与 final answer 比 |
| **答案比对对象** | GT answer (verifiable reward) | 轨迹自身的 final answer（无监督） |
| **reward 定义** | 相邻 prefix 的 answer quality 差 | 当前步 probe answer 与 final answer 的语义一致性 |
| **step 粒度** | Search turn 级别（multi-hop QA 的 turn） | Reasoning step 级别（数学推理的 CoT step） |
| **任务领域** | Agentic search / multi-hop QA | 数学推理 |
| **优化算法** | PPO（需要 value head） | GRPO（更适合数学推理的 grouped sampling） |
| **模型角色** | 共享模型做 search + evaluation | 同一模型做 generation + probing |
| **需要 GT?** | **是**（verifiable answer scoring） | **否**（与自身 final answer 对比） |
| **semantic equivalence** | 无（直接用 R(ŷ,y) 得分） | 有（核心技术贡献） |

### 关键差异化（SPC 相对 PRAISE 的优势）

1. **SPC 无需 GT**：这是最大差异。PRAISE 的 prefix scoring 完全依赖 $R(\tilde{y}_t, y)$，即 GT answer。SPC 与轨迹自身 final answer 对比，面向 URLVR。
2. **SPC 用 semantic equivalence**：PRAISE 用 exact/F1 match 比对。SPC 可以捕捉语义等价但形式不同的答案。
3. **SPC 面向数学推理**：粒度更细（reasoning step vs search turn），更直接面向 CoT credit assignment。
4. **SPC 做 short probe 而非 complete answer generation**：PRAISE 需要模型为每个 prefix 完整生成答案（开销大），SPC 只需短续写到能抽取答案。

### 需要在论文中如何引用 PRAISE

PRAISE 必须作为 **Related Work 中最重要的 baseline** 之一。建议的引用口径：

> PRAISE (Zhang et al., 2026) extracts prefix states from agentic search trajectories and derives step-level rewards from adjacent prefix answer quality gains. While sharing the intuition of "probing intermediate states to assess step contribution," PRAISE operates in the supervised RLVR setting with verifiable answer scoring, at the search-turn granularity, and for agentic multi-hop QA. In contrast, SPC operates in the unsupervised setting (no GT), at the reasoning-step granularity, uses semantic equivalence judgment, and targets mathematical reasoning with GRPO optimization.

## 面试 Q&A

- Q: PRAISE 的 step reward 是怎么算的？为什么用相邻 prefix 差值？🟡
- A: 从每个 prefix state 生成 intermediate answer，与 GT 比对得分 v_t。Step reward = α(v_t - v_{t-1})，即相邻 prefix 的 answer quality 差。这捕捉了每个 search turn 的 **marginal contribution**：如果这一轮搜索让 answer 变好了，正 reward；变差了，负 reward。比直接用 final outcome reward 更精细。

- Q: 为什么 joint optimization 比 frozen evaluator 好？🟡
- A: Frozen evaluator 无法跟上 policy 进化——随训练推进，policy 产生的搜索轨迹分布变化，frozen evaluator 对新类型的 prefix 评估不准确。Joint optimization 让 evaluator 和 policy 同步更新，评估信号始终与当前 policy 匹配。这与 Co-Evolving Verifier 的核心思想一致。
