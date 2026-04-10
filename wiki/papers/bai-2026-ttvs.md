---
title: "TTVS: Boosting Self-Exploring Reinforcement Learning via Test-time Variational Synthesis"
type: paper
tags: [URLVR, TTRL, test-time-adaptation, variational-synthesis, data-augmentation, hybrid-exploration, IGE, CGE, consistency, GRPO]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2604.08468]
status: active
---

# TTVS: 通过测试时变分合成增强自探索强化学习

## 基本信息
- **作者**: Sikai Bai, Haoxi Li, Jie Zhang*, Yongjiang Liu, Song Guo*
- **机构**: The Hong Kong University of Science and Technology（香港科技大学）
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2604.08468v1)
- **链接**: https://arxiv.org/abs/2604.08468
- **Base Models**: Qwen3-1.7B/4B/8B, Qwen2.5-Instruct-1.5B/3B/7B, LLaMA-3.2-3B, LLaMA-3.1-8B

## 一句话总结
> 提出 Test-Time Variational Synthesis（TTVS）——在 test-time 动态合成语义等价的变体查询来扩充训练数据，结合**组内探索（IGE）**和**跨组探索（CGE）**的混合策略，仅用无标注测试数据即**超越有监督 RLVR** 方法的性能。

## 摘要

RLVR 依赖大规模标注数据集，在专业领域难以获取监督信号。现有 test-time 方法（如 [[wiki/papers/zuo-2025-ttrl|TTRL]]）虽能在无标注数据上训练，但受限于静态测试集，容易过拟合表面文本模式。TTVS 提出两个协同模块：(1) **Online Variational Synthesis**——利用模型自身将静态测试查询动态转化为语义等价但表述不同的变体，迫使模型学习底层问题逻辑而非表面模式；(2) **Test-time Hybrid Exploration**——在变体数据上平衡精度驱动的利用（Intra-Group Exploration, IGE）和一致性驱动的探索（Cross-Group Exploration, CGE）。TTVS 在 8 种模型架构上均取得优异性能，仅用无标注数据即超越使用大规模标注数据训练的 SOTA RL 方法。

## 核心贡献

1. **动态数据增强突破静态测试集局限**：首次在 test-time RL 中引入在线变分合成，将静态查询转化为动态训练流，避免过拟合表面模式
2. **IGE + CGE 混合探索策略**：IGE 在每个 query 内独立做 GRPO 更新（精度导向），CGE 在语义等价 query cluster 间做混合投票和更新（一致性导向），二者协同提升泛化性
3. **无标注超越有监督**：在 Qwen2.5-Instruct-1.5B 上，TTVS（无标注）平均 33.4% 超过 DeepSeek-R1-Distill-7B（800K 标注数据）的 30.5%
4. **算法无关**：兼容 GRPO、OPO、DAPO 等多种 RL 优化算法

## 方法

### 1. Online Variational Synthesis

#### 伪标签生成
对原始查询 $q$ 采样 $N$ 个 rollout，通过 majority voting 生成伪标签 $y_q^*$：

$$\{o_1, \dots, o_N\} \sim \pi_\theta(\cdot | q)$$
$$y_q^* = \text{MajorityVote}((a_i)_{i=1}^N)$$

#### 变分数据合成
利用策略模型自身生成 $k$ 个语义等价变体查询：

$$\{q'_1, q'_2, \dots, q'_k\} \sim \pi_\theta(\cdot | \mathcal{P}, q, y_q^*)$$

其中 $\mathcal{P}$ 是指导模型改写问题的特定 prompt。生成的变体与原始查询共享相同答案 $y_q^*$，但表述方式不同。

#### 在线过滤
只保留满足以下条件的 query cluster：
1. **难度范围过滤**：原始 query 的 group accuracy 必须在 $[\tau_{\text{low}}, \tau_{\text{high}}]$ 范围内（默认 $[0.125, 0.875]$）

$$\text{acc}(q) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(a_i = y_q^*) \in [\tau_{\text{low}}, \tau_{\text{high}}]$$

2. **长度限制**：合成查询不超过最大 token 长度 $L_{\text{max}} = 1024$

过滤确保模型聚焦于不太简单也不太难的问题，避免无效训练。

### 2. Test-time Hybrid Exploration

#### Intra-Group Exploration (IGE) — 精度驱动
对 $\mathcal{D}_{\text{batch}}$ 中的每个 query（包括原始和变体），独立执行标准 GRPO 更新：
- 对 $\tilde{q}$ 采样 $N$ 个 rollout
- 组内 majority voting 生成 $y_{\tilde{q}}^*$
- 计算基于 $R(o_i, y_{\tilde{q}}^*)$ 的 advantage
- 独立执行 GRPO 策略更新

**目的**：将每个 query 视为独立优化目标，推动模型找到该特定表述下的正确答案。

#### Cross-Group Exploration (CGE) — 一致性驱动
在语义等价的 query cluster $\{q, q'_1, \dots, q'_k\}$ 之间操作：
- 从 cluster 中每个 query 各采样 $N/(k+1)$ 个 rollout，混合为一个池：

$$\mathcal{O}_{\text{mix}} = \bigcup_{j=0}^{k} \left\{ \text{Sample}_{i=1}^{N/(k+1)} \left( o_i \sim \pi_\theta(\cdot | q'_j) \right) \right\}$$

- 对混合池执行**跨组 majority voting** 得到联合伪标签 $y_{\text{mix}}^*$
- 所有 rollout 基于该统一标签计算 advantage 并更新

**目的**：强制模型在不同表述的同一问题上保持一致的推理过程，正则化策略以学习不变的问题逻辑。

### 实现细节

| 参数 | 值 |
|------|-----|
| 优化器 | GRPO (默认), 兼容 OPO/DAPO |
| 学习率 | $5 \times 10^{-7}$ (cosine schedule) |
| Rollout 数量 | 32 per query |
| 采样温度 | 0.6 |
| 最大生成长度 | 3072 tokens |
| $\tau_{\text{low}}$ / $\tau_{\text{high}}$ | 0.125 / 0.875 |
| $L_{\text{max}}$ | 1024 |
| IGE warmup | 40 steps |
| CGE warmup | 60 steps |
| GPU | 8× NVIDIA H800 80GB |

## 实验结果

### 主实验（pass@1, 16 candidates, temp=0.6）

| Model | Method | MATH500 | AIME2024 | AMC2023 | GPQA | Avg. |
|-------|--------|---------|----------|---------|------|------|
| Qwen3-1.7B-Inst | Base | 59.4 | 3.3 | 28.9 | 13.1 | 26.2 |
| | TTRL | 73.4 | 15.2 | 54.2 | 22.2 | 41.3 |
| | **TTVS** | **80.2** | **23.3** | **61.4** | **26.3** | **47.5** (+21.3) |
| Qwen3-4B | Base | 64.0 | 10.0 | 26.5 | 26.3 | 31.7 |
| | TTRL | 85.0 | 26.7 | 61.4 | 43.0 | 54.0 |
| | **TTVS** | **90.3** | **36.7** | **71.1** | **48.9** | **61.5** (+29.8) |
| Qwen3-8B | Base | 82.2 | 26.9 | 57.8 | 48.0 | 53.7 |
| | TTRL | 89.2 | 46.7 | 68.6 | 53.0 | 65.4 |
| | **TTVS** | **92.6** | **50.0** | **72.3** | **56.1** | **67.8** (+14.1) |
| Qwen2.5-7B-Inst | Base | 70.4 | 10.2 | 41.0 | 36.4 | 39.5 |
| | TTRL | 77.6 | 13.3 | 44.6 | 48.5 | 46.0 |
| | **TTVS** | **79.4** | **16.7** | **47.2** | **51.5** | **48.7** (+9.2) |

**核心数据**：
- Qwen3-4B 上 TTVS 相比 TTRL 的绝对增益：MATH500 +5.3, AIME2024 +10.0, AMC2023 +9.7, GPQA +5.9
- 超越有监督方法：TTVS (Qwen2.5-1.5B, 无标注) 平均 33.4% > DeepSeek-R1-Distill-7B (800K 标注) 30.5%

### 消融实验

| Method | MATH500 | AMC2023 | GPQA |
|--------|---------|---------|------|
| Qwen3-4B Base | 64.0 | 26.5 | 26.3 |
| CGE only | 86.4 | 62.6 | 43.9 |
| IGE only | 88.4 | 69.8 | 46.6 |
| **TTVS (IGE+CGE)** | **90.3** | **71.1** | **48.0** |

- IGE 贡献更大（精度提升），CGE 提供一致性正则化
- 两者互补，组合效果最优
- IGE 在训练过程中保持更高的 entropy（更好的探索能力）

### 不同 RL 算法兼容性

| Method | MATH500 | AIME2024 | AMC2023 |
|--------|---------|----------|---------|
| TTVS (w/ GRPO) | **90.3** | **36.7** | 71.1 |
| TTVS (w/ OPO) | 89.6 | 33.3 | 69.8 |
| TTVS (w/ DAPO) | 88.8 | **36.7** | **71.6** |

三种 RL 算法下性能接近，验证 TTVS 的通用性。

### 难度分级分析（MATH-500, Qwen3-4B）

| Level | Base | TTRL | TTVS |
|-------|------|------|------|
| L1 (最易) | - | 93.0 | 95.3 |
| L5 (最难) | 39.5 | 60.8 | **69.5** |

**关键洞察**：TTVS 与 TTRL 的性能差距随难度增加而扩大（L5 上 +8.7%），说明变分合成机制在难题上更有效——可能因为难题的 pattern 更多样，数据增强带来的泛化增益更大。

### 计算开销
- GPU 显存与 TTRL（32 rollouts）几乎持平
- 推理阶段 TTVS 生成更简洁的答案（1865 vs 1926 tokens）
- 相比 TTRL (64 rollouts)，TTVS (32 rollouts) 显存更低但准确率更高

## 与 SPC/URLVR 研究的关系

### 直接关联
1. **数据增强思想与 SPC 正交互补**：TTVS 从数据侧解决 TTRL 的过拟合问题（合成语义等价变体），SPC 从 reward 侧解决（step-level credit assignment）。两者可以叠加——在 TTVS 合成的多样化数据上应用 SPC 的精细 credit assignment
2. **Cross-Group Exploration 的一致性信号启发 SPC 设计**：CGE 的核心思想是"语义等价问题应产生一致答案"——这与 SPC 的 Semantic Process Consistency 理念高度契合。SPC 可以借鉴 CGE 的思路：在 step-level 检查模型对语义等价推理步骤的一致性
3. **在线过滤机制对 Co-Evolving Verifier 的启发**：TTVS 的难度过滤 $[\tau_{\text{low}}, \tau_{\text{high}}]$ 确保模型训练在"可学习"的样本上。Co-Evolving Verifier 也需要类似的课程策略——verifier 应聚焦于模型当前能力边界附近的样本

### 潜在整合
- **SPC + TTVS 组合方案**：用 TTVS 做数据增强（扩展训练分布），再用 SPC 做精细化 reward（step-level credit），双管齐下提升 URLVR 训练效果
- CGE 的跨组一致性检查可以为 SPC 的 semantic consistency 提供 query-level 的 ground truth 信号

## 面试 Q&A

### Q1: TTVS 如何解决 TTRL 的过拟合问题？其关键创新是什么？🔴
**A**: TTRL 在固定的静态测试集上训练，模型可能过拟合到文本的表面模式而非底层逻辑。TTVS 的创新在于 **Online Variational Synthesis**：利用模型自身将每个测试查询动态转化为多个语义等价但表述不同的变体（同答案、不同描述）。这迫使模型学习 invariant 的问题逻辑。具体流程：(1) 对原始 query majority voting 获得伪标签；(2) 用策略模型自身重写问题，生成 $k$ 个变体；(3) 在线过滤保证质量。再配合 **Hybrid Exploration**（IGE 精度驱动 + CGE 一致性驱动）的双模式更新，既优化单个 query 的准确率，又确保跨语义等价 query 的一致性。

### Q2: 为什么 TTVS 能超越使用大量标注数据的有监督 RLVR 方法？🔴
**A**: 有两个原因：(1) **Test-time adaptation advantage**——TTVS 直接在目标分布（测试数据）上训练，而有监督方法在通用训练集上训练后需要泛化到测试分布，存在 distribution gap；(2) **Dynamic data augmentation**——通过变分合成，TTVS 等效于在远多于原始测试集的数据量上训练，且合成数据与目标问题高度相关。这说明在特定测试场景下，"少量高相关性无标注数据 + 动态增强"可以超越"大量通用标注数据"，这对专业领域尤其有意义。

### Q3: IGE 和 CGE 各自的角色是什么？为什么需要两者结合？🟡
**A**: IGE（Intra-Group Exploration）在每个 query 内独立做 GRPO 更新，是精度驱动的——它推动模型找到每个具体问题的正确答案。CGE（Cross-Group Exploration）在语义等价 query cluster 之间做混合投票和更新，是一致性驱动的——它正则化模型使其推理过程对问题表述的变化保持不变。单独用 IGE 可能导致模型对每个 query 的特定表述过拟合，单独用 CGE 可能牺牲单个 query 的精度。两者互补：IGE 保证精度，CGE 保证泛化。消融实验证实组合效果（90.3%）优于任一单独组件（IGE: 88.4%, CGE: 86.4%）。
