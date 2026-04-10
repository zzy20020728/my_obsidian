---
title: "DistriTTRL: From the Inside Out — Progressive Distribution Refinement for Confidence Calibration"
type: paper
tags: [URLVR, TTRL, confidence-distribution, progressive-reward, reward-hacking, diversity-penalty, GMM, voting, test-time-training]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2603.16500]
status: active
---

# DistriTTRL: 基于置信度分布先验的渐进式伪标签优化

## 基本信息
- **作者**: Xizhong Yang, Yinan Xia, Huiming Wang, Mofei Song
- **机构**: Southeast University（东南大学）, Kuaishou Technology（快手）, SUTD
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.16500v1)
- **链接**: https://arxiv.org/abs/2603.16500
- **代码**: https://github.com/yxizhong/DistriTTRL
- **Base Models**: Qwen2.5-7B-Base, Qwen2.5-Math-7B/1.5B, Llama-3.1-8B-Instruct, Qwen3-8B

## 一句话总结
> 利用模型置信度的**分布先验**（GMM 建模）跨步渐进式构建更准确的伪标签，并通过**多样性惩罚**缓解投票策略导致的一致性 reward hacking，在多个模型和基准上超越 TTRL。

## 摘要

在无标注 Test-Time Training (TTT) 中，现有方法（如 [[wiki/papers/zuo-2025-ttrl|TTRL]]）依赖投票策略生成伪标签作为 reward，但存在两大问题：(1) 训练过程中参数动态更新导致不同步的置信度分布偏移，单步 rollout 信息不充分；(2) 基于投票的 TTS 策略在可训练参数场景下引发**一致性 reward hacking**——模型倾向输出一致但不一定正确的答案以获得更高奖励。DistriTTRL 提出：利用 GMM 对模型置信度建模，跨训练步渐进式构建分布先验并校正分布偏移，同时设计多样性惩罚机制来抑制 reward hacking。

## 核心贡献

1. **渐进式置信度分布构建**：跨步记录所有 rollout 的置信度，通过 GMM 建模正/负分布，并对历史步进行分布偏移校正（shift correction），累积更丰富的分布先验
2. **基于分布先验的伪标签估计**：利用全局正/负高斯分布将每个 query 的 rollout 划分为正/负子集，通过过滤+投票生成更准确的伪标签
3. **多样性惩罚（Diversity-Targeted Penalty）**：度量每个 query rollout 的答案多样性，对低多样性（趋同于一致性 hacking）的 query 降低 advantage 权重
4. **首次系统分析**投票策略在 TTT 中导致 reward hacking 的机制（majority ratio 快速上升但准确率停滞）

## 方法

### 1. 置信度建模

对每条 trajectory 计算置信度（基于 answer 部分 token 的 top-k 负对数概率）：

$$C_{\text{traj}} = -\frac{1}{N_G \times k} \sum_{i \in G} \sum_{j=1}^{k} \log \mathbf{P}_i(j)$$

其中 $G$ 是答案部分 token 的子集，$k$ 是 top-k 概率数量。

### 2. GMM 分布建模

对每步所有 query 的 rollout 置信度用两组分 GMM 建模：

$$p(x) = \pi_1 \mathcal{N}(x|\mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(x|\mu_2, \sigma_2^2)$$

高均值组分 $\mathcal{N}(\mu_{\text{pos}}, \sigma_{\text{pos}}^2)$ 代表正确推理路径，低均值组分代表错误推理路径。

### 3. 渐进式分布构建与偏移校正

维护全局变量 $\mathcal{C} \in \mathbb{R}^{S \times B \times G}$（步数 × batch × 采样数）。对历史步 $s < k$ 计算偏移量并校正：

$$\Delta_{s \to k} = \frac{\mu_{\text{pos}}^k + \mu_{\text{neg}}^k}{2} - \frac{\mu_{\text{pos}}^s + \mu_{\text{neg}}^s}{2}$$

$$\tilde{\mathcal{C}}_{s,\cdot,\cdot} = \mathcal{C}_{s,\cdot,\cdot} + \Delta_{s \to k}$$

**关键观察**：训练过程中置信度分布逐渐右移，偏移量在约 step 100 时趋近于零。

聚合所有历史校正后的置信度与当前步的置信度，拟合全局 GMM，用于指导当前步每个 query 的正/负划分。

### 4. 伪标签估计流程

1. 用全局 GMM 将 query $q_i$ 的 rollout 划分为正子集 $X_{\text{pos}}^{k,i}$ 和负子集 $X_{\text{neg}}^{k,i}$
2. 对负子集投票得到最可能的错误答案 $A_{\text{neg}}^{k,i}$
3. 从正子集中过滤掉产生该错误答案的样本
4. 对过滤后的正子集投票得到最终伪标签 $A_{\text{final}}^{k,i}$

### 5. 多样性惩罚

度量每个 query 的答案多样性（唯一答案数量）：

$$D(q_i) = |\{o_{i,j}: j=1,\dots,G\}_{\text{unique}}|$$

通过 softmax 归一化后作为 advantage 的权重：

$$\hat{A}_{i,j}' = \hat{A}_{i,j} \cdot \mathcal{D}(q_i)$$

其中低多样性（$D(q_i) \leq \tau \cdot G$, $\tau=0.1$）的 query 收到降低的权重，高多样性 query 权重为 1。

**目的**：防止模型通过输出一致答案来"骗取"奖励——当所有 rollout 趋同时，降低其对策略更新的贡献。

## 实验结果

### 主实验（16 repeats 评估）

| Model | Method | AIME2024 | AMC | MATH-500 | Avg. |
|-------|--------|----------|-----|----------|------|
| Qwen2.5-7B-Base | TTRL | 22.08 | 52.26 | 80.38 | 51.57 |
| | DistriTTRL-GMM | **23.54** (+1.46) | **56.48** (+4.22) | **81.46** (+1.08) | **53.83** (+2.26) |
| Qwen2.5-Math-7B | TTRL | 25.83 | 56.18 | 81.51 | 54.51 |
| | DistriTTRL-GMM | **33.33** (+7.50) | **61.37** (+5.19) | 81.83 (+0.32) | **58.84** (+4.33) |
| Qwen2.5-Math-1.5B | TTRL | 14.14 | 44.38 | 72.19 | 43.57 |
| | DistriTTRL-GMM | **14.55** (+0.41) | **45.37** (+0.99) | 72.31 (+0.12) | **44.08** (+0.51) |
| Llama-3.1-8B-Inst | TTRL | 7.92 | 30.27 | 58.43 | 32.21 |
| | DistriTTRL-GMM | **10.21** (+2.29) | 30.87 (+0.60) | 58.96 (+0.53) | **33.35** (+1.14) |
| Qwen3-8B | TTRL | 33.00 | 65.95 | 87.25 | 62.07 |
| | DistriTTRL-GMM | 33.44 (+0.44) | **70.31** (+4.36) | 87.21 (-0.04) | **63.65** (+1.58) |

**关键发现**：
- 在 Qwen2.5-Math-7B 上提升最显著：AIME2024 +7.50，AMC +5.19
- Qwen3-8B 上 AMC +4.36，但 MATH-500 几乎无变化（已接近 GT 上限）
- **Reward hacking 缓解**：使用多样性惩罚后，majority ratio 的增长速度明显减缓，准确率持续提升而非过早收敛

### 消融实验

1. **Progressive Distribution** 贡献显著（聚合历史信息优于单步）
2. **Diversity Penalty** 有效防止 majority ratio 过快饱和
3. **分布偏移校正**对长训练周期尤为关键

## 与 SPC/URLVR 研究的关系

### 直接关联
1. **置信度分布建模思路与 SPC 互补**：DistriTTRL 用 GMM 对 trajectory-level 置信度建模来区分正/负推理路径；SPC 则在 step-level 用 probing 来评估语义一致性。两者可以结合——用 SPC 的 step-level 信号指导 DistriTTRL 的 trajectory-level 分布先验
2. **Reward Hacking 分析为 Co-Evolving Verifier 提供动机**：DistriTTRL 详细分析了投票策略的 reward hacking 机制（majority ratio 快速饱和），这正是 Co-Evolving Verifier 需要解决的核心问题——verifier 需要能识别"一致但错误"的伪答案
3. **渐进式分布校正的思想可借鉴**：SPC 信号可能也需要类似的跨步校正机制，因为模型能力随训练提升，早期和晚期的 semantic consistency 阈值应有所不同

### 启发点
- DistriTTRL 的多样性惩罚可以作为 SPC 中的一个额外信号维度：答案多样性低的 query 可能需要更谨慎的 credit assignment
- Co-Evolving Verifier 可以参考 DistriTTRL 的 GMM 置信度划分来初始化 verifier 的正/负样本判别

## 面试 Q&A

### Q1: DistriTTRL 如何解决 TTRL 中的 reward hacking 问题？🔴
**A**: TTRL 的投票策略在可训练参数场景下导致 consistency reward hacking——模型学会输出一致答案（不管对错）以获高分。DistriTTRL 的解决方案是 **diversity-targeted penalty**：(1) 计算每个 query rollout 的唯一答案数量作为多样性度量；(2) 对低多样性 query（$D(q_i) \leq \tau \cdot G$）通过 softmax 归一化赋予较小的 advantage 权重；(3) 高多样性 query 保持权重为 1，不影响正常学习。效果：majority ratio 增长减缓，模型不会过早收敛到一致错误答案。

### Q2: 为什么需要跨步的分布偏移校正？GMM 在这里的作用是什么？🔴
**A**: RL 训练过程中模型参数持续更新，导致不同步的 rollout 置信度分布存在偏移（实验观察到分布逐渐右移）。如果直接聚合历史置信度会引入系统偏差。校正方法：对每个历史步 $s$ 的分布和当前步 $k$ 的分布分别拟合 GMM，计算正负组分均值的中心偏移量 $\Delta_{s \to k}$，加到历史置信度上。GMM 的作用有两个：(1) 建模正确/错误推理路径的双峰分布，提供全局先验指导每个 query 的样本划分；(2) 通过累积更多样本使分布估计更稳定，克服单步采样不足的问题。

### Q3: 对比 TTRL 的简单 majority voting，DistriTTRL 的伪标签估计有何优势？🟡
**A**: TTRL 仅对单步 rollout 做简单多数投票，受限于 (1) 采样数量有限（通常 32-64），(2) 无法利用历史信息。DistriTTRL 改进点：(1) 用置信度加权投票代替等权投票，高置信度 trajectory 的投票权更大；(2) 跨步累积 rollout 信息，通过偏移校正后聚合，等效于在更大样本上投票；(3) GMM 先验将 rollout 先划分为正/负子集，先识别并排除最可能错误的答案，再在高质量子集上投票。这种 "先筛后投" 的策略显著提升伪标签准确率。
