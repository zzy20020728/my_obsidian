---
title: "Rethinking Exploration in RLVR: From Entropy Regularization to Refinement via Bidirectional Entropy Modulation"
type: paper
tags: [RLVR, GRPO, entropy, exploration, asymmetric-modulation, informative-entropy, spurious-entropy]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2604.04894]
status: active
---

# AsymGRPO: 非对称 Group-Relative Policy Optimization

## 基本信息
- **作者**: Hengrui Gu, Xiaotian Han, Yujing Bian, Kaixiong Zhou
- **机构**: North Carolina State University, Case Western Reserve University
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2604.04894v1 [cs.CL])
- **链接**: https://arxiv.org/abs/2604.04894

## 一句话总结
> 通过推导 GRPO 的参数化形式，揭示策略熵可被概念性分解为 **informative entropy**（保留多样推理路径）和 **spurious entropy**（侵蚀推理模式），提出 AsymGRPO 显式解耦正/负 rollout 的调制强度，实现精准的 entropy refinement，在 Qwen3-4B 上 5 个数学推理 benchmark 平均 60.32%（+3.82% over GRPO）。

## 摘要
RLVR 面临 **restricted exploration** 的根本限制——策略快速收敛到狭窄解集。传统 entropy regularization 对 LLM 效果不佳（超参敏感、仅边际提升）。本文重新审视策略 entropy 与探索的关系：通过推导 group-relative advantage 的连续参数化形式并分析 entropy 动态，将策略 entropy 概念性分解为：
- **Informative entropy**: 保留多样推理路径，推动性能提升
- **Spurious entropy**: 侵蚀推理模式，引入噪声

GRPO 本身隐含 **entropy refinement** 机制：在正 rollout 上维持 informative entropy，在负 rollout 上抑制 spurious entropy。基于此，提出 AsymGRPO，解耦正负 rollout 的调制，实现独立控制。

## 核心贡献
1. **参数化 Advantage 族**: 提出 β-参数化的 advantage 函数族 $A^{(\beta)}(p)$，统一 REINFORCE (β=0) 和 GRPO (β=0.5)，为分析提供连续控制旋钮
2. **Entropy 二元分解**: 首次将策略 entropy 分解为 informative 和 spurious 两类，揭示 GRPO 的隐式 entropy refinement 机制
3. **双向 Entropy 调制理论**: 通过 covariance 分析证明 GRPO 在正 rollout 上对抗 entropy 下降（维持 informative entropy），在负 rollout 上加速 entropy 下降（剪除 spurious entropy）
4. **AsymGRPO 框架**: 解耦 β_pos 和 β_neg，实现非对称调制，允许独立调控探索保持和噪声抑制的强度

## 方法详解

### 1. RLVR 形式化与参数化 Advantage
标准 RLVR 目标：
$$\max_\theta \mathcal{J}_{\mathrm{RLVR}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}[r(x,y)]$$

在 binary reward $r \in \{0,1\}$ 下，GRPO 的 advantage 可表示为组内准确率 $p$ 的函数：
$$A_{\mathrm{pos}}^{\mathrm{GRPO}}(p) = \sqrt{\frac{1-p}{p}}, \quad A_{\mathrm{neg}}^{\mathrm{GRPO}}(p) = -\sqrt{\frac{p}{1-p}}$$

**β-参数化族**:
$$A_{\mathrm{pos}}^{(\beta)}(p) = \left(\frac{1-p}{p}\right)^\beta, \quad A_{\mathrm{neg}}^{(\beta)}(p) = -\left(\frac{p}{1-p}\right)^\beta$$

- β=0: REINFORCE（固定 ±1）
- β=0.5: 标准 GRPO
- β 控制 advantage 对组准确率的敏感程度

### 2. 双向 Entropy 动态分析
基于 natural policy gradient 的单步 bandit 近似，entropy 变化由 covariance 控制：
$$\Delta \mathcal{H}(\pi(\cdot|x)) \approx -\eta \cdot \mathrm{Cov}_{y \sim \pi(\cdot|x)}(\log \pi(y|x), A(y))$$

实验验证 covariance 与组准确率 p 正相关。结合 GRPO 的 reweighting 机制：
- **正 rollout**: advantage 权重随 p 递减 → **对抗**自然 entropy 下降趋势 → 维持 informative entropy
- **负 rollout**: advantage 权重随 p 递增 → **加速**自然 entropy 下降趋势 → 剪除 spurious entropy

### 3. 消融验证
- **Pos-Only Modulation** (β_pos=0.5, β_neg=0): 维持更高 entropy + 准确率提升 → 证实 informative entropy
- **Neg-Only Modulation** (β_pos=0, β_neg=0.5): entropy 大幅下降 + 准确率提升 → 证实 spurious entropy 的剪除
- **对抗翻转实验**: EntDecrease（翻转正 advantage）和 EntIncrease（翻转负 advantage）均导致性能下降，验证双向调制的必要性
- **Entropy Regularization**: 虽然提升 entropy，但无法匹配 Pos-Only 的推理准确率 → 盲目提升 entropy 不如精准 refinement

### 4. AsymGRPO 解耦公式
$$A_{i,t}(p) = \begin{cases} \left(\frac{1-p}{p}\right)^{\beta_{\mathrm{pos}}} & \text{if } r(x,y_i)=1 \\ -\left(\frac{p}{1-p}\right)^{\beta_{\mathrm{neg}}} & \text{if } r(x,y_i)=0 \end{cases}$$

策略梯度显式分解为正/负两部分：
$$\nabla_\theta \mathcal{J}_{\mathrm{Asym}} = \underbrace{\sum_{i \in \mathcal{I}^+} A_{\mathrm{pos}}^{(\beta_{\mathrm{pos}})} \nabla_\theta \log \pi_\theta}_{\text{Positive Rollout Gradient}} + \underbrace{\sum_{i \in \mathcal{I}^-} A_{\mathrm{neg}}^{(\beta_{\mathrm{neg}})} \nabla_\theta \log \pi_\theta}_{\text{Negative Rollout Gradient}}$$

## 实验结果

### 主要结果 (Qwen3-4B, 5 个数学推理 benchmark)

| Method | MATH-500 | AIME24 | AIME25 | AMC23 | Olympiad | Avg. |
|--------|----------|--------|--------|-------|----------|------|
| REINFORCE | 86.60 | 28.67 | 24.67 | 73.75 | 54.86 | 53.71 |
| GRPO | 88.20 | 31.00 | 27.33 | 78.25 | 57.74 | 56.50 |
| GRPO w/ Entro.Reg | 88.20 | 38.33 | 28.33 | 75.50 | 57.24 | 57.52 |
| GRPO w/ Clip-higher | 90.07 | 34.67 | 32.33 | 78.50 | 58.18 | 58.75 |
| Dr.GRPO | 88.87 | 36.33 | 30.00 | 78.25 | 57.24 | 58.14 |
| **AsymGRPO** | **89.33** | **39.33** | 28.67 | **81.00** | 58.48 | **59.36** |
| **AsymGRPO + Clip-higher** | 89.73 | 33.67 | **36.00** | **83.25** | **58.93** | **60.32** |

**关键发现**:
- AsymGRPO vs GRPO: +2.86% 平均提升
- AsymGRPO + Clip-higher vs GRPO + Clip-higher: +1.57% 平均提升
- 与 entropy regularization 方法可协同使用
- AIME24 上 AsymGRPO 达到 39.33%（+8.33% over GRPO），显示在难题上探索优势

### 机制分析发现
- 增加 β_pos → "none-solved" 比例显著下降 → 扩展可解边界
- 增加 β_neg → 正样本的 log probability increment 提升 → 缓解 Lazy Likelihood Displacement
- β_neg 过大 (0.75) 反而降低正样本增益 → 调制强度需平衡

## 与 SPC/URLVR 研究的关系

### 直接关联
1. **Informative vs Spurious Entropy 分解与 SPC 的 semantic consistency**: AsymGRPO 从 entropy 视角区分推理中的有效多样性与噪声，SPC 从语义一致性视角实现类似目标。AsymGRPO 的 informative entropy 概念可作为 SPC step-level credit 的互补信号
2. **Group accuracy 作为 probing 信号**: AsymGRPO 使用 $p$（组内准确率）作为 advantage 调制的锚点，这是一种 outcome-level 的 probing。SPC 的 probing 机制在 step-level 操作，两者可形成层级互补
3. **非对称处理正/负样本**: AsymGRPO 对正负 rollout 分别调制，SPC 的 credit assignment 也应区分推理路径中的有效步骤和错误步骤的贡献
4. **Lazy Likelihood Displacement 问题**: AsymGRPO 发现不恰当的负梯度会抑制正确路径的概率增长（因共享长 prefix），这对 SPC 的 step-level penalty 设计有重要启示——惩罚错误步骤时需避免影响正确前缀

### 对 Co-Evolving Verifier 的启示
- Verifier 的校准可以利用 informative/spurious entropy 分类：verifier 应更信任 informative entropy 高的区域（多样但有效），警惕 spurious entropy 高的区域（噪声）

## 面试 Q&A

### Q1: 为什么传统的 entropy regularization 在 LLM-RL 中效果不佳？
**A**: 三个原因：(1) **超参极度敏感**——LLM 的 vocab 巨大，entropy 尺度与常规 RL 差异大；(2) **entropy explosion 风险**——过大的 entropy bonus 导致近均匀分布，生成语义无意义内容；(3) **盲目提升**——entropy regularization 不区分 informative entropy（值得保留的推理多样性）和 spurious entropy（应被消除的噪声），等价于"一碗水端平"，无法精准调控。AsymGRPO 证明了 GRPO 本身的 entropy refinement 机制（通过 accuracy-dependent reweighting 隐式区分）比显式 entropy bonus 更有效。

### Q2: AsymGRPO 如何与 step-level credit assignment（如 SPAE、SPC）协同？
**A**: AsymGRPO 工作在 **rollout-level**，通过非对称 advantage 调制整个 trajectory 的梯度权重。它解决的是"哪些 rollout 应该被更强化/惩罚"的问题。Step-level 方法（SPAE/SPC）解决"rollout 内部哪些步骤贡献最大"的问题。两者正交且可叠加：AsymGRPO 决定外层权重，step-level 方法决定内部 credit 分配。实际上，AsymGRPO 的 β_pos 主要影响正样本的探索保持（宏观），而 SPC 的 step-level probing 在微观层面引导具体推理路径。

### Q3: β_neg 过大为什么会导致性能下降？
**A**: β_neg 过大时，对**低准确率组**（即难题）的惩罚反而变弱（因为 $(p/(1-p))^{\beta_{neg}}$ 在 p 小时值更小），而对**高准确率组**的惩罚过强。这导致：(1) 难题上的常见错误模式得不到足够惩罚，模型在困难问题上形成僵化行为；(2) 简单题上的偶发错误被过度惩罚，抑制了正确路径的概率增长（Lazy Likelihood Displacement）。关键洞察是负 rollout 调制需要平衡——既要有效剪除 spurious entropy，又不能冻结模型在困难问题上的改进能力。
