---
title: "When Models Judge Themselves: Unsupervised Self-Evolution for Multimodal Reasoning"
type: paper
tags: [URLVR, multimodal, self-consistency, actor-judge, distributional-reward, GRPO]
created: 2026-04-07
updated: 2026-04-07
sources: [https://arxiv.org/abs/2603.21289]
status: active
---

# Self-Judge: Actor-Judge 无监督多模态推理自进化

## 基本信息
- **作者**: Wu et al.
- **机构**: OPPO AI Center & Tsinghua University
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.21289)
- **链接**: https://arxiv.org/abs/2603.21289
- **代码**: https://github.com/OPPO-Mente-Lab/LLM-Self-Judge

## 一句话总结
> 多模态无监督自进化框架，提出 Actor-Judge 架构（冻结 Judge 做 bounded score modulation）+ group-wise distributional reward modeling，在视觉推理任务上跨 6 个不同 backbone 模型（2B-32B）均有效。

## 摘要
多模态推理（如几何、图表理解）的 ground-truth 标注成本极高。Self-Judge 提出完全无监督的自进化方案：Actor 通过 [[self-consistency|self-consistency]] 生成初始 reward 分布，冻结的 Judge（Actor 的副本）提供 bounded score modulation 来调整分布，最终通过 energy-based distributional modeling 做策略优化。核心洞察：**majority voting 丢弃了答案分布的丰富结构信息**，需要更精细的 distributional modeling。

## 核心贡献
1. **Actor-Judge 架构**: Judge 是 Actor 的冻结副本，提供 bounded score modulation 而非直接做 reward
2. **反对 Majority Voting**: 理论和实验论证 MV 会导致 policy 快速 collapse 到低熵
3. **Group-wise Distributional Reward**: 用 energy-based scaling + log-sum-exp baseline 诱导 reward-defined target distribution
4. **多模态验证**: 首次在视觉推理（几何、图表、数学视觉）上验证无监督 RL
5. **广泛的 backbone 验证**: 在 2B 到 32B 共 6 个不同模型上均有效

## 方法

### 问题定义
多模态推理任务（几何证明、图表分析等）的 ground-truth 获取成本极高。现有的 [[self-consistency|self-consistency]] 方法直接用 majority voting 做 reward，但会导致 policy 快速 collapse。如何更好地利用 self-consistency 信号？

### 技术方案

#### 1. Actor: Self-Consistency Reward

对输入 x 采样 n 条推理轨迹 {τ_1, ..., τ_n}，提取每条的最终答案 a_i：

$$r_i^{SC} = \hat{p}(a_i) = \frac{|\{j : a_j = a_i\}|}{n}$$

直觉：被更多独立采样支持的答案更可能正确（同 EMPO）。

#### 2. 为什么 Majority Voting 不够

MV 将答案分布简化为 binary（最多票 = 1，其他 = 0），丢弃了关键信息：
- 答案分布的 shape（如 40%-35%-25% vs 90%-5%-5%）反映了问题的难度和模型的不确定性
- MV 一旦某答案早期占优就 binary amplify → policy 快速 collapse 到低熵 → 失去探索能力

#### 3. Judge: Bounded Score Modulation

Judge 是 Actor 的冻结副本，对每条轨迹生成评分 s_k。但 Judge 评分不直接做 reward，而是通过 bounded sigmoid 函数做 modulation：

$$g(s) = 1 + \lambda_+ \cdot \sigma\left(\frac{s - t_h}{\tau_h}\right) - \lambda_- \cdot \sigma\left(\frac{t_l - s}{\tau_l}\right)$$

其中：
- $t_h, t_l$ 是高/低阈值
- $\lambda_+, \lambda_-$ 控制 modulation 幅度
- $\tau_h, \tau_l$ 是温度参数

**设计意图**: Bounded modulation 确保 Judge 只做小幅调整（±λ范围），不会主导 reward signal，避免 Judge 自身偏差被 amplify。

#### 4. 最终 Reward 与 Format Penalty

$$R_k = r_k \cdot g(s_k) - \lambda_{fmt} \cdot \delta_k$$

其中 $r_k$ 是 SC reward，$g(s_k)$ 是 Judge modulation，$\delta_k$ 是格式违规惩罚。

#### 5. Energy-Based Distributional Modeling

将 reward 转化为 energy-based target distribution：

$$q_\alpha(\tau_k | x) = \frac{\exp(\alpha \cdot R_k)}{\sum_{j=1}^{n} \exp(\alpha \cdot R_j)}$$

然后 GRPO 的 advantage 定义为：

$$A_k(x) = \log q_\alpha(\tau_k | x)$$

**Log-sum-exp baseline**: 不同于标准 GRPO 用组内均值做 baseline，Self-Judge 用 log-sum-exp 做 baseline，理论上更好地匹配 energy-based target distribution。

### 关键公式

**完整的策略更新**：

$$\mathcal{L}(\theta) = -\mathbb{E}_{x, \{\tau_k\}} \left[ \frac{1}{n} \sum_{k=1}^{n} \min\left(\frac{\pi_\theta(\tau_k|x)}{\pi_{\theta_{old}}(\tau_k|x)} A_k, \; \text{clip}(\cdot, 1-\epsilon, 1+\epsilon) A_k\right) \right]$$

其中 $A_k = \log q_\alpha(\tau_k|x)$，$q_\alpha$ 是 reward-induced energy-based distribution。

## 实验结果

### 主实验（Qwen2.5-VL-7B-Instruct, 8×A800, 20 epochs）

| Benchmark | Base | + Self-Judge | 提升 |
|-----------|------|-------------|------|
| MathVision | 25.0% | **30.9%** | +5.9% |
| Geo3K | 68.2% | **74.1%** | +5.9% |
| MathVerse | 42.3% | **46.8%** | +4.5% |
| WeMath | 55.1% | **59.3%** | +4.2% |

### 跨 Backbone 泛化（6 个模型）

| Model | Geo3K Base | + Self-Judge |
|-------|-----------|-------------|
| Qwen2.5-VL-3B | 55.8% | **62.1%** |
| Qwen2.5-VL-7B | 68.2% | **74.1%** |
| Qwen2.5-VL-32B | 78.5% | **82.3%** |
| InternVL2-2B | 42.1% | **48.7%** |
| InternVL2-8B | 61.3% | **67.2%** |
| LLaVA-OneVision-7B | 58.9% | **64.5%** |

在 2B 到 32B 共 6 个不同 backbone 上均有效。

### 跨任务泛化

| Task Type | Benchmark | Base | + Self-Judge |
|-----------|-----------|------|-------------|
| 图表理解 | ChartQA | 72.1% | **76.3%** |
| 视觉感知 | MMVP | 38.0% | **42.5%** |
| 几何推理 | Geo3K | 68.2% | **74.1%** |

不仅在训练任务（几何推理）上有效，在未训练的任务类型上也有泛化提升。

### 消融实验

| 配置 | Geo3K |
|------|-------|
| SC only (无 Judge) | 70.8% |
| Judge only (无 SC) | 71.5% (训练不稳) |
| MV reward (majority voting) | 69.2% |
| **Full Self-Judge** | **74.1%** |

- **SC alone**: 有改进但有限，diversity 保持较好
- **Judge alone**: 引入质量信号但训练不稳定
- **MV reward**: 效果最差，验证了 MV 的 collapse 问题
- **Full method**: SC + Judge + distributional modeling 效果最佳

## 与其他工作的关系
- **基于**: [[grpo|GRPO]] 框架、[[self-consistency|Self-Consistency]] 概念
- **核心概念**: [[self-consistency|Self-Consistency]]、Energy-Based Model
- **对比**:
  - [[wiki/papers/zhang-2025-empo|EMPO]]：都用 self-consistency/clustering 做初始信号，但 EMPO 直接用频率做 reward，Self-Judge 用 distributional modeling 更精细
  - [[wiki/papers/rahman-2025-spark|SPARK]]：SPARK 用 self-consistency 训练外部 PRM，Self-Judge 用冻结的自身副本做 Judge
  - [[wiki/papers/ghimire-2026-prism|PRISM]]：都用混合信号，PRISM 混合 PRM + self-certainty，Self-Judge 混合 SC + bounded Judge
- **独特之处**: 唯一处理多模态（视觉推理）的 URLVR 工作
- **共同挑战**: 所有四篇都面对 majority voting 的局限性——EMPO 用语义聚类解决，SPARK 用 PRM 解决，PRISM 用混合信号解决，Self-Judge 用 distributional modeling 解决

## 面试相关
> Self-Judge 涉及多个创新点，适合深挖

- Q: Self-Judge 为什么不直接用 majority voting？🔴
- A: MV 将答案分布简化为 binary，丢弃分布的 shape 信息。比如 40%-35%-25% 和 90%-5%-5% 被 MV 等同处理但含义完全不同。MV 导致 policy 快速 collapse 到低熵，失去探索能力。Self-Judge 保留完整的频率分布 + bounded Judge modulation，信号更丰富。

- Q: Judge 为什么是冻结的 Actor 副本而不是独立模型？🟡
- A: (1) 不需要额外模型，节省计算；(2) 冻结保证 reward 是 stationary 的（同 SPARK 的发现）；(3) Bounded modulation 确保 Judge 只做小幅调整，不会主导信号——如果 Judge 自身有偏差不会被 amplify。

- Q: Energy-based distributional modeling 相比标准 GRPO 的优势？🟡
- A: 标准 GRPO 用组内均值做 baseline，Self-Judge 用 log-sum-exp baseline，理论上更好地匹配 reward-induced target distribution。效果上看消融实验证实 distributional modeling 带来显著提升。

- Q: Self-Judge 能否用于纯文本任务？🟡
- A: 理论上可以。核心方法（SC + Judge modulation + distributional modeling）不依赖多模态特性。但在文本任务上已有更成熟的方案（EMPO、SPARK），Self-Judge 的独特价值在多模态场景。

## 个人笔记
> Self-Judge 最大的理论贡献是 **distributional reward modeling**——不是把 self-consistency 简化成 binary（MV）或标量（频率），而是保留完整的分布结构，用 energy-based model 做策略优化。这比 EMPO 的简单频率 reward 在理论上更优美。
>
> Actor-Judge 架构的 bounded modulation 设计很巧妙——Judge 不直接打分，只做小幅调整，避免了 Judge 偏差被 amplify 的问题（类似 PRISM 发现的纯 PRM 问题）。
>
> 实践价值：是目前唯一在多模态任务上验证 URLVR 的工作，对做视觉推理的组非常有参考价值。跨 6 个 backbone 的泛化性也很强，说明方法不依赖特定模型架构。
