---
title: "PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching"
type: paper
tags: [URLVR, GFlowNet, distribution-matching, sharpening, alpha-power, trajectory-balance, unsupervised, length-bias, creativity]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2603.18363]
status: active
---

# PowerFlow: GFlowNet + α-Power 分布匹配的无监督 LLM 微调

## 基本信息
- **作者**: Ruishuo Chen, Yu Chen, Zhuoran Li, Longbo Huang
- **机构**: IIIS, Tsinghua University (清华大学交叉信息研究院, 姚班)
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.18363v1)
- **链接**: https://arxiv.org/abs/2603.18363
- **Base Model**: Qwen2.5-1.5B/3B/7B, Qwen2.5-Math-1.5B/7B, Llama-3.2-3B-Instruct

## 一句话总结
> 将无监督 LLM 微调重新表述为**分布匹配问题**：用 GFlowNet 作为 amortized variational sampler 匹配 base model 的 $\alpha$-power 分布，通过 length-aware Trajectory-Balance 目标消除自回归生成固有的长度偏差；$\alpha > 1$ 锐化分布增强推理（匹配/超越有监督 GRPO），$\alpha < 1$ 平坦化分布释放创造力（同时提升多样性和质量）。

## 摘要
无监督 RLIF 方法依赖启发式内在 reward，缺乏明确的理论优化目标且易受退化偏差影响。PowerFlow 将无监督微调重新表述为**分布匹配问题**：以 GFlowNet 为 amortized variational sampler 匹配 base model 的 $\alpha$-power 分布 $p_\alpha(y|q) \propto p_{\text{base}}(y|q)^\alpha$。提出 **length-aware Trajectory-Balance (LA-TB)** 目标中和自回归生成的结构性长度偏差。$\alpha > 1$ 锐化分布提升推理能力，$\alpha < 1$ 平坦化分布释放创造力。实验证明 PowerFlow 一致超越 RLIF baselines，匹配甚至超越有监督 GRPO。

## 核心贡献
1. **Distribution Matching 范式**: 首次将无监督 LLM 微调从启发式 reward 工程转化为原理性的分布匹配问题，用 $\alpha$-power 分布作为理论优化目标
2. **Length-Aware Trajectory-Balance (LA-TB)**: 通过将 partition function 重参数化为 $Z_\phi(q,y) = (Z'_\phi(q))^{|y|}$，在长度归一化能量面上优化，消除自回归生成的指数级长度偏差
3. **双向能力激发**: 单一 $\alpha$ 参数控制锐化（推理）vs 平坦化（创造力），提供统一框架
4. **理论联系**: 证明 majority-voting 的 RLIF 方法可以被形式化为极端分布锐化（Theorem D.1）

## 方法

### 核心框架：分布匹配

#### 1. $\alpha$-Power 分布

目标分布是 base model 的 $\alpha$ 次幂归一化：

$$p_\alpha(y|q) = \frac{p_{\text{base}}(y|q)^\alpha}{Z(q, \alpha)}, \quad Z(q, \alpha) = \sum_y p_{\text{base}}(y|q)^\alpha$$

- $\alpha > 1$：**锐化**，概率质量集中到高概率路径 → 增强推理
- $\alpha < 1$：**平坦化**，概率质量分散到长尾区域 → 释放创造力
- $\alpha = 1$：保持 base model 不变

**$\alpha$-power 分布的优势**:
- 严格保持 base model 的相对概率排序和模式结构（单调性）
- 调节熵而不引入外部 bias
- 避免启发式 reward 设计的各种退化（长度坍缩、过度自信、模式坍缩）

#### 2. GFlowNet 作为 Amortized Sampler

匹配目标分布的 partition function $Z(q)$ 在 response 空间上不可计算。通过变分推断视角，最小化 reverse KL divergence：

$$\mathbb{D}_{\text{KL}}(\pi_\theta \| p_{\text{target}}) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|q)}{\tilde{p}_{\text{target}}(y|q)}\right] + \log Z(q)$$

GFlowNet 的 **Trajectory Balance (TB)** 目标将分布匹配转化为 RL-style 优化。对 LLM 的自回归结构，backward policy 简化为 $P_B = 1$，TB loss 为：

$$\mathcal{L}_{\text{TB}}(\theta, \phi; q, y) = \left(\log Z_\phi(q) + \sum_{t=1}^{T} \log \pi_\theta(y_t | y_{<t}, q) - \log \tilde{p}_{\text{target}}(y|q)\right)^2$$

### 关键问题：长度偏差

自回归生成的 $\log p(y|q) = \sum_{t=1}^{|y|} \log p(y_t | y_{<t}, q)$ 近似与序列长度 $|y|$ 负线性相关。直接分布匹配会被长度主导而非语义密度：
- $\alpha > 1$（锐化）→ 模型倾向生成极短平凡序列（长度坍缩）
- $\alpha < 1$（平坦化）→ 模型倾向生成重复的长序列
- Token-level 归一化（除以 $|y|$）：初期有效但后期退化，因为模型通过重复无意义 token 人为降低平均能量

### Length-Aware Trajectory-Balance (LA-TB)

**核心思想**: 将 partition function 重参数化为 length-aware 能量项：

$$Z_\phi(q, y) = (Z'_\phi(q))^{|y|}$$

加上 $|y|$ 归一化后的 LA-TB 目标：

$$\mathcal{L}_{\text{LA-TB}}(\theta, \phi; q, y) = \left(\log Z'_\phi(q) + \frac{1}{|y|} \log \frac{\pi_\theta(y|q)}{\tilde{p}_{\text{target}}(y|q)}\right)^2$$

收敛到 length-normalized 目标分布：

$$\pi^*(y|q) \propto \frac{\tilde{p}_{\text{target}}(y|q)}{Z'_\phi(q)^{|y|}}$$

### 最终 PowerFlow 目标

实例化为 $\alpha$-power 分布 + format penalty $\psi(y)$：

$$\mathcal{L}_{\text{PowerFlow}} = w \cdot \left(\log Z'_\phi(q) + \frac{1}{|y|} \log \pi_\theta(y|q) - \alpha \left[\frac{1}{|y|} \log p_{\text{base}}(y|q) + \psi(y)\right]\right)^2$$

其中 importance sampling ratio：

$$w = \text{clip}\left(\frac{\pi_\theta(y|q)}{\pi_{\text{old}}(y|q)}, 1-\epsilon, 1+\epsilon\right)^{\text{detach}}$$

$\psi(y)$：格式惩罚（缺少 $\boxed{}$ 时 $\psi = -0.5$，否则为 0）

### 理论联系

**Theorem D.1**: Majority-voting 的 RLIF（如 TTRL）等价于极端分布锐化，将策略推向 dominant mode。PowerFlow 提供了可控的中间锐化程度。

### 实现细节
| 参数 | 值 |
|------|-----|
| $\alpha$ (base model reasoning) | 4 |
| $\alpha$ (instruct model reasoning) | 2 |
| $\alpha$ (creativity/flattening) | 0.5 |
| Training data | NuminaMath-CoT 18k subset |
| Evaluation | avg@16, T=1.0, top-p=1.0 |
| Format penalty $\psi$ | -0.5 (if no \boxed{}) |

## 实验结果

### 主实验：推理能力（avg@16）

| Model | Method | MATH500 | Olympiad | AIME24 | AIME25 | AMC23 | GPQA | **Average** |
|-------|--------|---------|----------|--------|--------|-------|------|-------------|
| Qwen2.5-1.5B | Base | 6.20 | 1.90 | 0.00 | 0.00 | 1.40 | 25.80 | 5.88 |
| | Intuitor | 47.40 | 15.30 | 1.50 | 0.80 | 22.30 | 26.40 | 18.95 |
| | **PowerFlow** | **49.30** | **16.00** | 0.80 | **1.50** | **23.80** | **27.70** | **19.85** |
| | GRPO (supervised) | 45.40 | 14.10 | 1.00 | 0.40 | 21.90 | 26.00 | 18.13 |
| Qwen2.5-Math-1.5B | Base | 43.30 | 20.90 | 4.60 | 1.90 | 28.40 | 26.10 | 20.87 |
| | EMPO | 69.90 | 32.20 | 12.30 | 4.60 | 46.20 | 29.50 | 32.45 |
| | **PowerFlow** | **70.90** | **32.50** | 10.80 | **10.00** | **53.30** | 28.30 | **34.30** |
| | GRPO (supervised) | 71.40 | 34.00 | 8.10 | 6.70 | 49.50 | 26.80 | 32.75 |
| Qwen2.5-Math-7B | Base | 46.70 | 22.30 | 12.30 | 4.20 | 34.50 | 29.70 | 24.95 |
| | TTRL | 80.40 | 39.60 | 21.70 | 11.90 | 58.80 | 34.70 | 41.18 |
| | EMPO | 79.30 | 41.70 | 15.80 | 12.30 | 60.20 | 36.00 | 40.88 |
| | **PowerFlow** | 78.10 | 40.10 | 20.00 | **14.40** | **63.40** | **37.00** | **42.17** |
| | GRPO (supervised) | 78.40 | 42.50 | 22.70 | 12.90 | 63.40 | 34.40 | 42.38 |
| Llama-3.2-3B | Base | 40.10 | 10.30 | 4.00 | 0.00 | 18.80 | 29.50 | 17.12 |
| | Intuitor | 50.40 | 16.60 | 9.20 | 0.20 | 27.30 | 30.50 | 22.37 |
| | **PowerFlow** | **50.60** | **16.60** | **10.70** | 0.40 | **28.80** | 30.20 | **22.88** |
| | GRPO (supervised) | 50.10 | 17.20 | 11.20 | 0.00 | 25.00 | 30.50 | 22.33 |

**关键结果**: PowerFlow 在 3 个配置上**超越有监督 GRPO**（Qwen2.5-1.5B: 19.85 vs 18.13, Qwen2.5-Math-1.5B: 34.30 vs 32.75, Llama-3.2-3B: 22.88 vs 22.33），在 Qwen2.5-Math-7B 上接近（42.17 vs 42.38）。

### Solution Diversity

PowerFlow 在 AIME24/25 上的 diversity score 为 **4.05**，超过 EMPO (3.80) 和 GRPO (3.93)。$\alpha$-power 分布保持 base model 的多模态结构，不会坍缩到单一模式。

### 创造力释放（$\alpha = 0.5$, Instruct 模型）

PowerFlow ($\alpha < 1$) 是唯一在创意写作中**同时提升 diversity 和 quality** 的方法：
- High-temp：提升 diversity 但降低 quality
- VS-Standard：在 7B 及以下模型上降低 quality
- **PowerFlow**: 语义多样性超越 instruct model，quality 也超越——实现 Pareto 改进

### 稳定性分析（Figure 3）

| 方法 | 行为 |
|------|------|
| TB-traj (标准 TB) | 立即长度坍缩 |
| RL-traj (标准 RL + KL) | 立即长度坍缩 |
| TB-token (token 归一化 TB) | 初期提升，后期退化（重复 token 利用） |
| RL-token (token 归一化 RL) | 初期提升，后期退化 |
| **PowerFlow (LA-TB)** | **稳定长度 + 单调性能提升** |

## 与其他工作的关系

### 与 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的理论关系
- **Theorem D.1**: TTRL 的 majority voting 等价于极端分布锐化，将策略推向 dominant mode
- PowerFlow 提供了可控的 $\alpha$ 参数来调节锐化程度，避免 TTRL 的过度锐化/模式坍缩
- PowerFlow 不需要 pseudo-label 机制，完全基于 base model 自身分布

### 与 [[wiki/papers/zhang-2025-empo|EMPO]] 的比较
- EMPO 用语义熵作为 intrinsic reward（仍是启发式）
- PowerFlow 有明确的理论优化目标（$\alpha$-power 分布匹配）
- PowerFlow 在大部分 benchmark 上超越 EMPO

### 与 [[wiki/papers/ghimire-2026-prism|PRISM]] 的关系
- PRISM 指出内部信号方法的退化问题（重复 pattern、过度自信）
- PowerFlow 通过 LA-TB 从理论上避免了这些退化模式
- LA-TB 的长度归一化能量面防止了长度坍缩和重复爆炸

### 范式差异
PowerFlow 代表了与主流 URLVR 方法完全不同的思路：
- **TTRL/CoVerRL/SCRL**: 改进 pseudo-label 质量 → 更好的 reward → 标准 RL
- **PowerFlow**: 绕过 reward 设计，直接匹配目标分布 → GFlowNet TB loss

## 局限性与开放问题
1. **$\alpha$ 选择**: base model 用 $\alpha=4$, instruct model 用 $\alpha=2$，但最优值因模型而异，缺乏自动调整机制
2. **GFlowNet 的 on-policy 约束**: TB loss 需要 on-policy 样本，计算效率低于 off-policy 方法
3. **仅匹配 base model 分布**: 目标是 base model 的变换，无法引入 base model 不具备的新知识
4. **Reasoning 提升上限**: 在 Qwen2.5-Math-7B 上仅与 GRPO 持平，说明纯分布锐化在大模型上可能触及 base model 的能力天花板
5. **未与 pseudo-label 方法结合**: 理论上 PowerFlow 可以与 TTRL/SCRL 结合（先锐化再用 pseudo-label RL），但论文未探索

## 与 SPC/URLVR 研究的关系

### 对 SPC 方案的理论启发
1. **分布锐化作为统一视角**: PowerFlow 证明 URLVR 的本质是分布锐化（Theorem D.1）。SPC 的 step-level credit assignment 可以从这个角度理解——好的 step-level reward 应该帮助模型更精确地锐化到正确推理路径上。

2. **长度偏差问题的普适性**: LA-TB 揭示了自回归生成中长度偏差的严重性。SPC 在设计 step-level reward 时也需要考虑类似问题——长步骤和短步骤的 reward 需要适当归一化。

3. **互补性**: PowerFlow 是"无 reward"范式（直接分布匹配），SPC 是"精细 reward"范式（step-level credit）。两者可能互补：PowerFlow 提供稳定的全局分布锐化基础，SPC 在此基础上提供更精细的 step-level 引导。

4. **Co-Evolving Verifier 视角**: PowerFlow 不需要 verifier，但其 $\alpha$-power 分布的理论框架可以为 Co-Evolving Verifier 的 calibration 提供参考——verifier 应该引导策略匹配什么样的目标分布？

5. **创造力释放的意外启示**: $\alpha < 1$ 的实验表明 aligned model 的分布过度锐化会损害创造力。这提醒 SPC 在设计 step-level reward 时需要注意不要过度压缩推理路径的多样性。

## 面试相关

- **Q: PowerFlow 与传统 RLIF 方法（如 TTRL、Intuitor）的本质区别是什么？** 🔴
- A: 传统 RLIF 方法定义启发式 intrinsic reward（如 majority voting、self-certainty），然后用标准 RL（如 GRPO/PPO）最大化 reward。但这些 reward 缺乏明确的理论优化目标，易导致退化（长度坍缩、过度自信、模式坍缩）。PowerFlow 将问题重新表述为分布匹配——直接让策略匹配 base model 的 $\alpha$-power 分布 $p_\alpha(y|q) \propto p_{\text{base}}(y|q)^\alpha$，用 GFlowNet 的 Trajectory-Balance loss 优化。这消除了 reward 设计中的 bias，因为目标分布严格保持 base model 的相对概率排序。

- **Q: 为什么直接匹配 $\alpha$-power 分布会导致长度坍缩？LA-TB 如何解决？** 🔴
- A: 自回归生成中 $\log p(y|q) = \sum_t \log p(y_t | y_{<t}, q)$ 与长度近似负线性。当 $\alpha > 1$（锐化）时，短序列的 $p(y|q)^\alpha$ 相对更高，模型倾向生成极短平凡序列来最大化目标概率。LA-TB 将 partition function 重参数化为 $Z_\phi(q,y) = (Z'_\phi(q))^{|y|}$，使损失变为 $(\log Z'_\phi + \frac{1}{|y|}\log \frac{\pi_\theta}{\tilde{p}_{\text{target}}})^2$——在长度归一化的能量面上优化，梯度不再被序列长度主导，而是关注语义密度。

- **Q: PowerFlow 如何在无监督设定下超越有监督 GRPO？** 🟡
- A: 这基于"verification-generation asymmetry"原理——LLM 识别正确答案比生成正确答案更容易，意味着 base model 分布中已经编码了正确推理路径但概率不够集中。PowerFlow 通过 $\alpha > 1$ 锐化恰好"放大"了这些已有路径的概率。相比之下，有监督 GRPO 可能引入 reward 分布与 base model 分布不匹配的问题，导致分布漂移。此外，PowerFlow 保持了 solution diversity（diversity score 4.05 vs GRPO 3.93），避免了 GRPO 的模式坍缩问题。
