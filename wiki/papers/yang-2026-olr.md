---
title: "OLR: Can LLMs Learn to Reason Robustly under Noisy Supervision?"
type: paper
tags: [RLVR, noisy-labels, label-refinement, noise-robustness, self-correction, majority-voting, GRPO, early-correctness-coherence]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2604.03993]
status: active
---

# OLR: 噪声监督下的鲁棒推理学习

## 基本信息
- **作者**: Shenzhi Yang, Guangcheng Zhu, Bowen Song, Sharon Li, Haobo Wang‡, Xing Zheng, Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Gang Chen‡
- **机构**: Zhejiang University（浙江大学）, Ant Group（蚂蚁集团）, University of Wisconsin-Madison
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2604.03993v1)
- **链接**: https://arxiv.org/abs/2604.03993
- **代码**: https://github.com/ShenzhiYang2000/OLR
- **Base Models**: Qwen3-4B-Base, Qwen3-1.7B-Base, Qwen2.5-3B

## 一句话总结
> 首次系统分析 RLVR 中噪声标签的机制，区分 **inactive**（降低数据效率）和 **active**（误导模型分布）噪声标签，发现 **Early Correctness Coherence** 现象，提出 Online Label Refinement (OLR) 通过 pass rate 斜率 + 历史一致性两个准则渐进式自修正噪声标签，在 0.1-0.9 噪声比下均鲁棒。

## 摘要

RLVR 依赖大量完美标注数据，但实际中专家稀缺导致噪声标签不可避免，且这一问题在 RLVR 中几乎未被研究。不同于传统分类中所有错误标签都贡献 loss，RLVR 具有独特的 **rollout-based** 特性：标签的影响取决于当前策略是否能生成 rollout 来实现它。基于此，本文将噪声标签分为：
- **Inactive 噪声标签**：模型无法生成对应 rollout（$\pi_\theta(\tilde{y}|x) = 0$），减少数据效率但不主动误导
- **Active 噪声标签**：模型可以生成对应 rollout（$\pi_\theta(\tilde{y}|x) > 0$），获得正 advantage 将模型推向错误分布

实验发现 **Early Correctness Coherence**：训练早期，clean 和 noisy 样本的准确率同步上升。OLR 利用这一现象，当 majority answer 的 pass rate 呈正斜率且历史一致时，用模型自身生成的多数答案替换原始标签。

## 核心贡献

1. **首次系统化分析 RLVR 中的噪声标签机制**：提出 Rollout Feasibility 概念，将噪声分为 inactive（不可生成→浪费数据）和 active（可生成→误导模型）两类
2. **发现 Early Correctness Coherence 现象**：即使在噪声监督下，训练早期 clean 和 noisy 样本的准确率同步提升——因为共享参数的跨样本耦合效应使 clean 样本的梯度间接帮助 noisy 样本
3. **提出 OLR 方法**：首个针对 RLVR 的噪声标签缓解方法，基于 pass rate 正斜率（模型能力提升的证据）+ 历史一致性（过滤偶然多数答案）两个准则进行渐进式标签修正
4. **理论保证**：证明 OLR 替换的标签正确概率 $\geq 1 - \epsilon$，有效噪声比 $\rho_{\text{eff}} = \rho(1-\Delta) < \rho$，提高了噪声容忍阈值

## 方法

### 1. 噪声标签形式化

**Rollout Feasibility**（定义 3.1）：解 $y$ 在策略 $\pi_\theta$ 下是 rollout-feasible 的当且仅当 $\pi_\theta(y|x) > 0$。

**Inactive 噪声标签**（定义 3.2）：$\tilde{y}(x) \neq y^\star(x)$ 且 $\pi_\theta(\tilde{y}|x) = 0$
- 所有 rollout 都无法匹配噪声标签，advantage 全为 0 或全一致
- 影响：浪费 rollout 资源，降低数据效率

**Active 噪声标签**（定义 3.3）：$\tilde{y}(x) \neq y^\star(x)$ 且 $\pi_\theta(\tilde{y}|x) > 0$
- 模型有概率生成与噪声标签匹配的 rollout，获得正 advantage
- 影响：将模型推向错误分布，危害更大

### 2. Early Correctness Coherence 理论分析

**定理 3.4**（非正式版）：令 $\rho$ 为噪声比，$L_t(x) = \log \frac{p_t(y^\star(x)|x)}{p_t(\tilde{y}(x)|x)}$ 为正确/错误答案的对数概率比。在以下条件下：
1. 初始时 $p_0(y^\star) > p_0(\tilde{y})$
2. Clean 与 noisy 样本之间存在正的梯度耦合 $\gamma > 0$
3. 噪声比 $\rho < \rho_c = \frac{\gamma G_c}{\gamma G_c + G_n}$

则：

$$L_t(x) \geq L_0(x) + \eta t \left( \Delta_s - O\left(\sqrt{\frac{\log(T/\delta)}{K}}\right) \right)$$

**含义**：即使在噪声监督下，正确答案的概率也会随训练逐步增大（$p_t(y^\star|x) \gg p_t(\tilde{y}|x)$），这是因为 clean 样本的梯度通过共享参数间接提升了 noisy 样本上正确答案的概率。

### 3. Online Label Refinement (OLR)

在每个 epoch $t$，对每个 prompt $x$ 生成 $K$ 个 rollout，追踪 majority answer 及其 pass rate：

$$y_t^{\text{maj}}(x) = \arg\max_c |\{y \in \mathcal{Y}_t(x) : y = c\}|$$

$$p_t^{\text{maj}}(x) = \frac{1}{K} |\{y \in \mathcal{Y}_t(x) : y = y_t^{\text{maj}}(x)\}|$$

维护历史轨迹 $\mathcal{H}_t(x)$，基于两个准则决定是否替换标签：

#### 准则 1: 正收敛斜率 (Positive Convergence Slope)

对 pass rate 序列做线性回归：

$$S_t(x) = \frac{(\mathbf{t} - \bar{t}\mathbf{1})^\top (\mathbf{p}_t(x) - \bar{p}\mathbf{1})}{(\mathbf{t} - \bar{t}\mathbf{1})^\top (\mathbf{t} - \bar{t}\mathbf{1})}$$

正斜率 $S_t(x) > \delta_{\text{slope}}$ 表明模型对该答案越来越有信心，rollout 越来越收敛。

#### 准则 2: 历史一致性 (Historical Consistency)

检查当前 majority answer 是否与历史上最频繁的 majority answer 一致：

$$C_t(x) = \mathbb{I}(y_t^{\text{maj}}(x) = y_t^{\text{hist}}(x))$$

其中 $y_t^{\text{hist}}(x) = \arg\max_y |\{(t', y_{t'}^{\text{maj}}(x)) \in \mathcal{H}_t(x) : y_{t'}^{\text{maj}}(x) = y\}|$

#### 标签替换规则

经过初始学习阶段 $T$ 之后：

$$\hat{y}_t(x) = \begin{cases} y_t^{\text{maj}}(x) & \text{if } S_t(x) > \delta_{\text{slope}} \text{ and } C_t(x) \\ \tilde{y}(x) & \text{otherwise} \end{cases}$$

### 理论保证

**定理 3.6**：OLR 替换的标签正确概率 $\Pr(\hat{y}_t(x) = y^\star(x)) \geq 1 - \epsilon$，其中 $\epsilon = O(\exp(-K\Delta_p^2))$。有效噪声比降低为 $\rho_{\text{eff}} = \rho(1-\Delta) < \rho$，噪声容忍阈值提高为 $\rho_c^{\text{OLR}} = \frac{\rho_c}{1-\Delta} > \rho_c$。

## 实验结果

### 主实验：不同噪声比下的性能（Qwen3-4B-Base, 800 samples from DAPO-Math）

#### Inactive Noise

| 噪声比 $\rho$ | GRPO ID Avg. | w/ OLR ID Avg. | 增益 | GRPO OOD Avg. | w/ OLR OOD Avg. | 增益 |
|---------------|-------------|----------------|------|---------------|-----------------|------|
| 0.1 | 44.5 | **44.9** | +0.4 | 60.9 | **62.1** | +1.2 |
| 0.3 | 38.7 | **44.4** | +5.7 | 58.0 | **60.7** | +2.7 |
| 0.5 | 36.1 | **44.9** | **+8.8** | 57.2 | **61.0** | +3.8 |
| 0.7 | 36.8 | **38.9** | +2.1 | 54.7 | **58.2** | +3.5 |
| 0.9 | 36.2 | **37.4** | +1.2 | 51.8 | **57.2** | +5.4 |

#### Active Noise

| 噪声比 $\rho$ | GRPO ID Avg. | w/ OLR ID Avg. | 增益 | GRPO OOD Avg. | w/ OLR OOD Avg. | 增益 |
|---------------|-------------|----------------|------|---------------|-----------------|------|
| 0.1 | 38.2 | **44.0** | +5.8 | 56.8 | **60.4** | +3.6 |
| 0.3 | 37.0 | **38.7** | +1.7 | 52.6 | **58.5** | +5.9 |
| 0.5 | 35.5 | **41.9** | **+6.4** | 49.1 | **53.9** | +4.8 |
| 0.7 | 35.0 | **37.2** | +2.2 | 46.7 | **54.8** | +8.1 |
| 0.9 | 21.7 | **25.1** | +3.4 | 25.8 | **26.5** | +0.7 |

**关键发现**：
- OLR 在所有噪声比（0.1-0.9）和两种噪声类型下均有正增益
- **Inactive noise $\rho=0.5$** 下 ID 平均增益最大（+8.8%），OLR 将性能恢复到接近低噪声水平
- **Active noise $\rho=0.7$** 下 OOD 增益最大（+8.1%），说明 OLR 对分布外泛化的改善尤为显著

### 与基线方法对比（$\rho=0.5$ Active Noise）

| Method | ID Avg. | OOD Avg. |
|--------|---------|----------|
| GRPO (baseline) | 35.5 | 49.1 |
| w/ Confidence Penalty | 36.6 (+1.1) | 56.2 (+7.1) |
| w/ Label Smoothing | 35.1 (-0.4) | 42.4 (-6.7) |
| w/ Small-loss Select | 18.1 (-17.4) | 22.9 (-26.2) |
| w/ Random Select | 34.8 (-0.7) | 35.5 (-13.6) |
| TTRL (无监督) | 36.9 | 57.7 |
| **w/ OLR** | **41.9** (+6.4) | **53.9** (+4.8) |

**关键发现**：
- 传统噪声学习方法（label smoothing, small-loss selection）在 RLVR 中表现很差甚至有害
- Small-loss selection 在 RLVR 中灾难性失败（-17.4%），因为 RLVR 中 small-loss ≠ clean sample
- OLR 显著优于所有 noise-robust 基线，ID 和 OOD 均为最优

### 4K 数据量扩展性验证（$\rho=0.5$ Active Noise）

| Method | ID Avg. | OOD Avg. |
|--------|---------|----------|
| w/o OLR | 37.8 | 55.3 |
| w/ OLR | **41.4** (+3.6) | 55.4 (+0.1) |

OLR 在更大数据量下仍有效，ID 增益 +3.6%。

## 与 SPC/URLVR 研究的关系

### 高度相关
1. **OLR 直接解决 URLVR 中的核心挑战——伪标签噪声**：TTRL 和所有基于 majority voting 的 URLVR 方法本质上都面临 noisy label 问题（投票结果不一定正确）。OLR 的 inactive/active noise 分类框架可以直接应用于分析 TTRL 伪标签的错误模式
2. **Early Correctness Coherence 为 SPC 提供理论基础**：该现象表明训练早期模型已经"隐含"了正确答案——这与 SPC 的 probing 思想一致：通过探测模型内部状态来发掘这些隐含的正确信号。SPC 可以被视为在 step-level 利用 Early Correctness Coherence 的一种方法
3. **OLR 的双准则设计启发 Co-Evolving Verifier**：
   - **Positive slope** ≈ verifier 应关注模型能力提升的信号（置信度在提升的答案更可能正确）
   - **Historical consistency** ≈ verifier 应有"记忆"，只信任持续稳定的信号
   - Co-Evolving Verifier 可以内化这两个准则，而不是作为后处理规则

### 整合方向
- **SPC + OLR**：SPC 在 step-level 做 credit assignment 时，可以借用 OLR 的思想——对 step-level reward 信号也做 "positive slope + historical consistency" 检查，只有持续可靠的 step reward 才被采纳
- **Co-Evolving Verifier 的噪声鲁棒性**：OLR 的理论框架（有效噪声比降低、噪声容忍阈值提高）可以作为 Co-Evolving Verifier 的设计目标——verifier 的核心任务之一就是降低训练信号中的有效噪声比
- **Active vs Inactive noise 的区分对 URLVR 的启示**：在 URLVR 中，错误伪标签如果恰好是模型容易生成的答案（active noise），危害远大于模型不会生成的答案（inactive noise）。SPC 的 step-level 分析可能帮助区分这两种情况

## 面试 Q&A

### Q1: RLVR 中的噪声标签为什么与传统分类中的不同？Active 和 Inactive 噪声标签有何区别？🔴
**A**: 核心区别在于 RLVR 的 **rollout-based** 特性。传统分类中，所有错误标签都直接通过 loss 影响模型。但在 RLVR 中，标签是否影响训练取决于模型是否能生成匹配该标签的 rollout（Rollout Feasibility）。**Inactive 噪声标签**：模型无法生成匹配的 rollout，所以该标签实际上不参与 advantage 计算——它只是浪费 rollout 资源，降低数据效率。**Active 噪声标签**：模型有概率生成匹配的 rollout，这些 rollout 会获得正 advantage 被强化，将模型推向错误方向。Active noise 更危险，因为它直接扭曲策略分布。

### Q2: Early Correctness Coherence 现象的机制是什么？OLR 如何利用它？🔴
**A**: **机制**：训练早期，尽管 noisy 样本的标签是错的，但 clean 样本的正确梯度通过共享参数（cross-sample coupling）间接帮助 noisy 样本生成正确答案。理论上，当噪声比 $\rho < \rho_c$ 时，clean 样本的正向耦合力大于 noisy 样本的负向力，使得正确答案的概率单调上升。**OLR 的利用方式**：既然训练过程本身会使正确答案逐渐涌现，OLR 通过监控两个信号来检测这种涌现：(1) majority answer 的 pass rate 正斜率——说明模型越来越倾向于该答案；(2) 历史一致性——同一答案在多个 epoch 中持续占据多数。两个条件同时满足时，用模型自己的 majority answer 替换原始（可能错误的）标签。

### Q3: 为什么传统的噪声学习方法（如 small-loss selection）在 RLVR 中失效？🟡
**A**: 传统方法基于"small loss ≈ clean sample"的假设，在分类任务中通常成立。但在 RLVR 中这个假设不成立：(1) 模型无法生成任何正确 rollout 的 **难题**（所有 reward 为 0）loss 很小但实际是 clean 的有价值样本；(2) 模型恰好总能生成匹配 noisy label 的 rollout 的**简单噪声题** loss 也很小但实际是有害的 active noise。Small-loss selection 因此可能同时丢弃有价值的难题并保留有害的简单噪声题，导致灾难性性能下降（实验中 -17.4%）。这说明 RLVR 需要专门的噪声标签处理方法，而非简单移植分类领域的技术。
