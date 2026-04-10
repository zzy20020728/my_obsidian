---
title: "Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling"
type: paper
tags: [URLVR, multimodal, self-evolution, softened-reward, retracing, geometric-reasoning, Qwen2.5-VL]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2604.03647]
status: active
---

# CSRS: Continuous Softened Retracing reSampling

## 基本信息
- **作者**: Yunyao Yu*, Zhengxian Wu*, Zhuohong Chen*, Hangrui Xu, Zirui Liao, Xiangwen Deng, Zhifang Liu, Senyuan Shi, Haoqian Wang†
- **机构**: Tsinghua University, Hefei University of Technology, University of Arizona, MAIS (Institute of Automation, CAS)
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2604.03647v2 [cs.CV])
- **链接**: https://arxiv.org/abs/2604.03647
- **代码**: https://github.com/yyy195/CSRS

## 一句话总结
> 针对多模态 LLM 无监督自演化中 majority voting 导致的 **model collapse** 问题，提出 **CSRS** 框架：(1) **RRM** 在 maternal trajectory 的 anchor point 处回溯重推理以扩展长尾推理路径探索；(2) **SFR** 用连续 softened frequency reward 替代 binary reward，理论证明其抑制分布极化；(3) **VSP** 通过视觉语义扰动迫使模型依赖数学逻辑而非视觉捷径。在 Qwen2.5-VL-7B 上 4 个多模态几何推理 benchmark 全面超越 MM-UPT baseline（MathVision +2.34%, WeMath +3.28%）。

## 摘要
多模态大模型的无监督自演化中，post-training 的反馈信号质量至关重要。现有方法主要依赖 **majority voting** 选取最频繁输出作为 pseudo-golden answer，但这可能源于模型内在偏置而非客观正确性，导致 **confirmation bias cycle** 和 **model collapse**——模型退化为确定性映射，丧失探索能力。

CSRS 提出三个协同组件：
1. **Retracing Re-inference Mechanism (RRM)**: 在初始推理轨迹中设置回溯锚点，从锚点重新推理以探索长尾推理路径
2. **Softened Frequency Reward (SFR)**: 用连续信号替代 binary reward，基于 answer 在采样集中的频率校准奖励
3. **Visual Semantic Perturbation (VSP)**: 对图像施加高斯噪声，迫使模型优先学习不变的数学逻辑而非表面视觉特征

## 核心贡献
1. **理论分析 model collapse 机制**: 通过推导 GRPO 的闭式解，揭示无监督自演化本质是 **指数自强化过程**，majority voting 的 binary reward 导致 Contrastive Factor 以 $e^\eta$ 最大速率指数膨胀，加速分布极化
2. **RRM 回溯重推理**: 突破传统静态采样的局限，在关键逻辑决策节点进行密集局部采样，有效扩展长尾样本空间
3. **SFR 连续奖励**: 理论证明 softened reward 的 Contrastive Factor $G_{SR} = \exp(\eta(\rho - \epsilon)) < G_{MV} = \exp(\eta)$，有效抑制分布极化
4. **VSP 视觉扰动**: 防止模型利用表面视觉特征作弊，确保自演化由真正的推理驱动
5. **SOTA 无监督自演化性能**: 在 4 个多模态数学推理 benchmark 上达到无监督自演化的 SOTA

## 方法详解

### 1. 理论分析：Model Collapse 的根源
GRPO 在无监督自演化中的策略迭代闭式解：
$$P_{n+1}(x) = \frac{P_n(x) e^{A_n(x)/\beta}}{Z}$$

定义 **Contrastive Factor** 衡量分布极化速度：
$$G_n = \frac{R_{n+1}}{R_n} = \exp(\eta \Delta r_n)$$

其中 $R_n = P_n(x_1)/P_n(x_2)$，$x_1$ 为高频样本，$x_2$ 为长尾样本。

- **Majority Voting**: $\Delta r_{MV} = 1$，$G_{MV} = e^\eta$ → 最大速率指数膨胀
- **Softened Frequency Reward**: $G_{SR} = \exp(\eta(\rho - \epsilon)) < G_{MV}$，因 $0 < \rho - \epsilon < 1$

**关键洞察**: SFR 通过引入连续性来阻尼分布极化，保留自演化过程中的逻辑多样性。

### 2. Retracing Re-inference Mechanism (RRM)
给定 prompt-image 对 $(X_p, X_i)$，模型生成 $n$ 条 maternal trajectories $\mathcal{A}_n = \{R_{p_1}, \ldots, R_{p_n}\}$。对每条轨迹 $R_p = (y_1, y_2, \ldots, y_n)$，在锚点处截断并拼接原始 prompt：

$$X_p' = \text{Concat}(X_p, (y_1, y_2, \ldots, y_{\lfloor \omega \cdot \text{len}(R_p) \rfloor}))$$

其中 $\omega \in (0,1)$ 为 **retracing rate**，$y_{\lfloor \omega \cdot \text{len}(R_p) \rfloor}$ 为回溯锚点。从锚点出发进行 $m$ 次局部探索，共产生 $m \times n$ 条 re-inference trajectories $\mathcal{A}_m$。

**核心思想**: 将广泛的全局搜索转化为关键逻辑分叉处的密集局部采样。

### 3. Softened Frequency Reward (SFR)
构建综合答案集 $\mathcal{A}_{all} = \mathcal{A}_n \cup \mathcal{A}_m$，包含 $(m+1) \times n$ 条轨迹答案。

**Base reward**（频率统计）：
$$R_{base}(a) = f_{base}(a) = \frac{\text{Count}(a, \mathcal{A}_{all})}{(m+1) \times n}$$

**Final reward**（频率方差校准）：
$$R_{final} = \left(\gamma \cdot \tanh\left(\beta \cdot \left(\frac{f_r}{f_{base} + \epsilon} - 1\right)\right) + 1\right) \times R_{base}$$

其中 $f_r = \frac{\text{Count}(a, \mathcal{A}_m)}{m \times n}$ 为答案在 re-inference 集中的频率。

- $\gamma$ 控制调制幅度（最优 $\gamma = 0.2$）
- $\beta$ 控制 tanh 敏感度（最优 $\beta = 5.0$）
- 若答案在 re-inference 后频率上升 → 路径具有潜在鲁棒性 → 奖励增强
- tanh 饱和性限制高频样本的奖励上界 → 防止过拟合

### 4. Visual Semantic Perturbation (VSP)
在 re-inference 阶段对原始图像施加高斯噪声：
$$I' = I + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

迫使模型依赖不变的逻辑结构，防止利用表面视觉特征作弊。

## 实验结果

### 主要结果（Qwen2.5-VL-7B，4 个 benchmark）

| Model / Method | Unsupervised? | Training Data | MathVision | MathVerse | MathVista | WeMath | Avg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | ✗ | - | 22.47 | 34.54 | 62.30 | 57.53 | 44.21 |
| Qwen2.5-VL-7B | ✗ | - | 25.40 | 44.24 | 66.42 | 67.65 | 50.93 |
| MM-UPT | ✓ | Geometry3K | 26.95 | 44.53 | 66.47 | 68.49 | 51.61 |
| MM-UPT | ✓ | GeoQA | 26.61 | 44.15 | 65.84 | 68.25 | 51.21 |
| MM-UPT | ✓ | MMR1 | 25.98 | 45.12 | 66.27 | 69.14 | 51.63 |
| **CSRS** | ✓ | Geometry3K | **27.97** | **46.01** | **67.81** | **71.77** | **53.39** |
| **CSRS** | ✓ | GeoQA | **28.95** | 45.82 | **68.25** | 69.32 | **53.09** |
| **CSRS** | ✓ | MMR1 | 27.86 | 45.89 | 67.81 | 70.53 | **53.05** |

**关键发现**:
- CSRS vs MM-UPT（最佳配置 Geometry3K）：MathVision +2.34%, MathVerse +1.48%, MathVista +2.41%, WeMath +3.28%
- MathVista 达到 68.25%，展示无监督自演化的巨大潜力
- 跨三个训练集（Geometry3K, GeoQA, MMR1）均稳定优于 baseline

### 消融实验

**组件消融（Table 5）**:

| Module | M-Vis. | M-Ver. | M-Vist. | WeMath |
|---|---|---|---|---|
| MajorityVote (baseline) | 25.40 | 44.24 | 66.42 | 67.65 |
| +SFR | 26.52 | 45.10 | 67.15 | 69.42 |
| +RRM | 26.85 | 45.35 | 67.38 | 69.85 |
| +VSP | 25.55 | 44.35 | 66.55 | 67.82 |
| +SFR+RRM | 27.68 | 45.85 | 67.62 | 71.30 |
| +SFR+VSP | 26.70 | 45.22 | 67.28 | 69.65 |
| +RRM+VSP | 27.02 | 45.48 | 67.50 | 70.12 |
| **CSRS (all)** | **27.97** | **46.01** | **67.81** | **71.77** |

- SFR 和 RRM 是主要贡献者，单独即可带来显著提升
- SFR+RRM 组合效果最突出（MathVista 67.62%, WeMath 71.30%）
- VSP 单独效果有限，但与 SFR/RRM 组合后提供额外增益

**超参数消融**:

| γ (β=5.0) | Acc. | β (γ=0.2) | Acc. |
|---|---|---|---|
| 0.05 | 44.65 | 3.5 | 45.16 |
| 0.1 | 45.12 | **5.0** | **46.01** |
| **0.2** | **46.01** | 6.5 | 45.87 |
| 0.35 | 43.78 | 8.0 | 45.72 |

**Re-inference rollout 数量**:

| Rollout Size | MathVista |
|---|---|
| R=4 | 66.45 |
| **R=5** | **68.25** |
| R=6 | 67.46 |
| R=7 | 66.01 |
| R=8 | 64.98 |

- 最优 rollout 数量为 5，超过后累积误差和计算开销反而降低性能

**Retracing rate**:

| ω | MathVista |
|---|---|
| 0.1 | 66.31 |
| 0.3 | 67.01 |
| 0.5 | 66.58 |
| **0.7** | **68.25** |
| 0.9 | 66.43 |

- 最优 retracing rate 为 0.7（保留前 70% 推理内容，在后 30% 处回溯探索）
- 过低的 ω 使搜索空间退化为传统 majority voting
- 过高的 ω 使推理路径几乎确定，留给探索的空间不足

### 训练动态
- CSRS 维持更低的高置信度样本比例（frequency ∈ [0.8, 1.0]），避免 model collapse
- Entropy 下降速率慢于 majority voting → 保持更好的探索能力
- t-SNE 可视化显示：maternal trajectories 逐渐收缩到狭窄语义空间，而 CSRS 的 re-inference trajectories 在锚点周围维持更大的搜索空间

## 与 SPC/URLVR 研究的关系

### 直接关联
1. **最直接的 URLVR 相关工作**: CSRS 是当前最相关的多模态无监督自演化方法，与我们 SPC 的研究方向高度吻合——同样关注无 ground-truth 下如何提供高质量训练信号
2. **SFR 的连续 reward 与 SPC 的 step-level credit**: SFR 通过频率方差校准将 binary reward 转化为连续信号，本质上是一种 outcome-level 的 soft credit。SPC 的 step-level probing 可以看作 SFR 的细粒度版本——SFR 在 trajectory-level 给连续分，SPC 在 step-level 分配 credit
3. **RRM 的锚点回溯与 SPC 的 semantic consistency**: RRM 从锚点重新推理并检查答案是否一致，与 SPC 的核心思想（检查推理步骤的语义一致性）在精神上一致。差异在于：RRM 是 **answer-level** 的一致性检测，SPC 是 **step-level** 的语义一致性检测
4. **Model collapse 的理论分析**: CSRS 对 Contrastive Factor 的分析（Eq 4-5）为 SPC 的 reward 设计提供了理论指导——任何 reward 设计都应满足 $G_{SR} < G_{MV}$ 以避免分布极化
5. **长尾推理路径探索**: CSRS 证明了在无监督设定下，低频但正确的推理路径极其重要。SPC 的 probing 机制可以识别这些低频路径中的关键正确步骤，实现比 SFR 更精准的 credit assignment

### 对 Co-Evolving Verifier 的启示
- **频率 vs 正确性**: CSRS 用 frequency 作为 correctness 的代理，但承认这存在 bias。Co-Evolving Verifier 可以在此基础上提供更可靠的 correctness signal，替代 frequency-based pseudo-label
- **锚点策略的启发**: RRM 的固定比例锚点（ω=0.7）是一种简单但有效的策略。Verifier 可以学习自适应地选择锚点位置，基于推理的不确定性而非固定比例
- **VSP 的数据增强思路**: 视觉扰动的思路可扩展到 verifier 的训练——通过扰动输入测试推理的鲁棒性，作为 verifier 的 calibration signal

### 与其他已读论文的联系
- [[gu-2026-asymgrpo|AsymGRPO]]: AsymGRPO 的 entropy refinement 与 CSRS 的 model collapse 缓解目标一致，但 AsymGRPO 通过 advantage 调制，CSRS 通过采样策略和 reward 设计
- [[kim-2026-dbb|DBB]]: DBB 的 Bayesian smoothing 与 SFR 的连续 reward 都在缓解 reward 信号的不稳定性，但 DBB 从统计估计角度，SFR 从频率校准角度

## 面试 Q&A

### Q1: 为什么 majority voting 在无监督自演化中会导致 model collapse？CSRS 如何从理论上缓解？
**A**: 无监督自演化的策略迭代本质是指数自强化：$P_{n+1}(x) \propto P_n(x) e^{A_n(x)/\beta}$。Majority voting 的 binary reward 导致 Contrastive Factor $G_{MV} = e^\eta$，即高频样本与长尾样本的概率比每次迭代按最大速率 $e^\eta$ 指数膨胀。初始分布中的微小偏置被快速极化，模型退化为确定性映射。CSRS 的 SFR 将 binary reward 替换为连续信号，使 $G_{SR} = \exp(\eta(\rho - \epsilon))$，因 $\rho - \epsilon < 1$，极化速率被有效抑制。同时 RRM 通过回溯重推理扩展长尾样本的覆盖，从采样端对抗分布收缩。

### Q2: RRM 的 retracing rate ω=0.7 意味着什么？为什么不是 0.5 或更小的值？
**A**: ω=0.7 表示保留前 70% 的推理内容作为 context，从剩余 30% 处开始重新推理。数学推理中，**前期步骤通常是问题理解和方案规划**，**后期步骤是具体计算和推导**。保留前 70% 保证了问题理解和主要推理方向不变，在关键的计算推导阶段引入多样性。ω 过小（如 0.1）几乎等于从头采样，搜索空间退化为传统 majority voting 的指数级空间 $\mathcal{O}(b^L)$；ω 过大（如 0.9）使路径几乎确定，探索空间极小。0.7 是局部探索密度与搜索空间大小的最优平衡点。这也呼应了 **Critical Reasoning Pivots** 假设——数学推理中的关键决策节点通常出现在推理链的后半段。

### Q3: CSRS 与 SPC 的本质区别是什么？能否组合？
**A**: CSRS 工作在 **trajectory-level 的采样和 reward 设计**，通过 RRM 扩展采样空间 + SFR 提供连续 reward；SPC 工作在 **step-level 的 credit assignment**，通过 probing + semantic consistency 为每个推理步骤分配精细 credit。两者在不同粒度解决相关但不同的问题：CSRS 回答"如何采样更好的 trajectory 并给出更好的 trajectory-level reward"，SPC 回答"给定 trajectory，如何识别哪些步骤贡献了正确推理"。组合方式：用 CSRS 的 RRM 扩展采样空间 → 用 SPC 的 probing 在 step-level 分配 credit → 用 SFR 的频率校准作为 SPC credit 的补充验证信号。这实现了从采样到 reward 到 credit 的全链路优化。
