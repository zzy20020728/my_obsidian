---
title: "How Far Can Unsupervised RLVR Scale LLM Training?"
type: paper
tags: [URLVR, survey, intrinsic-reward, external-reward, sharpening, model-collapse, MCS, TTRL, EMPO, generation-verification-asymmetry, Tsinghua, ICLR-2026]
created: 2026-04-08
updated: 2026-04-08
sources: [https://arxiv.org/abs/2603.08660, https://github.com/PRIME-RL/TTRL]
status: active
---

# How Far Can Unsupervised RLVR Scale LLM Training?

## 基本信息
- **作者**: Bingxiang He*, Yuxin Zuo*+ (Project Lead), Zeyuan Liu*, Shangziqui Zhao*, Zixuan Fu, Junlin Yang, Cheng Qian, Kaiyan Zhang, Yuchen Fan, Ganqu Cui, Xiushi Chen, Youbang Sun, Xingtai Lv, Xuekai Zhu, Li Sheng, Ran Li, Huan-ang Gao, Yuchen Zhang, Bowen Zhou, Zhiyuan Liu, Ning Ding
- **机构**: Tsinghua University, Shanghai AI Lab, Xi'an Jiaotong University, UIUC, Frontis.AI, SJTU, Peking University
- **年份**: 2026 (arXiv: Mar 2026)
- **会议/期刊**: **ICLR 2026**
- **链接**: https://arxiv.org/abs/2603.08660
- **代码**: https://github.com/PRIME-RL/TTRL

## 一句话总结
> URLVR 领域最重要的综合分析论文：建立统一分类体系（intrinsic vs external），**证明所有 intrinsic 方法本质上都在做 sharpening model's initial distribution**，发现 rise-then-fall pattern 不可避免，提出 Model Collapse Step (MCS) 作为 RL trainability 的实用指标，并初步验证 external reward（self-verification）可以突破 intrinsic 的天花板。

## 摘要
本文对 URLVR 方法进行系统性综合分析。将方法分为 **intrinsic**（内部信号）和 **external**（外部信号）两类。提出统一理论框架，**数学证明**所有 intrinsic 方法收敛到 sharpening model's initial distribution。关键发现：(1) 所有 intrinsic 方法呈现 **rise-then-fall pattern**，collapse 时间由 model prior 决定；(2) 小数据集（32-128 samples）可避免 collapse，使 test-time training 成为安全应用场景；(3) **Model Collapse Step (MCS)**——Reward Accuracy 降至 1% 以下的步数——是 RL trainability 的高效预测指标（5.6x faster than full RL, 无需 GT）；(4) External reward（self-verification、unlabeled data）可以 escape confidence-correctness ceiling。

## 核心贡献
1. **统一分类体系**: Intrinsic rewards (certainty-based + ensemble-based) vs External rewards (unlabeled data + generation-verification asymmetry)
2. **Sharpening Theorem (Theorem 1)**: 数学证明 majority voting reward 下策略几何收敛到初始 majority answer 的确定性策略，并推广到所有 intrinsic rewards
3. **Rise-then-Fall 是本质限制**: 穷举 5 种 intrinsic reward x 4 种超参数组合，collapse 不可避免，只是时间早晚的区别
4. **三种不同的 Collapse Failure Modes**: Gradual degradation (Self-Certainty, Majority Voting) / Length collapse (Probability) / Repetition collapse (Token/Trajectory Entropy)
5. **Per-Problem Sharpening Analysis**: 训练是 amplification 而非 correction——22/25 问题只是加强了初始偏好（无论对错）
6. **Small Dataset Safety**: 32-128 samples -> localized overfitting -> 避免 systematic policy shift -> test-time training 安全
7. **Model Collapse Step (MCS)**: Reward Accuracy < 1% 的步数，5.6x faster than full RL，无需 GT labels，比 pass@k 更准确
8. **External Reward 初步验证**: Self-verification on Countdown task，Reward Accuracy 先降后恢复并稳定 > 0.5，远优于 intrinsic

## 方法详解

### 1. URLVR 统一分类体系

#### Intrinsic Reward Methods（模型自身信号）

**Certainty-Based（基于确定性）**: 奖励高置信预测

| 方法 | 估计器 | 描述 |
|------|--------|------|
| RLIF | Self-Certainty | avg KL divergence from uniform distribution |
| EM-RL | Trajectory-Level Entropy | avg log-prob over sequence |
| EM-RL, RENT | Token-Level Entropy | negative avg token entropy |
| RLSC | Probability | product of token probabilities |
| RLSF | Probability Disparity | top-2 token probability gap |

**核心洞察**: 所有 certainty-based rewards 本质上都在 reward high-confidence predictions，只是数学形式不同。

**Ensemble-Based（基于集成）**: 奖励跨样本一致性

| 方法 | 估计器 |
|------|--------|
| [[wiki/papers/zuo-2025-ttrl|TTRL]], SRT, ETRL, SeRL, SQLM, R-Zero | Majority Voting |
| Co-Reward | Majority Voting across Rephrased Questions |
| RLCCF | Self-consistency Weighted Voting |
| [[wiki/papers/zhang-2025-empo|EMPO]] | Semantic Similarity (semantic clustering) |
| CoVo | Trajectory Consistency and Volatility |

**Proposer-Solver 架构**: R-Zero（50% uncertainty 最大化）、SQLM、SeRL、CPMobius 等。

#### External Reward Methods（外部信号）

**利用无标注数据生成 Reward**:
- **RPT**: 在 unlabeled text 上做 next-token prediction，用预测正确率作为 reward
- **TPT**: 通过 step-by-step reasoning 预测 token
- **DuPO**: dual reconstruction objective
- **SEAL**: 自动生成 QA pairs -> meta-learning
- **Nemotron-CrossThink**: 从 CommonCrawl 挖掘多领域 QA

**利用 Generation-Verification Asymmetry**:
- **LADDER**: 不定积分——求原函数难，验证（数值代入）简单
- **RLSR**: Countdown 数学游戏——构造难，检查 trivial
- **Absolute Zero**: 代码生成——写代码难，跑测试 instant
- **AlphaProof**: 定理证明——发现证明难，Lean 验证 seconds

**关键区别**: Intrinsic rewards 受限于模型已有知识；External rewards 随数据/计算规模增长，不会退化。

### 2. Sharpening Mechanism 理论（核心贡献）

以 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的 majority voting reward 为例，从 KL-regularized RL objective 出发：

$$\max_{\pi_\theta} \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(x,y)] - \beta D_{KL}[\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)]$$

最优策略闭式解：

$$\pi_\theta^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

由于 majority voting reward 是 binary（0 或 1），majority answer 的概率被放大 $e^{1/\beta}$ 倍：

$$p_{maj}^{*,(k+1)} = \frac{p_{maj}^{(k)} \cdot e^{1/\beta}}{p_{maj}^{(k)} \cdot e^{1/\beta} + (1 - p_{maj}^{(k)})}$$

实际训练中（一步梯度更新）：$p_{maj}^{*,(k+1)} \geq p_{maj}^{(k+1)} \geq p_{maj}^{(k)}$

**Theorem 1 (Geometric Convergence to Deterministic Policy)**:

在两个假设下：
- **(A1) Majority Stability**: maj_k(Y_k) = maj_0(Y_0) for all k（N 足够大时成立）
- **(A2) Effective Learning**: p_maj^(k+1) > p_maj^(k) for all k

p_maj^(k) 以速率 rho = e^{-1/beta} 几何收敛到 1，策略收敛到：

$$\lim_{k \to \infty} \pi_\theta^{(k)}(y|x) = \begin{cases} \frac{\pi_{ref}(y|x)}{\sum_{y'} \pi_{ref}(y'|x)}, & \text{if ans}(y) = \text{maj}_0(Y_0) \\ 0, & \text{otherwise} \end{cases}$$

**Unified Reward Framework**: 论文进一步证明所有 intrinsic rewards 都可以通过统一框架理解——manipulating cross-entropy between carefully chosen distributions，从而推广 sharpening 分析到所有方法。

**核心含义**:
- 当 maj_0 = correct answer -> sharpening 加强正确答案 -> accuracy 上升
- 当 maj_0 = wrong answer -> sharpening 加强错误答案 -> **catastrophic failure**

### 3. Rise-then-Fall Pattern（实验验证）

训练 Qwen3-1.7B-Base on DAPO-17k，比较 majority voting training vs ground-truth training：

- **Early phase**: Majority voting training 匹配甚至超过 GT training
- **Later phase**: Majority Voting Reward 持续上升，但 Reward Accuracy 和 validation performance 下降 -> **reward hacking**
- Actor Entropy 下降速度比 GT training 更快

**穷举调参实验**: 对 5 种 intrinsic reward x {temperature, mini-batch size, KL regularization, rollout number} 进行网格搜索——某些设置延迟 collapse（mini-batch size 和 rollout number 很关键），但 **nearly all settings eventually degrade**。即使用最稳定的超参组合，collapse 仍在 ~1000 步（约 4 epochs）发生。

### 4. 三种 Failure Modes

| Failure Mode | 方法 | 表现 |
|:------------|:-----|:-----|
| **Gradual degradation** | Self-Certainty, Majority Voting | 最慢衰退，Label Accuracy 保持较高 |
| **Length collapse** | Probability | 奖励短序列（概率乘积偏好短输出），生成过短答案 |
| **Repetition collapse** | Token-Level / Trajectory-Level Entropy | 平均 entropy 被重复 high-prob tokens 最小化，生成重复文本 |

Self-Certainty 最稳定（对 uniform 分布做 sharpening，较温和）；Majority Voting 稳定（答案级而非 token 级）。

### 5. Per-Problem Sharpening 分析

在 25 个 MATH-500 问题上逐个训练（100 epochs each），观察四种 patterns：

| Pattern | 占比 | 描述 |
|---------|:----:|------|
| **Amplifying success** | 大多数 | 初始 greedy 正确，训练加强信心 |
| **Amplifying failure** | 部分 | 初始 greedy 错误，训练加强错误信心 |
| **Wrong -> Correct** | 3/25 (12%) | 最高奖励样本多数正确，引导模型翻转 |
| **Correct -> Wrong** | 少数 | 最高奖励样本不稳定，导致退化 |

**核心结论**: 训练是 **amplification 而非 correction**——22/25 问题只是加强了初始偏好。

**OOD 泛化惊喜**: 在 6 个初始 majority vote 错误的问题上训练，虽然 Train Label Accuracy 约等于 0，但在两个 OOD 问题上 Test Label Accuracy 从 0 -> 1。说明 sharpening 可以跨问题泛化——在问题 A 上放大错误，在问题 B 上可能放大正确的能力。

### 6. Small Dataset Safety & Test-Time Training

**Dataset Size 实验**: 在 {32, 128, 512, 2048, 8192, 16384} 样本上训练：
- **32-128 samples**: Ground Truth Reward 和 Majority Voting Reward 同步上升并稳定，**无 collapse**
- **512+**: 开始出现 rise-then-fall pattern
- **原理**: 小数据集 -> localized overfitting -> 不会 systematic policy shift

**Test-Time Training**: 40 道 AMC23 问题上 TTT -> Majority Voting Reward 上升并稳定 -> AMC23 和 AIME24 都提升 -> **无 collapse**。对比同时在 DAPO-17k 上训练 -> 出现 rise-then-fall。

**极端实验**: 选 32 道初始 majority vote 几乎全部错误的问题训练——Label Accuracy 约等于 0（100 步后持续为 0），但 AIME24 和 AMC23 仍然提升！说明小数据集的 localized overfitting 不同于大数据集的 systematic shift。

### 7. Model Collapse Step (MCS)

**定义**: 训练过程中 Reward Accuracy 降至 1% 以下的步数。

**MCS 预测 RL Trainability**: 7 个模型（OLMo/LLaMA/Qwen 三家族）对比：
- **GT Gain**: 用 GT reward 做标准 RL 的提升（gold standard）
- **Pass@k Gain**: pass@256 - pass@1
- **MCS**: Reward Accuracy < 1% 的步数

MCS 与 GT Gain 的相关性 > Pass@k 与 GT Gain 的相关性。

**效率**: MCS 计算仅需 **1.19B tokens**（GT Gain 需 6.66B tokens），**5.6x faster**，且**不需要 GT labels**。

| 指标 | 计算成本 | 总 Tokens | 需要 GT? |
|------|----------|-----------|----------|
| GT Gain | 7k x 8 x 17k x 7 | 6.66B (baseline) | Yes |
| Model Collapse Step | 7k x 8 x 662 x 32 | **1.19B (5.6x faster)** | **No** |

**加速**: 使用 aggressive hyperparameters（mini-batch=1, rollout=8）可以进一步加速 collapse 到最快 14 步，同时**保持模型排名一致**。

**实际意义**: 选择 base model 时，不需要跑完整 RL，只需跑短暂 intrinsic URLVR -> 看 MCS -> MCS 越大的模型越适合做 RL 训练。

### 8. External Reward: Self-Verification 实验

在 Countdown 任务上（4k train / 1k val），比较：
- **Self-Verification**: 模型生成解 -> 用 verification prompt 自评 -> binary correctness reward
- **Trajectory-Level Entropy**: intrinsic baseline
- **Oracle Supervision**: GT reward

**结果**: Self-Verification >> Trajectory-Level Entropy，接近 Oracle。
- Reward Accuracy 初始下降（~step 200，policy 试图 exploit verifier）-> 然后恢复并稳定 > 0.5
- Ground Truth Reward 持续上升
- **Instruction-aligned model (Qwen3-1.7B) >> Base model**: 起始 >60% accuracy，对两种 verification prompt 都 robust

**关键启示**: External reward 不退化因为验证器（编译器/数值检查/Lean）不随模型提升而变弱。

## 实验结果摘要

### 主要发现表

| 发现 | 证据 |
|------|------|
| 所有 intrinsic 方法 rise-then-fall | 5 种方法 x 多组超参，~1000步内必 collapse |
| Collapse 时间由 model prior 决定 | SFT model > base model（Qwen: SFT 不 collapse vs base ~200步；LLaMA: SFT 延迟 collapse） |
| 小数据集安全 | 32-128 samples 无 collapse（600步内） |
| MCS 准确预测 RL trainability | 与 GT Gain 高相关，优于 pass@k |
| External > Intrinsic（长期） | Self-verification Reward Accuracy 恢复稳定，intrinsic 必 collapse |
| 训练是 amplification 非 correction | 22/25 问题只加强初始偏好 |
| OOD 泛化存在 | 在错误问题上训练仍可提升 OOD 正确问题 |

## 与其他工作的关系

### 统一理论解释力

| 论文 | He et al. 的理论解释 |
|------|---------------------|
| [[wiki/papers/zuo-2025-ttrl|TTRL]] | Majority voting = ensemble-based intrinsic reward -> sharpening -> 小数据 TTT 安全，大数据最终 collapse |
| [[wiki/papers/zhang-2025-empo|EMPO]] | Semantic entropy minimization = sharpening via semantic clustering -> Entropy Thresholding 是启发式过滤 misaligned 样本 |
| [[wiki/papers/ghimire-2026-prism|PRISM]] | Self-certainty/token entropy/trajectory entropy 全部 collapse = certainty-based intrinsic 的必然结果 |
| [[wiki/papers/wu-2026-self-judge|Self-Judge]] | SC 频率 = ensemble-based intrinsic；frozen Judge = 半 external -> 混合方案部分缓解 |
| [[wiki/papers/rahman-2025-spark|SPARK]] | Trained PRM = external reward -> 不受 rise-then-fall 限制 |
| [[wiki/papers/royer-2026-mcnig|MCNIG]] | Trained PRM = external reward -> 稳定 |
| [[wiki/papers/wu-2026-spae|SPAE]] | Confidence = certainty-based（会 sharpen）；Correctness = 需要 GT（external） |
| [[wiki/papers/zhang-2026-grad2reward|Grad2Reward]] | Self-judging = 半 external（frozen copy）；gradient attribution 不同于纯 intrinsic |

### 对 Step-Level SE 研究方案的影响

[[wiki/synthesis/step-level-se-proposal|双层架构]] 中的 Semantic Certainty (SC) 信号：
- SC = 1 - SE/SE_max -> 本质上是 **ensemble-based intrinsic reward**（跨 probe 续写的语义一致性）
- 根据 He et al. 理论，SC 长期训练也会 sharpen -> 可能 collapse
- **但**：TTRL 做 outcome anchor -> 限制了 step-level sharpening 的影响范围
- **关键问题**：SC 作为 step-level signal 的 sharpening 速度如何？是否比 outcome-level 的 majority voting 更快/更慢？

## 面试相关 Q&A

### Q1: URLVR 领域的 intrinsic vs external reward 有什么区别？🔴
**A**: Intrinsic rewards 完全来自模型自身（certainty-based: logits 置信度；ensemble-based: 多次采样一致性）。External rewards 来自模型外部（trained PRM、代码执行、unlabeled data 的 next-token prediction）。He et al. (ICLR 2026) 证明所有 intrinsic 方法都在做 sharpening——锐化模型初始分布，成功依赖 confidence-correctness alignment。External rewards 不受此限制。

### Q2: 什么是 Sharpening Mechanism？为什么是所有 intrinsic 方法的统一机制？🔴
**A**: 数学证明（Theorem 1）：以 majority voting 为例，每步 RL 更新将 majority answer 的概率放大 e^{1/beta} 倍，形成 "rich-get-richer" 动态，几何收敛到以初始 majority answer 为唯一输出的确定性策略。通过 Unified Reward Framework，证明 certainty-based rewards（entropy minimization 等）也收敛到相同的 sharpening 行为——不同 intrinsic rewards 只是 cross-entropy between different distributions 的不同实例化。

### Q3: Rise-then-fall pattern 是否可以通过超参调优避免？🔴
**A**: 不能。论文对 5 种方法 x 4 种超参（temperature/mini-batch/KL/rollout number）进行穷举搜索，即使用最稳定的超参组合，collapse 仍在 ~1000 步发生。这是 fundamental limitation，不是 engineering problem。

### Q4: Model Collapse Step (MCS) 如何使用？🟡
**A**: 定义：intrinsic URLVR 训练中 Reward Accuracy 降至 <1% 的步数。用途：(1) 评估 base model 的 RL trainability——MCS 越大越适合做 RL；(2) 比 pass@k 更准确预测 GT RL gains；(3) 仅需 5.6x 更少 tokens，且不需要 GT labels；(4) 可用 aggressive hyperparameters 加速到 ~14 步。

### Q5: 小数据集为什么能避免 collapse？这对实践有什么启示？🟡
**A**: 32-128 samples 训练是 localized overfitting 而非 systematic policy shift。即使所有训练样本的 majority vote 都是错的，OOD 问题仍可能提升。启示：intrinsic URLVR 最安全的应用场景是 **test-time training on small domain-specific datasets**（如 TTRL 在 40 道 AMC23 上 TTT）。

### Q6: External reward 为什么能突破 intrinsic 的天花板？🔴
**A**: 两个原因：(1) External reward 不随模型提升而退化——编译器/Lean 验证器/数值检查的可靠性不变；(2) 可以随数据/计算规模增长——unlabeled text 无限，verification computation 可扩展。Self-verification 实验显示 Reward Accuracy 先降后恢复稳定（模型学会 exploit -> 但 verifier 不退化 -> 最终正确学习），远优于 intrinsic。

## 个人笔记

### 对研究方案的关键影响

1. **Semantic Certainty (SC) 是 ensemble-based intrinsic signal** -> 根据 He et al. 理论，纯 SC 长期训练会 sharpen -> 需要 outcome anchor (TTRL) 来约束
2. **双层架构 (TTRL + SPAE-SE) 的理论合理性增强**：
   - TTRL (outcome layer) = ensemble-based intrinsic，在 TTT 设置下安全（小数据集）
   - SC (step layer) = ensemble-based intrinsic，但作为 step-level 辅助信号，受 TTRL outcome 约束
   - 整体架构：**两层 intrinsic signals 互相约束，比单层 intrinsic 更安全**
3. **MCS 可用于实验**: Phase 1 验证 SC 时，可以用 MCS 快速评估 SC 信号的质量
4. **External reward 启示**: 长远来看，step-level SE 如果不够稳定，可以探索 generation-verification asymmetry 的 step-level 版本

### 论文的方法论价值
- 这是 URLVR 领域**唯一有严格数学证明**的工作（Theorem 1 + Unified Reward Framework）
- 实验设计极其 thorough：5 种方法 x 穷举超参 x 7 个模型 x 多数据集大小 x per-problem 分析
- MCS 概念简洁有效，具有很高的实用价值

### 与 TTRL 论文的关系
- TTRL (Zuo et al. 2025) 和本文 (He et al. 2026) 有大量共同作者（Yuxin Zuo 是两篇的 Project Lead）
- 本文是 TTRL 的理论深化——解释了 TTRL 为什么有效（Lucky Hit + small dataset TTT）以及什么时候会失败（large-scale training -> sharpening -> collapse）
- TTRL 的 "超越 maj@n 上界" 现象在本文中被更精确地解释：OOD 泛化 + localized sharpening 在某些问题上恰好 align with correctness
