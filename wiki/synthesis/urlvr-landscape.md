---
title: URLVR 领域综述：无监督/无参考强化学习推理
type: synthesis
tags: [URLVR, 综述, 对比分析, reward-signal, PRM, self-consistency, RAG, sharpening, MCS, failure-modes, external-reward, CoVo, consistency, volatility, dual-consensus, distribution-aware, tool-verification, calibration, contrastive-learning, GRPO-flaw, small-world, diversity, metacognition, co-evolution, label-noise, credit-assignment, Bayesian-reward, variational-synthesis, entropy-decomposition, policy-reward-co-evolution]
created: 2026-04-07
updated: 2026-04-10
sources: [wiki/papers/zuo-2025-ttrl.md, wiki/papers/zhang-2025-empo.md, wiki/papers/zhang-2025-covo.md, wiki/papers/rahman-2025-spark.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/wu-2026-self-judge.md, wiki/papers/royer-2026-mcnig.md, wiki/papers/wang-2026-prorag.md, wiki/papers/tan-2026-ctrl-rag.md, wiki/papers/he-2026-urlvr-scale.md, wiki/papers/wu-2026-spae.md, wiki/papers/zhang-2026-grad2reward.md, wiki/papers/du-2026-dual-consensus.md, wiki/papers/du-2026-dare.md, wiki/papers/liao-2026-t3rl.md, wiki/papers/ma-2026-dcpo.md, wiki/papers/cui-2026-clipo.md, wiki/papers/wang-2026-pipo.md, wiki/papers/wang-2026-sarl.md, wiki/papers/huang-2026-darl.md, wiki/papers/tan-2026-meta-ttrl.md, wiki/papers/wang-2026-v-zero.md, wiki/papers/yang-2026-distribttrl.md, wiki/papers/bai-2026-ttvs.md, wiki/papers/yang-2026-olr.md, wiki/papers/gu-2026-asymgrpo.md, wiki/papers/kim-2026-dbb.md, wiki/papers/yu-2026-csrs.md, wiki/papers/ai-2026-shape.md, wiki/papers/plesner-2026-imperfect-verifier.md, wiki/papers/wang-2026-self-guide.md]
status: active
---

# URLVR 领域综述：无监督/无参考强化学习推理

## 领域定义

**URLVR (Unsupervised RL for Verifiable Reasoning)** / **Reference-Free RLVR**：在没有 ground-truth 答案标注的情况下，通过强化学习提升 LLM 推理能力的研究方向。

**核心挑战**：标准 RLVR（如 DeepSeek-R1）依赖 ground-truth 做 rule-based verification，但大量实际场景（开放域推理、多模态推理、RAG、复杂任务）无法获取标注。如何在没有标注的情况下构建可靠的训练信号？

## 三十一篇核心论文速览

| 论文 | 机构 | 核心方法 | Reward 来源 | 任务 | 年份 |
|------|------|----------|-------------|------|------|
| [[wiki/papers/zuo-2025-ttrl\|TTRL]] | Tsinghua + Shanghai AI Lab | Majority voting pseudo-reward + online RL | 纯内部（cross-sample consensus） | 数学/通用推理 | 2025 |
| [[wiki/papers/zhang-2025-empo\|EMPO]] | Tianjin U + Tencent | 语义熵最小化 | 纯内部（semantic entropy） | 数学/通用推理 | 2025 |
| [[wiki/papers/zhang-2025-covo\|CoVo]] | Zhejiang U + Alibaba + NTU | Consistency + Volatility reward | 纯内部（process-aware likelihood consistency） | 数学/通用推理 | 2025 |
| [[wiki/papers/rahman-2025-spark\|SPARK]] | Amazon + UCLA | 三阶段 PRM 训练 | 外部（trained PRM） | 数学推理 | 2025 |
| [[wiki/papers/ghimire-2026-prism\|PRISM]] | ASU + AWS | PRM + self-certainty 混合 | 混合（PRM + 内部信号） | 数学推理 | 2026 |
| [[wiki/papers/wu-2026-self-judge\|Self-Judge]] | OPPO + Tsinghua | Actor-Judge + distributional | 混合（SC + Judge modulation） | 多模态推理 | 2026 |
| [[wiki/papers/royer-2026-mcnig\|MCNIG]] | IBM Research | 信息论 PRM 数据生成 | 外部（trained PRM） | 数学/代码/SQL/医学 | 2026 |
| [[wiki/papers/wang-2026-prorag\|ProRAG]] | Renmin University | 四阶段 RAG RL + MCTS PRM | 混合（PRM + outcome F1） | Multi-hop QA | 2026 |
| [[wiki/papers/tan-2026-ctrl-rag\|CTRL-RAG]] | Ant Group | Contrastive Likelihood Reward | 混合（CLR + accuracy） | RAG Faithfulness | 2026 |
| [[wiki/papers/he-2026-urlvr-scale\|How Far URLVR Scale?]] | Tsinghua | 统一分析 + sharpening 理论 | 综合分析（intrinsic vs external） | 综合 | 2026 (ICLR) |
| [[wiki/papers/wu-2026-spae\|SPAE]] | — | Confidence + Correctness reward | 混合（certainty + GT） | 数学推理 | 2026 |
| [[wiki/papers/zhang-2026-grad2reward\|Grad2Reward]] | — | Gradient attribution self-judging | 半外部（frozen copy） | 数学推理 | 2026 |
| [[wiki/papers/du-2026-dual-consensus\|DCRL]] | Beihang | Anchor-Explorer dual consensus + harmonic mean | 纯内部（dual-view consensus） | 数学/通用推理 | 2026 |
| [[wiki/papers/du-2026-dare\|DARE]] | — | Uncertainty-normalized distribution reward | 纯内部（distribution-aware） | 数学推理 | 2026 (ICML) |
| [[wiki/papers/liao-2026-t3rl\|T³RL]] | LMU Munich + Stanford | Code execution verification + weighted MV | 混合（tool verification + MV） | 数学推理 | 2026 |
| [[wiki/papers/ma-2026-dcpo\|DCPO]] | CAS | Masked gradient decoupling accuracy vs calibration | N/A（训练方法，非 reward） | 数学推理 | 2026 |
| [[wiki/papers/cui-2026-clipo\|CLIPO]] | Alibaba Qwen | InfoNCE contrastive on successful rollouts | 辅助信号（contrastive regularization） | 数学推理 | 2026 |
| [[wiki/papers/wang-2026-pipo\|PIPO]] | Beihang + PKU | PIRL framework + dual-stage explore-verify | N/A（优化框架改进） | 数学推理 | 2026 |
| [[wiki/papers/wang-2026-sarl\|SARL]] | Purdue | Small-world network topology reward | 纯内部（结构化 reward） | 数学 + 开放域 | 2026 |
| [[wiki/papers/huang-2026-darl\|DARL]] | 厦门大学 + Kuaishou | Dynamic diversity reward | 需要 GT（diversity bonus） | 通用推理 | 2026 |
| [[wiki/papers/tan-2026-meta-ttrl\|Meta-TTRL]] | — | Metacognitive rubric-based T2I reward | 纯内部（self-introspection） | T2I 生成 | 2026 |
| [[wiki/papers/wang-2026-v-zero\|V-Zero]] | Zhejiang U | Questioner-Solver co-evolution + Dual-Track Reward | 纯内部（intuition vs reasoning） | 多模态 VLM 推理 | 2026 |
| [[wiki/papers/yang-2026-distribttrl\|DistriTTRL]] | SEU+Kuaishou | GMM+shift correction+diversity penalty | 纯内部（confidence distribution） | 数学推理 | 2026 |
| [[wiki/papers/bai-2026-ttvs\|TTVS]] | HKUST | Variational synthesis+hybrid exploration | 纯内部（augmented consensus） | 数学推理 | 2026 |
| [[wiki/papers/yang-2026-olr\|OLR]] | ZJU+Ant | Online Label Refinement | N/A（噪声鲁棒性分析） | 数学推理 | 2026 |
| [[wiki/papers/gu-2026-asymgrpo\|AsymGRPO]] | NCSU | Asymmetric entropy modulation | N/A（优化框架改进） | 数学推理 | 2026 |
| [[wiki/papers/kim-2026-dbb\|DBB]] | KAIST | Beta-Bernoulli reward estimation | 纯内部（Bayesian posterior） | 数学推理 | 2026 |
| [[wiki/papers/yu-2026-csrs\|CSRS]] | Tsinghua | Retracing+Softened Reward+VSP | 纯内部（frequency consensus） | 多模态几何推理 | 2026 |
| [[wiki/papers/ai-2026-shape\|SHAPE]] | — | Hierarchical segment+token credit | N/A（credit assignment改进） | 数学推理 | 2026 |
| [[wiki/papers/plesner-2026-imperfect-verifier\|Imperfect Verifier]] | — | Noise robustness analysis | N/A（理论分析） | 数学推理 | 2026 |
| [[wiki/papers/wang-2026-self-guide\|Self-Guide]] | — | Policy-reward co-evolution loop | 混合（internal reward+environment） | Agent tasks | 2026 |

---

## 多维分类体系

### 维度一：按 Reward 信号来源

```
 纯内部信号 ─────────────── 混合信号 ──────────── 纯外部信号
    │      │      │         │    │    │               │      │
  TTRL    EMPO   CoVo     PRISM  S-J  CTRL-RAG       SPARK  MCNIG
(maj vote)(sem ent)(Con+Vol)(PRM+SC)(SC+J)(CLR+acc)  (PRM)  (PRM)
  DCRL    DARE   SARL      ProRAG  T³RL  Self-Guide
(dual-view)(dist)(topology)(PRM+F1)(tool+MV)(co-evo)
 Meta-TTRL V-Zero DistriTTRL
(self-intro)(co-evo)(GMM conf)
  TTVS     DBB    CSRS
(var synth)(Bayes)(freq cons)

 优化框架改进 ──────────── 需要 GT ────── Label Noise 分析
    │      │      │   │        │              │      │
  DCPO   PIPO   CLIPO AsymGRPO DARL          OLR  Imperfect
(grad decouple)(PIRL)(InfoNCE)(asym ent)(diversity)(online refine)(noise theory)
  SHAPE
(hier credit)

 多模态自进化
    │      │      │
 Meta-TTRL V-Zero CSRS
(T2I rubric)(Q-S)(retrace+VSP)
```

| 类型 | 代表 | 优势 | 劣势 |
|------|------|------|------|
| **纯内部** | TTRL, EMPO, CoVo, DCRL, DARE, SARL, Meta-TTRL, V-Zero, DistriTTRL, TTVS, DBB, CSRS | 无需任何外部模型，完全自主 | TTRL: 无 step-level credit；EMPO: 长期训练 [[reward-hacking\|reward hack]]；CoVo: 更强的过程信号但底层仍是 likelihood consistency；本质都受 sharpening 风险约束 |
| **混合** | PRISM, Self-Judge, ProRAG, CTRL-RAG, T³RL, Self-Guide | 互补信号，更稳健 | 需要调节多信号权重（γ, β 等超参），T³RL 需要 code interpreter |
| **纯外部** | SPARK, MCNIG | 最稳定（stationary reward） | 需要额外训练 PRM，计算成本高 |
| **优化框架改进** | DCPO, PIPO, CLIPO, AsymGRPO, SHAPE | 从优化层面提升训练稳定性和效果 | 不直接改进 reward 信号，而是改进如何利用信号 |
| **Label Noise 分析** | OLR, Imperfect Verifier | 揭示 RLVR 对噪声的固有鲁棒性，提供实用验证器设计指南 | 非方法提出，偏分析和理论 |
| **需要 GT** | DARL | 多样性 bonus 有效 | 非纯 URLVR，需要 ground-truth |

### 维度二：按打分粒度

| 粒度 | 论文 | 描述 |
|------|------|------|
| **答案级 (Outcome-level)** | TTRL, EMPO, Self-Judge, CTRL-RAG, DCRL, DARE, T³RL, SARL, DARL, V-Zero, DistriTTRL, TTVS, DBB, OLR, Imperfect Verifier | 只评估最终答案的质量/一致性/忠实度 |
| **轨迹级过程感知 (Trajectory-level)** | CoVo | 看中间状态是否持续支持最终答案，但 reward 仍落在整条轨迹上 |
| **步骤级 (Step-level)** | SPARK, PRISM, MCNIG, SHAPE, Self-Guide | 评估每个推理步骤的正确性 |
| **双粒度 (Dual-granularity)** | ProRAG, [[wiki/synthesis/step-level-se-proposal\|SPC 提案]] | 同时使用 outcome + step-level 信号 |
| **训练稳定性改进** | DCPO, PIPO, CLIPO, AsymGRPO | 不直接改进 reward 粒度，而是从优化层面改进训练过程：DCPO 解耦 accuracy/calibration 梯度，PIPO 修复 GRPO 梯度爆炸，CLIPO 通过 contrastive learning 抑制 spurious reasoning，AsymGRPO 分解 informative/spurious entropy |

### 维度三：按 PRM 训练数据生成方式

| 方法 | 论文 | 策略 | 复杂度 |
|------|------|------|--------|
| **Step-level Self-consistency** | [[wiki/papers/rahman-2025-spark\|SPARK]] | 从每步重新采样 N 次看答案一致性 | O(M×N) |
| **信息论 (MCNIG)** | [[wiki/papers/royer-2026-mcnig\|MCNIG]] | 对比 correct/incorrect 答案的 log-prob 变化 | O(N) |
| **MCTS + Contrastive Labeling** | [[wiki/papers/wang-2026-prorag\|ProRAG]] | MCTS 探索 + GPT-4o 标注 sibling nodes | O(NlogN) |
| **现成 PRM** | [[wiki/papers/ghimire-2026-prism\|PRISM]] | 直接用 GenPRM-7B | 0（无需训练） |

### 维度四：按任务类型

| 任务类型 | 论文 | Base Model |
|----------|------|------------|
| **数学推理** | TTRL, EMPO, CoVo, SPARK, PRISM, MCNIG | Qwen2.5-Math-7B, Qwen2.5-3B/7B/32B, Ministral-8B/14B, LLaMA-3.x |
| **通用推理** | TTRL, EMPO, CoVo | Qwen2.5-7B (MMLU-Pro, GPQA) |
| **多模态视觉推理** | Self-Judge | Qwen2.5-VL-7B (几何、图表) |
| **代码 & SQL** | MCNIG | Ministral-8B/14B |
| **Multi-hop QA (RAG)** | ProRAG, CTRL-RAG | Qwen3-8B, Qwen3-30B-A3B |
| **医学 QA** | MCNIG | Ministral-8B/14B |

### 维度五：按优化框架

所有论文都基于 [[grpo|GRPO]]，但有细微差异：

| 论文 | 优化框架 | Advantage 计算 |
|------|----------|---------------|
| EMPO | 标准 GRPO | 语义聚类 reward 的组内 z-score |
| CoVo | Reinforce++ | Consistency/Volatility 聚合 reward + curiosity bonus |
| SPARK | 标准 GRPO | PRM reward 的组内 z-score |
| PRISM | 标准 GRPO | γ·Â_SC + Â_PRM |
| Self-Judge | 改进 GRPO | Energy-based log-sum-exp baseline |
| MCNIG | — (仅 best-of-K) | 未集成到 RL |
| ProRAG | 标准 GRPO | A^out + β·A^proc（dual-granularity） |
| CTRL-RAG | 标准 GRPO | R_CLR × R_acc（multiplicative gating） |
| DCRL | GRPO + 三级 reward | Harmonic mean consensus |
| DARE | GRPO | Distribution-aware reward (uncertainty-normalized) |
| T³RL | GRPO | Verification-weighted majority voting |
| DCPO | GRPO + masked gradient | Decoupled reasoning vs confidence gradients |
| CLIPO | GRPO/DAPO/Dr.GRPO/PRIME | InfoNCE reward augmentation |
| PIPO | GRPO/GSPO/DAPO + PIRL | Policy Improvement Reward (explore-verify) |
| SARL | GRPO (algorithm-agnostic) | Small-world topology SR(G) |
| DARL | GRPO | Dynamic diversity reward |
| Meta-TTRL | GRPO | Rubric-based geometric mean |
| V-Zero | GRPO | Dual-Track intuition-reasoning |
| DistriTTRL | GRPO + shift correction reward | GMM confidence distribution + diversity penalty |
| TTVS | GRPO + variational synthesis augmentation | Augmented consensus reward + hybrid IGE/CGE exploration |
| OLR | GRPO + online label refinement | Early Correctness Coherence + progressive self-correction |
| AsymGRPO | GRPO + asymmetric entropy modulation | Informative/spurious entropy decomposition |
| DBB | GRPO + Beta-Bernoulli posterior reward | Bayesian reward estimation + zero extra compute |
| CSRS | GRPO + softened frequency reward | Retracing Re-inference + Visual Semantic Perturbation |
| SHAPE | GRPO + hierarchical advantage | Segment-level solvability potential + token-level entropy redistribution |
| Imperfect Verifier | GRPO（analysis framework） | Noise robustness theoretical analysis |
| Self-Guide | GRPO + internal reward co-evolution | Policy-reward co-evolution loop + inference-time guidance |

### 维度六：TTRL Reward 改进方法对比

| 方法 | 改进维度 | 核心机制 | 是否需要外部资源 | AIME24 提升 |
|------|----------|----------|-----------------|------------|
| TTRL (baseline) | — | Naive majority voting | 否 | — |
| DCRL | 投票机制 | Dual consensus (anchor+explorer) | 否 | +15.1% |
| DARE | Reward 计算 | Uncertainty-normalized distribution | 否 | +25.3% |
| T³RL | Reward 锚定 | Code execution verification | 是（code interpreter） | +31.6% |
| ETTRL | 采样策略 | 高熵 token 分叉采样 | 否 | 已有数据 |
| DistriTTRL | Reward 计算 | GMM confidence distribution + shift correction | 否 | AIME24 +7.50 |
| TTVS | 数据增广 | Variational synthesis + hybrid exploration | 否 | 无标签 1.5B > 7B supervised |
| DBB | Reward 方差 | Beta-Bernoulli posterior estimation | 否 | OOD +12.49% |

**设计空间分析**：多种改进方式正交——可以先用 DCRL 改进投票机制，再用 DARE/DistriTTRL 改进 reward 计算方式，用 TTVS 增广数据多样性，用 DBB 降低 reward 方差，最后用 T³RL 引入外部验证锚定。理论上多者可以叠加使用。

### 维度七：标签噪声鲁棒性

| 方法 | 噪声类型 | 核心发现 | 实用指南 |
|------|---------|---------|---------|
| [[wiki/papers/yang-2026-olr\|OLR]] | Inactive + Active | Early Correctness Coherence；传统 small-loss selection 在 RLVR 中灾难性失败（-17.4%） | Online Label Refinement progressive self-correction |
| [[wiki/papers/plesner-2026-imperfect-verifier\|Imperfect Verifier]] | Random noise injection | 15% 噪声率下性能仅降 2pp；"moderate accuracy + high precision" 原则 | 优先保证验证器精确率（precision）而非召回率（recall） |

**关键共识**：RLVR 对验证噪声具有固有鲁棒性（Imperfect Verifier 证明 15% 噪声仍可接受），但传统 SFT 时代的噪声处理方法（如 small-loss selection）在 RL 中不适用（OLR 发现 -17.4% 灾难性退化），因为 RL 的 loss landscape 与 SFT 本质不同。

---

## 统一理论框架：Sharpening Mechanism

> 来自 [[wiki/papers/he-2026-urlvr-scale|He et al. 2026 (ICLR 2026)]]

### 核心论点

所有 intrinsic reward methods 本质上都在做同一件事——**sharpening the model's initial distribution**（锐化模型初始分布）。

### Sharpening Theorem (Theorem 1) — 数学证明

以 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的 majority voting reward 为例。从 KL-regularized RL objective 出发：

$$\max_{\pi_\theta} \mathbb{E}_{y \sim \pi_\theta}[r(x,y)] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$$

最优策略闭式解中，majority answer 概率被放大 $e^{1/\beta}$ 倍，形成 **"rich-get-richer"** 动态。

**定理**: 在两个假设下（Majority Stability + Effective Learning），$p_{maj}^{(k)}$ 以速率 $\rho = e^{-1/\beta}$ **几何收敛到 1**，策略收敛到以初始 majority answer 为唯一输出的确定性策略。

**Unified Reward Framework**: 进一步证明所有 intrinsic rewards 都可统一理解为 manipulating cross-entropy between carefully chosen distributions → 推广 sharpening 分析到 certainty-based、ensemble-based 所有方法。

### Sharpening 的成功条件

$$\text{Sharpening 有效} \iff P_{init}(\text{correct answer}) > P_{init}(\text{any incorrect answer})$$

- **Confidence-Correctness Aligned**: 模型对正确答案最有信心 → sharpening 提升 accuracy
- **Misaligned**: 模型对错误答案更有信心 → sharpening **灾难性失败**

### Rise-then-Fall Pattern

所有 intrinsic 方法的训练曲线呈现相同模式：
1. **Rise phase**: 在 aligned 样本上快速提升
2. **Fall phase**: 在 misaligned 样本上过度 sharpen → accuracy 急剧下降

这解释了 PRISM 观察到的"100 步后崩溃"现象。

**穷举调参实验**: 5 种 intrinsic reward × 4 种超参（temperature/mini-batch/KL/rollout number）网格搜索——某些设置延迟 collapse，但 **nearly all settings eventually degrade**（~1000步/4 epochs 内）。Rise-then-fall 是 **fundamental limitation，不是 engineering problem**。

### 三种 Failure Modes

| Failure Mode | 方法 | 表现 | 严重程度 |
|:------------|:-----|:-----|:---------|
| **Gradual degradation** | Self-Certainty, Majority Voting | 最慢衰退，Label Accuracy 保持较高 | 最温和 |
| **Length collapse** | Probability | 奖励短序列（概率乘积偏好短输出） | 中等 |
| **Repetition collapse** | Token/Trajectory-Level Entropy | 重复 high-prob tokens 最小化 entropy | 最严重 |

Self-Certainty 和 Majority Voting 最稳定（答案级 sharpening 而非 token 级）。

### Per-Problem 分析：Amplification ≠ Correction

25 个 MATH-500 问题逐个训练（100 epochs each）：22/25 问题只是**加强了初始偏好**（无论对错），仅 3/25 (12%) 实现了 wrong → correct 翻转。训练本质是 **amplification 而非 correction**。

**OOD 泛化惊喜**: 在 6 个初始错误的问题上训练，虽然 train label accuracy ≈ 0，但在 OOD 问题上仍可能提升——sharpening 可以跨问题泛化。

### Small Dataset Safety & Test-Time Training

| 数据集大小 | 表现 |
|-----------|------|
| 32-128 samples | **无 collapse**（localized overfitting，不 systematic shift） |
| 512+ | 开始出现 rise-then-fall |
| 16k+ (DAPO-17k) | 必然 collapse |

**启示**: Intrinsic URLVR 最安全的应用场景是 **test-time training on small domain-specific datasets**（如 [[wiki/papers/zuo-2025-ttrl|TTRL]] 在 40 道 AMC23 上 TTT）。

### Model Collapse Step (MCS) — 实用指标

- **定义**: 训练过程中 Reward Accuracy 降至 <1% 的步数
- **优势**: 5.6x faster than full RL（1.19B vs 6.66B tokens），**不需要 GT labels**
- **准确性**: MCS 与 GT Gain 的相关性 > Pass@k 与 GT Gain 的相关性
- **用途**: 选择 base model 时，跑短暂 intrinsic URLVR → 看 MCS → MCS 越大越适合 RL

### External Reward：突破 Intrinsic 天花板

Self-Verification 实验（Countdown 任务）：
- Reward Accuracy 先降（模型试图 exploit verifier）→ 然后恢复并稳定 > 0.5
- 远优于 intrinsic 方法（trajectory-level entropy 持续 collapse）
- **关键原因**: External verifier（编译器/Lean/数值检查）不随模型提升而退化

两大 External Reward 路径：
1. **Unlabeled data**: RPT/TPT/DuPO/SEAL — next-token prediction 做 reward
2. **Generation-Verification Asymmetry**: LADDER（不定积分）/ RLSR（Countdown）/ Absolute Zero（代码）/ AlphaProof（定理证明）

### 对各论文的理论解释

| 论文 | Sharpening 视角 |
|------|----------------|
| TTRL | Majority voting = ensemble-based intrinsic → 小数据 TTT 安全，大数据最终 collapse |
| EMPO | Semantic entropy minimization = sharpening；entropy thresholding 是过滤 misaligned 样本的启发式 |
| CoVo | Consistency/Volatility 提供更强的过程信息，但 distance 仍由 likelihood 定义，因此仍可能属于 certainty-based / sharpening 范式 |
| PRISM | Rise-then-fall = sharpening 的必然结果；PRM 提供非 sharpening 的 external 信号补救 |
| SPARK | Trained PRM = external reward，不受 sharpening 限制，所以最稳定 |
| Self-Judge | Frozen Judge = external 校准，SC 频率 = intrinsic sharpening；混合部分缓解 |
| CTRL-RAG | CLR 利用文档对比 = computational asymmetry，可能属于 external |
| SPAE | Confidence = certainty-based（会 sharpen）；Correctness 需要 GT（external） |
| Grad2Reward | Self-judging = 半 external（frozen copy）；gradient attribution 不同于纯 intrinsic |

---

## 性能对比

### 数学推理 Benchmark（Qwen2.5-Math-7B 或相近模型）

| 方法 | MATH500 | GSM8K | Minerva | 平均 | vs GT RLVR |
|------|---------|-------|---------|------|-----------|
| Base | ~37% | ~60% | ~11% | ~25% | — |
| GT RLVR | ~73% | ~88% | ~29% | ~44% | baseline |
| **EMPO** | 70.4% | 88.7% | 35.5% | ~48% | **~97%** |
| **CoVo** | 在 7 个 benchmark 上稳定优于 TTRL/EMPO | — | — | 接近/超过 supervised RL | 强过程感知内在奖励 |
| **SPARK (PRM-CoT)** | 74.0% | 87.1% | 32.4% | ~47% | **>100%** |
| **PRISM** | 80.8% | 92.1% | 38.6% | ~53%* | **>100%** |

*PRISM 数据来自 Qwen2.5-7B on DAPO-17k，不完全可比。

### Best-of-K Reranking（MCNIG）

| 方法 | 8 Benchmarks 平均 |
|------|-----------------|
| Majority Voting | 52.2% |
| QwenPRM 7B | 56.8% |
| **MCNIG 8B** | **62.3%** |
| **MCNIG 14B** | **63.4%** |

### RAG Benchmarks (Multi-hop QA)

| 方法 | PopQA | HotpotQA | MuSiQue | 平均 |
|------|-------|----------|---------|------|
| Search-R1 | 56.7 | 54.3 | 47.0 | ~51% |
| **ProRAG** | **60.3** | **57.6** | **55.3** | **~56%** |
| **CTRL-RAG** (think) | — | 84.3* | 82.1* | **~85%*** |

*CTRL-RAG 使用 Qwen3-8B think mode 且训练数据可能不同，与 ProRAG 不完全可比。

### 关键发现
- 最好的 URLVR 方法已经**达到甚至超过 ground-truth RLVR** 的效果（数学推理）
- CoVo 说明即使不训练 PRM，只要 reward 里显式建模"过程是否支持答案"，也能显著超过纯 outcome-level intrinsic 方法
- RAG 任务上，process supervision（ProRAG）和 faithfulness reward（CTRL-RAG）都大幅提升基线
- MCNIG 的 PRM 在 best-of-K 上超过所有基线，且计算成本最低

### TTRL 改进方法对比（Qwen2.5-Math-1.5B）

| 方法 | AIME24 相对提升 | Avg | 核心改进 |
|------|:--------------:|:---:|----------|
| TTRL | baseline | 41.5 | — |
| DARE | +25.3% | 44.2 | Distribution-aware reward |
| T³RL | +31.6% | 48.8 | Tool verification |

### 优化框架改进（GRPO 系列）

| 方法 | 代表结果 | 核心发现 |
|------|----------|----------|
| PIPO | GSPO AIME25 +7.4%, DAPO 4B Avg 51.9 | GRPO η(p) gradient explosion |
| DCPO | ECE -71.6%, AUROC 0.914 | Accuracy-calibration gradient conflict |
| CLIPO | 跨 4 种 RL 算法一致提升 +1-2 avg | Contrastive learning 抑制 spurious reasoning |

### Label-Free / 开放域

| 方法 | 数学 Avg 变化 | 开放域变化 | 核心 |
|------|:---:|:---:|------|
| SARL | +7.65 (超 GT RL +7.15) | WildBench +9.10 | Small-world topology |
| EMPO | +~5.0 | -0.71 (退化) | Semantic entropy |

### 多模态自进化

| 方法 | 核心发现 | Base Model |
|------|----------|------------|
| Meta-TTRL | Self-introspection 7B > External 235B | Janus-Pro-7B |
| V-Zero | Unsupervised 51.9 > Supervised GRPO 50.8 | Qwen2.5-VL-7B |

---

## 核心发现与共识

### 1. 纯内部信号长期不可靠
**共识度**: ⭐⭐⭐⭐⭐（所有论文都涉及）

PRISM 系统性证明，SPARK 间接验证（self-consistency collapse），EMPO 通过 entropy thresholding 部分缓解。He et al. (ICLR 2026) 从理论上**数学证明**（Theorem 1: 几何收敛到确定性策略）+ **穷举实验验证**（5 种方法 × 4 超参网格搜索，~1000步内必 collapse）。这是 fundamental limitation，不是 engineering problem。

CoVo 是这个背景下一个重要进展：它表明纯内部信号并非只能做答案级 majority 或 entropy，也可以显式利用中间状态与最终答案的一致性/波动性；但由于其底层仍依赖 likelihood，是否能从根本上摆脱 sharpening，仍需更长期验证。

### 2. Stationary Reward 优于 Non-stationary
**共识度**: ⭐⭐⭐⭐⭐

冻结的外部模型（SPARK 的 PRM、Self-Judge 的 frozen Judge、ProRAG 的 PRM）比在线计算的信号（直接 self-consistency、直接 entropy）更可靠。

### 3. Step-level 信号通常优于 Outcome-level
**共识度**: ⭐⭐⭐⭐

SPARK 的 PRM-CoT 超过 GT RLVR（outcome-level），PRISM 的 PRM 比 self-certainty 更可靠，ProRAG 的 dual-granularity 超过纯 outcome。

### 4. 混合多种信号更稳健
**共识度**: ⭐⭐⭐⭐

PRISM 的 PRM+SC，Self-Judge 的 SC+Judge，ProRAG 的 outcome+process，CTRL-RAG 的 CLR×Acc——所有混合方案都超过单一信号。

### 5. Reward Hacking 是核心挑战
**共识度**: ⭐⭐⭐⭐⭐

所有论文都发现或讨论了 [[reward-hacking|reward hacking]]，包括 verbose collapse（CTRL-RAG）、step inflation/reduction（SPARK）、confidence inflation（PRISM）。

### 6. GRPO 是当前默认框架
**共识度**: ⭐⭐⭐⭐⭐

7 篇（除 MCNIG 仅做 best-of-K）都用 [[grpo|GRPO]]，说明它在 LLM RL 中的主导地位。

> **但 GRPO 存在已知缺陷**：PIPO 严格证明 group-relative normalization 引入 η(p) ∝ 1/[p(1-p)]，在 p→0 或 p→1 时梯度爆炸→mode collapse。DCPO 进一步发现 accuracy 和 calibration 目标存在 fundamental gradient conflict。建议考虑 DAPO 或 PIPO 的 PIRL 替代框架。

### 7. PRM 训练数据可以自动生成
**共识度**: ⭐⭐⭐⭐（SPARK, MCNIG, ProRAG 共同验证）

三种不同的自动标注方案（self-consistency、信息论、MCTS+contrastive）都证明合成 PRM 数据质量足够高，甚至超过 ground-truth 标注。

### 8. Contrastive Learning 为 RLVR 提供跨轨迹结构信号
**共识度**: ⭐⭐⭐（CLIPO 单独验证，但跨 4 种算法一致有效）

CLIPO 证明正确 rollout 之间的 contrastive learning 能抑制 spurious reasoning（答案对但推理错），与 outcome reward 正交互补。

### 9. 推理结构 reward 可超越 GT RL
**共识度**: ⭐⭐⭐（SARL 单篇验证，但结果令人信服）

SARL 证明完全不看答案对错、只看推理链拓扑结构（small-world network）的 reward，在数学任务上竟超越使用 GT 的标准 GRPO。这为 process-level reward 的价值提供了最强证据。

### 10. 自我评估可优于外部强评估（Metacognitive Synergy）
**共识度**: ⭐⭐⭐（Meta-TTRL + V-Zero 两篇验证）

Meta-TTRL 发现 7B 自我内省 > 235B 外部评估。V-Zero 发现无监督 co-evolution > 有监督 GRPO。共同指向：capacity-matched signals > absolute evaluator strength。

### 11. 标签噪声鲁棒性：RLVR 的固有优势与传统方法的失效
**共识度**: ⭐⭐⭐⭐（OLR + Imperfect Verifier 两篇从不同角度验证）

OLR 和 Imperfect Verifier 共同证明 RLVR 对验证噪声具有固有鲁棒性（15% 噪声率下性能仅降 2pp），但传统噪声处理方法（如 small-loss selection）在 RLVR 中灾难性失败（-17.4%）。原因在于 RL 的 loss landscape 本质不同于 SFT——RL 中 loss 大小不指示样本噪声程度。OLR 进一步提出 Early Correctness Coherence 作为替代噪声检测信号，Imperfect Verifier 提出 "moderate accuracy + high precision" 的实用验证器设计原则。

### 12. 层次化信用分配的 token 效率
**共识度**: ⭐⭐⭐（SHAPE 单篇验证，但效果显著）

SHAPE 证明 segment-level solvability potential + token-level entropy redistribution 的层次化信用分配方案可以同时提升准确率（+3%）和减少 token 消耗（-30%）。这打破了"更多 token = 更好推理"的直觉，说明精确的 credit assignment 比粗粒度的 outcome reward 更高效。

---

## 研究谱系与信号演化

```
Self-Consistency (Wang et al., 2022)
    │
    ├── 直接做 Reward ──→ Collapse (~150步, SPARK发现)
    │
    ├── 语义聚类版本 ──→ EMPO (Semantic Entropy Minimization)
    │                      └── 短期有效，本质是 sharpening (He et al.)
    │
    ├── 过程一致性版本 ──→ CoVo (Consistency + Volatility)
    │                      └── 从最终答案一致推进到中间状态支持度
    │
    ├── 生成PRM训练数据 ──→ SPARK (Step-level PRM, 超越GT)
    │                      └── 最稳定的方案
    │
    ├── + Self-Certainty ──→ PRISM (混合信号, 互补)
    │                      └── 系统性失败分析的价值
    │
    ├── + Judge Modulation ──→ Self-Judge (多模态, Distributional)
    │                          └── 分布建模的理论贡献
    │
    └── 统一理论 ──→ He et al. (ICLR 2026): Sharpening + MCS

Information Theory
    │
    └── Information Gain ──→ MCNIG (O(N) PRM 训练数据, 跨任务)
                              └── 信息论替代重采样方法

RAG + RL
    │
    ├── MCTS + PRM Pipeline ──→ ProRAG (四阶段, dual-granularity)
    │                            └── 最完整的 RAG RL 框架
    │
    └── Contrastive Likelihood ──→ CTRL-RAG (轻量 faithfulness reward)
                                    └── 无需额外模型的 RAG reward

TTRL Reward 改进方向
    │
    ├── 统计分布改进 ──→ DARE (Distribution-Aware, ICML)
    │                     └── Theorem 2.1 Information Collapse
    │
    ├── 多视角共识 ──→ DCRL (Dual Consensus)
    │                   └── Anchor-Explorer + Harmonic Mean
    │
    ├── 外部工具锚定 ──→ T³RL (Code Execution Verification)
    │                     └── N=16 > TTRL N=64
    │
    ├── Confidence 分布建模 ──→ DistriTTRL (GMM + Shift Correction)
    │                            └── Diversity penalty 防 collapse
    │
    ├── 数据增广 ──→ TTVS (Variational Synthesis)
    │                 └── Hybrid IGE/CGE exploration, 无标签 1.5B > 7B supervised
    │
    └── Bayesian 方差降低 ──→ DBB (Beta-Bernoulli Posterior)
                               └── OOD +12.49%, zero extra compute

优化框架改进
    │
    ├── 梯度爆炸修复 ──→ PIPO (η(p) → ∞ 发现 + PIRL)
    │
    ├── 梯度冲突解耦 ──→ DCPO (Accuracy vs Calibration)
    │
    ├── 跨轨迹正则化 ──→ CLIPO (InfoNCE Contrastive)
    │
    └── 熵分解调制 ──→ AsymGRPO (Informative vs Spurious Entropy)

标签噪声鲁棒性
    │
    ├── 噪声类型分析 + 在线矫正 ──→ OLR (Inactive/Active Noise + Online Refinement)
    │                                  └── Small-loss selection 灾难性失败 (-17.4%)
    │
    └── 理论鲁棒性证明 ──→ Imperfect Verifier (15% noise ≈ 2pp drop)
                            └── "Moderate accuracy + high precision" 原则

Step-Level Credit Assignment
    │
    ├── 层次化分配 ──→ SHAPE (Segment Solvability + Token Entropy)
    │                    └── +3% acc, -30% tokens
    │
    └── 语义过程信号 ──→ SPAE / SPC (Confidence + Semantic Consistency)

Policy-Reward Co-Evolution
    │
    ├── 推理-奖励共进化 ──→ Self-Guide (Internal Reward + Inference Guidance)
    │                         └── +8%, step-level co-evolution
    │
    └── 相关: CoVerRL (Coverage-based Verifier RL)

Label-Free Process Reward
    │
    ├── 拓扑结构 ──→ SARL (Small-World Network)
    │
    └── 语义一致性 ──→ SPC (Semantic Process Consistency)

多模态自进化
    │
    ├── T2I 元认知 ──→ Meta-TTRL (Rubric Decomposition)
    │
    ├── VLM Co-Evolution ──→ V-Zero (Questioner-Solver)
    │
    └── 多模态 Retracing ──→ CSRS (Retracing Re-inference + Softened Reward + VSP)
                               └── 几何推理无监督自进化
```

---

## 扩展分类维度

### 按 Reward 模型实现方式（完整版）

| 实现方式 | 论文 | 描述 |
|----------|------|------|
| 语义聚类频率 | EMPO | 多次采样 → 语义聚类 → 频率做 reward |
| Likelihood consistency + volatility | CoVo | 中间状态到候选答案的 distance matrix → consistency/volatility 聚合 |
| Trained PRM (self-consistency) | SPARK | 合成数据训练的 generative PRM |
| Trained PRM (信息论) | MCNIG | [[information-gain\|MCNIG]] 自动标注训练的 PRM |
| Trained PRM (MCTS) | ProRAG | [[mcts\|MCTS]] 探索 + GPT-4o 标注训练的 PRM |
| 现成 PRM | PRISM | 直接用 GenPRM-7B |
| Self-consistency + Modulation | Self-Judge | SC 频率 × bounded Judge 评分 |
| [[contrastive-likelihood\|Contrastive Likelihood]] | CTRL-RAG | 有/无文档的 log-likelihood 对比 |

### 按 Length Bias 处理方式

| 论文 | Length Bias 处理 |
|------|----------------|
| EMPO | Token-level loss normalization (1/\|o_i\|) |
| SPARK | PRM-CoT 本身对长度不敏感 |
| PRISM | GRPO 标准归一化 |
| Self-Judge | Group-wise distributional baseline |
| MCNIG | N/A (best-of-K) |
| ProRAG | Z-score normalization |
| CTRL-RAG | **√T 归一化**（关键发现：/T 太强，无归一化导致 verbose collapse） |

---

## 开放问题与研究方向

### 1. 长期训练稳定性
即使最好的方法（SPARK、PRISM），在非常长期的训练中是否仍然稳定？He et al. 的 **MCS (Model Collapse Step)** 提供了实用预测工具（5.6x faster，无需 GT），但其精确度尚未在大规模训练中验证。小数据集（32-128 samples）可避免 collapse → test-time training 是当前最安全的应用场景。

### 2. 跨任务泛化
EMPO 和 SPARK 主要在数学上验证，Self-Judge 在多模态上验证，ProRAG/CTRL-RAG 在 RAG 上验证。MCNIG 展示了跨任务的 PRM 训练（数学+代码+SQL+医学），但未集成到 RL。是否能统一到一个框架？

### 3. PRM 训练数据方法的统一
三种自动生成方案（SPARK self-consistency、MCNIG 信息论、ProRAG MCTS）各有优劣。能否组合或选择最优？MCNIG 效率最高但需要 ground-truth 分组，SPARK 完全无监督但 O(M×N)，ProRAG 质量最高但成本最大。

### 4. RAG + Process Supervision 的深化
ProRAG 和 CTRL-RAG 开辟了 RAG RL 的新方向。能否将 ProRAG 的 step-level PRM 和 CTRL-RAG 的 faithfulness reward 结合？

### 5. Sharpening 的根本突破
He et al. 证明 intrinsic rewards 受限于 confidence-correctness alignment（Theorem 1）。如何突破？两条可行路径：(1) **Computational asymmetries / Generation-Verification Asymmetry**——验证比生成容易的任务可以构建 external reward（LADDER/RLSR/Absolute Zero/AlphaProof）；(2) **信息论方法**（MCNIG）利用 log-prob 变化构建信号。He et al. 的 self-verification 实验已初步验证 external reward 在 Countdown 任务上的有效性。

CoVo 和当前的 [[wiki/synthesis/step-level-se-proposal|SPC 提案]] 代表另一条值得探索的路线：不直接跳到 external reward，而是先把 internal reward 从“最终答案是否一致”升级为“过程是否真的支持该答案”。其中 CoVo 仍停留在 likelihood 空间，SPC 则进一步走向 semantic rollout 空间。

### 6. MCNIG 集成到 RL Pipeline
MCNIG 目前仅做 best-of-K reranking。如果用 MCNIG 训练的 PRM 做 RL reward（类似 SPARK 的 Stage 3），效果会如何？

### 7. Reward Hacking 的根本解决
当前方案都是缓解而非根本解决。是否存在理论上不可 hack 的 reward 构建方式？

### 8. GRPO 的系统性替代
PIPO 和 DCPO 分别揭露了 GRPO 的梯度爆炸和梯度冲突。DAPO 已部分缓解（Dynamic Sampling 过滤全对/全错 groups），但是否需要更根本性的替代框架？PIPO 的 PIRL（累积改进优化）是一个方向。

### 9. 推理结构 reward 的上限
SARL 证明拓扑 reward 在数学上可超越 GT RL，但此结论的普适性尚需验证。是否存在"最优推理拓扑"？不同任务的最优拓扑是否不同？

---

## 面试综合题

- Q: 介绍一下 URLVR 领域的主要方法路线和它们的优缺点。🔴
- A: 三条路线：(1) **纯内部信号**（EMPO 的语义熵）——简单自主但长期不可靠，本质是 sharpening；(2) **外部 PRM**（SPARK、MCNIG）——最稳定但计算成本高；(3) **混合信号**（PRISM 的 PRM+SC、Self-Judge 的 SC+Judge、ProRAG 的 PRM+F1、CTRL-RAG 的 CLR+Acc）——兼顾稳定性和效率。He et al. (ICLR 2026) 统一理论：所有 intrinsic 方法都在做 sharpening，成功取决于 confidence-correctness alignment。

- Q: 对比三种 PRM 训练数据自动生成方法。🔴
- A: (1) SPARK：从每步重新采样 N 次看答案一致性，完全无监督但 O(M×N)；(2) MCNIG：用信息论（log-prob 变化）标注，O(N) 最高效，但需要知道正确/错误答案分组；(3) ProRAG MCTS：MCTS 探索 + GPT-4o contrastive labeling，96% 一致性最高质量，但成本最大。各有适用场景。

- Q: 如果让你设计一个新的 URLVR 方法，你会怎么做？🔴
- A: 基于当前最佳实践：(1) 用 MCNIG 高效生成 PRM 训练数据（O(N) 复杂度）；(2) 像 PRISM 一样混合 PRM 和 self-certainty 信号做 advantage；(3) 像 Self-Judge 一样用 energy-based distributional modeling 替代简单的 z-score baseline；(4) 加 EMPO 的 entropy thresholding 过滤极端样本；(5) 借鉴 He et al. 的 MCS 来决定何时停止训练或切换到 external rewards；(6) 如果是 RAG 任务，加入 CTRL-RAG 的 CLR 做 faithfulness 信号。

- Q: RAG 任务上的 RL 有哪些方法？各自优缺点？🔴
- A: 两种代表方案：(1) **ProRAG**——四阶段 pipeline（SFT → MCTS PRM → RFT → Process-Supervised RL），dual-granularity advantage，效果最好但最重；(2) **CTRL-RAG**——contrastive likelihood reward（有/无文档的 log-likelihood 对比），轻量无需额外模型，重点关注 faithfulness。ProRAG 提供 step-level 精细信号，CTRL-RAG 提供 outcome-level 的 faithfulness 信号。资源充足选 ProRAG，轻量部署选 CTRL-RAG，两者可以结合。

- Q: 对比三种改进 TTRL reward 的方法（DCRL / DARE / T³RL）。🔴
- A: 三种方法在不同维度改进 TTRL 的 majority voting：(1) **DCRL** 改进投票机制——通过 unlearning 构造 explorer 引入第二视角，用 harmonic mean 选取 pseudo-label，解决 spurious majority；(2) **DARE** 改进 reward 计算——用 uncertainty-normalized distribution 替代 hard voting，保留分布信息，AIME24 +25.3%；(3) **T³RL** 引入外部锚定——用 code execution 验证 rollout，verified 答案获 5x 权重，属于 external reward，AIME24 +31.6%。三者正交可叠加。

- Q: GRPO 有哪些已知的数学缺陷？如何解决？🔴
- A: 两大缺陷：(1) **PIPO 发现梯度爆炸**——group-relative normalization 引入 η(p) ∝ 1/[p(1-p)]，当 p→0(全错) 或 p→1(全对) 时梯度发散→mode collapse。解决方案：PIRL 框架（优化累积改进而非绝对 reward）或 DAPO（Dynamic Sampling 过滤极端 groups）。(2) **DCPO 发现梯度冲突**——accuracy 和 calibration 在 Fisher 信息度量下梯度内积<0，同时优化必然矛盾。解决方案：masked gradient 将 reasoning tokens 和 confidence tokens 的优化解耦。

- Q: 有哪些证据表明模型自评估可以优于外部强评估？🟡
- A: 两篇论文提供了证据：(1) **Meta-TTRL** 发现 7B 模型自我内省产生的 T2I reward 信号比 235B 外部模型（GPT-4o/Gemini）更有效——这是 "Metacognitive Synergy" 效应，capacity-matched signals 比 absolute evaluator strength 更重要；(2) **V-Zero** 发现无监督 Questioner-Solver co-evolution 的 VLM 推理性能 (51.9) 超越有监督 GRPO (50.8)。启示：自评估信号天然匹配模型当前能力水平。

- Q: RLVR 对验证器噪声有多鲁棒？如何设计实用的不完美验证器？🟡
- A: OLR 和 Imperfect Verifier 共同证明：(1) 15% 噪声率下性能仅降 2pp；(2) 传统噪声鲁棒方法（small-loss selection）在 RLVR 中灾难性失败（-17.4%），因为 RL 的 loss landscape 不同于 SFT；(3) "moderate accuracy + high precision" 原则——verifier 宁可漏判也不要误判，因为 false positive（错误通过）的危害远大于 false negative。OLR 进一步区分了 inactive noise（错误答案被标为正确）和 active noise（正确答案被标为错误），提出 Early Correctness Coherence 作为噪声检测信号，并设计 Online Label Refinement 实现 progressive self-correction，在 0.1-0.9 噪声率范围内保持鲁棒。
