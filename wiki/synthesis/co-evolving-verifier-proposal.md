---
title: "Co-Evolving Verifier: 让 PRM 跟着 RL 训练一起进化"
type: synthesis
tags: [co-evolving, PRM, URLVR, SPC, bootstrapping, self-improving, reward-model, research-proposal]
created: 2026-04-08
updated: 2026-04-08
sources: [wiki/synthesis/step-level-se-proposal.md, wiki/papers/rahman-2025-spark.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/he-2026-urlvr-scale.md, wiki/papers/wu-2026-spae.md, wiki/papers/zhang-2025-covo.md]
status: draft
---

# Co-Evolving Verifier: 让 PRM 跟着 RL 训练一起进化

## 与 SPC 方案的关系

本文档是 [[wiki/synthesis/step-level-se-proposal|SPC 方案]] 的**第二分支方向**。

SPC 解决的是 "怎么从推理过程中提取无监督 step-level signal"。但 SPC 本身有一个 fundamental limitation：它是 intrinsic signal，long-run 会随 policy sharpening 而退化。

Co-Evolving Verifier 解决的是另一个问题："怎么让 reward model 不被 policy 甩在后面"。

两者的关系类似于 [[wiki/papers/wu-2026-spae|SPAE]] 中 sampling-based probing 和 SEP 探针的关系——前者是 accurate-but-expensive 的信号源，后者是 cheap-but-needs-calibration 的加速器。在我们的语境中：

- **SPC** = accurate-but-expensive 的 step-level signal（需要 M=5 次短续写 + answer extraction）
- **Co-Evolving Verifier** = cheap-but-needs-calibration 的在线 PRM（用 SPC 标签训练，日常 RL 中替代 probing）

## 研究动机

### 冻结 PRM 路线的两个根本缺陷

现有 PRM 路线（[[wiki/papers/rahman-2025-spark|SPARK]]、[[wiki/papers/royer-2026-mcnig|MCNIG]]、[[wiki/papers/ghimire-2026-prism|PRISM]]）都有同一个模式：先训练 PRM，然后冻结，再做 RL。这有两个问题：

1. **额外训练成本**：需要一个完整的 PRM 训练 Stage。SPARK 需要 8 个解法 × 16 次验证 × 8000 题 = ~1M 次 inference 来生成训练数据。
2. **能力天花板被锁死**：冻结 PRM 的判断力不会超过其训练时的水平。当 policy 在 RL 中变强后，PRM 见到了它训练数据中从未出现过的更高质量（或更诡异的错误）推理 pattern，reward 信号会越来越 noisy。

SPARK (Rahman et al.) 自己也发现了这个问题——直接用 self-consistency 做在线 reward 会在 ~150 步后 collapse，因为在线信号是 non-stationary 的。但冻结 PRM 的 stationarity 代价是失去了自我进化的能力。

### 用户的核心问题

> 能不能让 PRM 跟着 RL 训练一起进化？

## 文献调研：谁在做 Co-Evolving？

### 已有工作

| 论文 | 核心 idea | 有无 GT anchor | 与 URLVR 的距离 |
|------|-----------|---------------|-----------------|
| **SPARK (Liu et al., 2025/09)** arXiv:2509.22624 | Policy 和 Reward Model 同时训练。回收 RLVR 的 rollout + correctness 信号，同时训练模型作为 generative reward model（pointwise + pairwise + reflection 目标混合）。正反馈循环。 | **有** — 依赖 verifiable reward（代码执行/答案验证） | 近，但不是纯 URLVR |
| **rePIRL (Wu et al., 2026/02)** arXiv:2602.07832 | 用 Inverse RL 框架交替更新 policy 和 PRM。理论上统一了 online 和 offline PRM learning。 | **需要 expert demonstrations** | 中等 |
| **Sci-CoE (He et al., 2026/02)** arXiv:2602.12164 | 两阶段 co-evolving：Stage 1 用少量标注数据建 verifier anchor；Stage 2 用 geometric consensus（一致性 + 可靠性 + 多样性）驱动自迭代。Solver 和 Verifier 一起进化。 | **Sparse supervision**（少量标注） | 较近 |
| **Self-Judge (Wu et al., 2026)** 已读 | Actor 和 Judge 分离但都来自同一模型。Judge frozen。 | 否 | 近，但 Judge 不进化 |
| **SPARK (Rahman et al., 2025)** 已读 | 自训练 PRM 但冻结使用。 | PRM 训练时用 self-consistency | 近，但 PRM 不进化 |

### 关键发现

1. **SPARK (Liu et al.)** 是最接近"co-evolving"的工作，验证了正反馈循环的有效性。但它依赖 verifiable reward 作为 anchor。
2. **Sci-CoE** 最接近 URLVR 设定，用 sparse supervision → unsupervised 过渡。其 geometric consensus 机制值得借鉴。
3. **rePIRL** 在数学上最优雅（IRL 框架），但需要 expert demonstrations。
4. **纯 URLVR 下的 co-evolving PRM 目前没有人做过。** 这是一个空白。

## 核心挑战：Mutual Sharpening

### 理论风险

如果 PRM 和 Policy 都从同一个模型出发、都没有 external anchor，它们会陷入 **mutual reinforcement of errors**：

1. Policy 生成一条看似合理但实际错误的推理链
2. PRM（如果也在进化中）可能把这条链评为"好的"
3. 这个正反馈被放大

这本质上是 [[wiki/papers/he-2026-urlvr-scale|He et al. 2026]] 的 **Sharpening Theorem 的双重版本**：不仅 policy 在 sharpen，PRM 也在 sharpen，两者互相加速。

### 为什么 SPARK (Liu et al.) 能避免这个问题

因为它有 **verifiable reward 作为硬锚点**。即使 policy 和 reward model 互相迎合，代码执行 / 数学答案验证 / Lean proof checker 会把真实 correctness 信号注入进来，阻止 mutual drift。

在 URLVR 设定下，我们没有这个硬锚点。所以需要替代方案。

## 方案：SPC-Anchored Co-Evolving Verifier

### 核心 Insight

SPC 虽然也是 intrinsic signal，但它比直接的 self-consistency 或 majority voting 更不容易被 sharpen，因为它检查的是 **semantic rollout behavior**（模型实际会生成什么答案），而不是 **likelihood judgment**（模型觉得哪个答案概率高）。

因此，SPC 可以作为 co-evolving verifier 的 **soft anchor** —— 不如 GT 那么硬，但比纯 self-consistency 更稳。

### 三层自举架构

```
Layer 1: TTRL outcome anchor
    │  majority voting → response-level pseudo reward
    ▼
Layer 2: SPC step-level signal (expensive, ~accurate)
    │  M=5 probing → SPC_k → Φ_SPC → step-level labels
    │  周期性运行（每 N_recal 步一次）
    ▼
Layer 3: Lightweight Co-Evolving Verifier (cheap, needs calibration)
    │  用 SPC 生成的 step-level labels 训练
    │  日常 RL 训练中替代 SPC probing
    │  每 N_recal 步被 SPC 重新校准
    ▼
Policy Improvement
    │  更好的 rollouts
    ▼
SPC recalibrates → Verifier re-trains → ...
```

### 具体设计

#### Phase A: SPC-only 冷启动（= SPC 方案首发论文）

前 $N_{warm}$ 步使用纯 SPC probing 做 step-level credit assignment。此阶段：

- 积累 SPC 标签数据：`{(hidden_state_k, SPC_k, Conf_k, Φ_SPC_k)}`
- 验证 SPC 信号有效性
- 建立 baseline 性能

#### Phase B: 蒸馏 Verifier

用 Phase A 积累的数据训练轻量 Verifier。两种实现选择：

| 方案 | 架构 | 输入 | 输出 | 训练成本 | 推理成本 |
|------|------|------|------|----------|----------|
| **Linear Probe** | 单层线性 | $h_k$（hidden state） | $\hat{SPC}_k$ 或 $\hat{\Phi}_k$ | 极低 | 近零 |
| **Lightweight MLP** | 2-3 层 MLP | $h_k$ + position encoding | $\hat{SPC}_k$ + $\hat{Conf}_k$ | 低 | 极低 |

这一步本质上就是 SEP 探针的 SPC 版本。

#### Phase C: 在线 Co-Evolution

关键设计——**周期性校准循环**：

```
for each RL training epoch:
    for step t in [1, ..., N_total]:
        if t % N_recal == 0:
            # 校准阶段：用昂贵的 SPC probing
            run full SPC probing on current batch
            collect (h_k, SPC_k) pairs
            fine-tune Verifier on new data
            log calibration metrics
        else:
            # 日常阶段：用 Verifier 快速推理
            Φ_pred = Verifier(h_k)
            use Φ_pred for step-level advantage estimation
```

**$N_{recal}$ 的选择**：

- 太频繁（如每步都校准）→ 退化为纯 SPC，没有加速效果
- 太稀疏（如每 500 步校准一次）→ Verifier 可能严重过时
- 建议范围：**每 20-50 步校准一次**，每次用 ~100 个样本更新 Verifier

#### Phase D: 自适应校准触发

更进一步：不用固定频率校准，而是当 Verifier 和 SPC 的判断出现显著分歧时才触发：

$$
\text{Trigger} = \mathbb{I}\left[\frac{1}{|B|}\sum_{k \in B} |\hat{\Phi}_{Verifier}(k) - \Phi_{SPC}(k)| > \epsilon_{drift}\right]
$$

如果 Verifier 的预测和 SPC ground-truth 的平均偏差超过阈值 $\epsilon_{drift}$，说明 policy 已经 drift 到 Verifier 训练分布之外，需要重新校准。

## 与 SPC 方案的分工

| 维度 | SPC 方案（首发论文） | Co-Evolving Verifier（延伸方向） |
|------|---------------------|-------------------------------|
| **核心贡献** | 提出 SPC signal 本身 | 让 SPC 的加速版本能跟着训练进化 |
| **对应 SPAE 中的角色** | Probing mechanism | SEP 探针 |
| **必要性** | 独立成文，不依赖 co-evolving | 需要 SPC 作为基础 |
| **工程复杂度** | 中等 | 较高（在线训练 Verifier） |
| **理论风险** | intrinsic sharpening | mutual sharpening（但有 SPC 校准缓解） |
| **时间线** | 先做 | 后做，或作为 SPC 论文的 Phase 4 扩展 |

## 为什么不直接做 Co-Evolving 跳过 SPC？

1. **SPC 是 co-evolving 的数据源**：没有 SPC，Verifier 没有 step-level labels 来训练。
2. **SPC 是 co-evolving 的校准锚**：周期性校准需要一个 ground-truth signal，SPC 就是这个 anchor。
3. **SPC 独立有价值**：即使不做 co-evolving，SPC 本身就比 TTRL / CoVo-style shaping 提供更好的 step-level credit assignment。
4. **论文节奏**：SPC 是一篇完整论文；co-evolving 可以是第二篇，或者 SPC 论文的 Phase 4 扩展。

## 风险评估

### 风险 1: Verifier 过拟合到早期 policy 的 pattern

**描述**：如果 Verifier 在冷启动阶段学到的 SPC pattern 不能泛化到后期更强的 policy，校准频率需要很高，削弱加速效果。

**缓解**：
1. 使用 replay buffer，保留历史校准数据，防止 catastrophic forgetting
2. 正则化 Verifier（dropout / weight decay）
3. 监控 Verifier 预测 vs SPC ground-truth 的相关性，自适应调整校准频率

### 风险 2: SPC 本身 sharpen 后，校准信号也变差

**描述**：SPC 依赖 policy 的 rollout。当 policy 严重 sharpen 后，所有 rollout 都收敛到同一答案，SPC_k → 1 everywhere，失去区分度。此时用 SPC 校准 Verifier 也无意义。

**缓解**：
1. 监控 SPC 的方差——如果 $\text{Var}(SPC_k)$ 在所有步骤上都趋近 0，说明 SPC 已失效
2. 此时应该**停止 co-evolving**，转入 early stopping 或切换到 external reward
3. 这也是 He et al. 理论的实际应用：用 SPC 方差作为 process-level MCS 的早期指标

### 风险 3: Verifier 引入额外的 reward hacking 表面

**描述**：Policy 可能学到 exploit Verifier 的特定弱点，而不是真正改善推理。

**缓解**：
1. TTRL outcome anchor 作为最终 gate——即使 Verifier 给高分，outcome reward 不过关也没用
2. 周期性 SPC 校准会暴露 Verifier 的 systematic bias
3. 参考 [[wiki/papers/ghimire-2026-prism|PRISM]] 的经验：min aggregation 比 mean aggregation 更抗 hack

## 实验设计（如果独立成文）

### Exp 1: SPC vs Verifier 的信号质量对比

- 在离线数据上，比较 SPC ground-truth 和 Verifier 预测的：
  - step-level AUROC
  - trajectory-level Pearson correlation with correctness
- 分析 Verifier 在不同校准频率下的退化速度

### Exp 2: Co-Evolving vs Frozen Verifier

| 实验组                    | Verifier 训练方式      | 校准   |
| ---------------------- | ------------------ | ---- |
| Frozen SPC             | 全程用昂贵的 SPC probing | —    |
| Frozen Verifier        | 冷启动阶段训练，然后冻结       | 无    |
| Co-Evolving (fixed)    | 冷启动 + 每 N 步校准      | 固定频率 |
| Co-Evolving (adaptive) | 冷启动 + drift 触发校准   | 自适应  |

评估：
1. 最终 accuracy（AIME/AMC/MATH500）
2. 训练曲线稳定性（是否延缓 rise-then-fall）
3. 总计算成本（wall-clock time）
4. Verifier-SPC 一致性随训练步数的变化

### Exp 3: 长期稳定性

- 跑 >1000 步训练
- 监控 SPC 方差是否趋零
- 监控 co-evolving 是否比 frozen Verifier 更晚进入 collapse
- 与 [[wiki/papers/he-2026-urlvr-scale|He et al.]] 的 MCS 概念对接：用 Verifier accuracy 定义 process-level MCS

## 潜在标题

- *"Self-Calibrating Process Reward: Co-Evolving Step-Level Verifiers for Unsupervised RLVR"*
- *"From Probing to Prediction: SPC-Anchored Co-Evolution of Policy and Process Verifier"*
- *"Bootstrapping Step-Level Supervision: Co-Evolving Semantic Process Verifiers Without Ground Truth"*

## 潜在故事线

1. SPC 提出了一个有效的无监督 step-level signal，但 probing 成本高。
2. 自然的加速方案是蒸馏为探针（如 SEP），但冻结探针会被 policy drift 甩在后面。
3. 我们提出 SPC-anchored co-evolving verifier：用周期性 SPC 校准来驱动轻量 Verifier 的在线进化。
4. 这形成一个三层自举：TTRL (outcome) → SPC (step-level, expensive) → Verifier (step-level, cheap)。
5. 实验证明 co-evolving 比 frozen verifier 更稳定，比全量 SPC 更高效。

## Positioning

- vs [[wiki/papers/rahman-2025-spark|SPARK (Rahman)]]：SPARK 冻结 PRM；我们让 PRM 跟着训练一起进化，且不需要 GT 训练数据。
- vs **SPARK (Liu et al., 2509.22624)**：SPARK 的 co-evolving 依赖 verifiable reward；我们用 SPC 作为 soft anchor，实现纯 URLVR 下的 co-evolution。
- vs **rePIRL (Wu et al., 2602.07832)**：rePIRL 用 IRL 框架需要 expert demonstrations；我们完全无监督。
- vs **Sci-CoE (He et al., 2602.12164)**：Sci-CoE 需要 sparse supervision 建 verifier anchor；我们用 SPC 作为 anchor，不需要任何标注。
- vs [[wiki/papers/ghimire-2026-prism|PRISM]]：PRISM 混合冻结 PRM + self-certainty；我们让 PRM 动态进化，且用语义 rollout consistency 替代 self-certainty。
