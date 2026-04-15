---
title: "Co-Evolving Verifier: 让 PRM 跟着 RL 训练一起进化"
type: synthesis
tags: [co-evolving, PRM, URLVR, SPC, bootstrapping, self-improving, reward-model, research-proposal, V-Zero, Meta-TTRL, DCRL, DARE, metacognitive-synergy, capacity-matched, CoVerRL, Self-Guide, imperfect-verifier, balanced-training, beta-posterior, noise-tolerance, SCRL, PowerFlow, DistriTTRL, OLR, DBB, dual-network, outcome-driven, judge-training, LoRA-fork]
created: 2026-04-08
updated: 2026-04-13
sources: [wiki/synthesis/step-level-se-proposal.md, wiki/papers/rahman-2025-spark.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/he-2026-urlvr-scale.md, wiki/papers/wu-2026-spae.md, wiki/papers/zhang-2025-covo.md, wiki/papers/wang-2026-v-zero.md, wiki/papers/tan-2026-meta-ttrl.md, wiki/papers/du-2026-dual-consensus.md, wiki/papers/du-2026-dare.md, wiki/papers/cui-2026-clipo.md, wiki/papers/pan-2026-coverrl.md, wiki/papers/wang-2026-self-guide.md, wiki/papers/plesner-2026-imperfect-verifier.md, wiki/papers/yan-2026-scrl.md, wiki/papers/chen-2026-powerflow.md, wiki/papers/yang-2026-distribttrl.md, wiki/papers/yang-2026-olr.md, wiki/papers/kim-2026-dbb.md]
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

| 论文                                                                                             | 核心 idea                                                                                                                                               | 有无 GT anchor                            | 与 URLVR 的距离                                          |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ---------------------------------------------------- |
| **SPARK (Liu et al., 2025/09)** arXiv:2509.22624                                               | Policy 和 Reward Model 同时训练。回收 RLVR 的 rollout + correctness 信号，同时训练模型作为 generative reward model（pointwise + pairwise + reflection 目标混合）。正反馈循环。         | **有** — 依赖 verifiable reward（代码执行/答案验证） | 近，但不是纯 URLVR                                         |
| **rePIRL (Wu et al., 2026/02)** arXiv:2602.07832                                               | 用 Inverse RL 框架交替更新 policy 和 PRM。理论上统一了 online 和 offline PRM learning。                                                                                | **需要 expert demonstrations**            | 中等                                                   |
| **Sci-CoE (He et al., 2026/02)** arXiv:2602.12164                                              | 两阶段 co-evolving：Stage 1 用少量标注数据建 verifier anchor；Stage 2 用 geometric consensus（一致性 + 可靠性 + 多样性）驱动自迭代。Solver 和 Verifier 一起进化。                          | **Sparse supervision**（少量标注）            | 较近                                                   |
| **Self-Judge (Wu et al., 2026)** 已读                                                            | Actor 和 Judge 分离但都来自同一模型。Judge frozen。                                                                                                                | 否                                       | 近，但 Judge 不进化                                        |
| **SPARK (Rahman et al., 2025)** 已读                                                             | 自训练 PRM 但冻结使用。                                                                                                                                        | PRM 训练时用 self-consistency               | 近，但 PRM 不进化                                          |
| **V-Zero (Wang et al., 2026/01)** arXiv:2601.10094                                             | Questioner-Solver co-evolution。Questioner 生成 MCQ + intuitive answer，Solver 用 CoT 推理 + MV 获得 pseudo-label，Dual-Track Reward 对比 intuition vs reasoning。 | **否** — 完全无标注                           | **非常近** — 纯 URLVR co-evolution                       |
| **Meta-TTRL (Tan et al., 2026/03)** arXiv:2603.15724                                           | TTRL 扩展到 T2I。元认知架构（生成器+内省器），rubric-based 二值评估 → 几何平均聚合 reward。                                                                                        | **否** — 自我评估                            | 中等 — T2I 领域，但 metacognitive synergy 发现高度相关           |
| **[[wiki/papers/pan-2026-coverrl\|CoVerRL (Pan et al., 2026)]]**                               | 单一模型交替充当 generator/verifier，MV 提供对比训练信号实现 co-evolution。Balanced training（$\|V^+\| = \|V^-\|$）是关键。Verification accuracy 55%→85%，+4.7-5.9% over TTRL。   | 否（MV-based）                             | **非常近** — 直接验证了 co-evolving 可行性，但 outcome-level      |
| **[[wiki/papers/wang-2026-self-guide\|Self-Guide (Wang et al., 2026)]]**                       | 同一模型生成 internal reward，用于 inference-time guidance + training-time step-level reward。Policy-reward co-evolution loop。                                  | 否（internal reward）                      | **近** — policy-reward loop 与我们的 SPC-Verifier loop 同构 |
| **[[wiki/papers/plesner-2026-imperfect-verifier\|Imperfect Verifier (Plesner et al., 2026)]]** | 证明 15% 噪声率内 RLVR 仍鲁棒。"Moderate accuracy + high precision" 原则。                                                                                         | N/A（理论分析）                               | **中等** — 为不完美 co-evolving verifier 提供容错理论            |

### 关键发现

1. **SPARK (Liu et al.)** 是最接近"co-evolving"的工作，验证了正反馈循环的有效性。但它依赖 verifiable reward 作为 anchor。
2. **Sci-CoE** 最接近 URLVR 设定，用 sparse supervision → unsupervised 过渡。其 geometric consensus 机制值得借鉴。
3. **rePIRL** 在数学上最优雅（IRL 框架），但需要 expert demonstrations。
4. **纯 URLVR 下的 co-evolving PRM 目前没有人做过。** 这是一个空白。
5. **V-Zero 是纯 URLVR 下 co-evolution 的最强证据**。它证明了两个关键点：(a) 完全无标注的 co-evolution 可以超越有监督 GRPO（51.9 vs 50.8）；(b) Dual-Track Reward 提供了一种非 GT 的 reward 信号替代方案。V-Zero 的 Questioner-Solver 架构与我们的 SPC-PRM co-evolving 方案高度类似——Questioner 类似 SPC（提供评估信号），Solver 类似 Policy。
6. **Meta-TTRL 的 Metacognitive Synergy 发现**：自我内省（7B）产生的 reward 信号优于外部强模型（235B GPT-4o/Gemini）。这为 SPC 的 probing-based self-evaluation 路线提供了理论支撑——capacity-matched signals 比 absolute evaluator strength 更有效。如果自评估信号天然匹配模型当前能力，那么 SPC-anchored Co-Evolving Verifier 作为自评估的加速版本，理论上也应优于外部冻结 PRM。
7. **[[wiki/papers/pan-2026-coverrl|CoVerRL]] 直接验证了 co-evolving 可行性且无需 GT**。单一模型交替充当 generator/verifier，MV 提供对比训练信号，verification accuracy 从 55%→85%。关键区别：CoVerRL 是 outcome-level，我们的方案是 step-level（通过 SPC），粒度更细。
8. **[[wiki/papers/plesner-2026-imperfect-verifier|Imperfect Verifier]] 为不完美 co-evolving 提供容错理论**。只要 verifier 的 precision 足够高（即使 recall 不完美），RLVR 训练仍然鲁棒。这直接减轻了"co-evolving verifier 不够准确"的担忧。
9. **[[wiki/papers/wang-2026-self-guide|Self-Guide]] 验证了 policy-reward co-evolution loop 的有效性**。在 agent 任务上 +8% over environment-reward-only baselines。其架构（同一模型生成 internal reward）与我们的 SPC-anchored verifier 高度同构。

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

### 新论文对 Mutual Sharpening 风险的缓解

多篇新论文为缓解 mutual sharpening 提供了新工具：

1. **DCRL/DARE 升级 Layer 1 Anchor**：如果将 Layer 1 的 naive majority voting 升级为 [[wiki/papers/du-2026-dual-consensus|DCRL]] 的 dual consensus 或 [[wiki/papers/du-2026-dare|DARE]] 的 distribution-aware estimation，Layer 1 的 outcome anchor 本身更可靠。更可靠的 anchor 意味着 SPC 校准信号也更可靠（因为 SPC 的 a_final 来源于 outcome anchor），从而减少 mutual sharpening 的起点误差。

2. **CLIPO Contrastive Regularization**：[[wiki/papers/cui-2026-clipo|CLIPO]] 的 InfoNCE 对比学习可以作为 co-evolving 过程中的额外正则化——强制正确推理在表示空间中聚拢，防止 policy 和 verifier 在 spurious reasoning 上达成"虚假共识"。

3. **V-Zero 的 Ambiguity Reward**：V-Zero 在 consistency case 给 ambiguity reward（$\min(c, 1-c)$），主动抑制过度 sharpening。Co-Evolving Verifier 可以借鉴：当 SPC 和 Verifier 的判断高度一致时（可能已经 sharpened），降低 reward 信号强度；当两者分歧时（真正需要校准的区间），增大信号。

4. **[[wiki/papers/pan-2026-coverrl|CoVerRL]] 的 Balanced Training**：CoVerRL 发现 balanced training（$|V^+| = |V^-|$）是防止 co-evolution 退化的关键。在我们的架构中，这意味着 SPC 校准数据应保持正负样本平衡——不能只用 high-SPC 步骤训练 verifier，也要包含 low-SPC 步骤。

5. **[[wiki/papers/kim-2026-dbb|DBB]] 的 Beta Posterior 平滑**：DBB 的 Beta-Bernoulli posterior 可以作为 co-evolving verifier 的在线校准信号——利用历史 rollout 统计信息平滑当前 batch 的 verifier 判断，防止单 batch 噪声导致 mutual drift。零额外计算开销。

6. **[[wiki/papers/plesner-2026-imperfect-verifier|Imperfect Verifier]] 的容错边界**：15% 噪声率内仍鲁棒。这意味着 co-evolving verifier 不需要完美——只要维持 >85% precision，就足以提供有效训练信号。

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

### 风险 4: V-Zero 式 co-evolution 可能需要 Questioner 角色

**描述**：V-Zero 的成功部分归因于 Questioner 动态调整题目难度，形成 curriculum learning 效果。我们的方案中 SPC 只做评估，没有"出题"角色——可能缺少 curriculum 信号。

**缓解**：
1. SPC 本身有隐含的 curriculum 效果——随着 policy 变强，SPC 在简单题上持续给高分（减少信号），在难题上仍有区分度
2. 可以引入 DCRL 的 Dynamic Sampling 思路——优先采样 SPC 和 Verifier 分歧大的"困难"样本进行校准
3. 如果资源允许，可以考虑增加 Questioner 模块作为第四层

### 风险 5: CoVerRL 已证明 outcome-level co-evolving 有效，step-level 是否有额外价值？

**描述**：[[wiki/papers/pan-2026-coverrl|CoVerRL]] 用相对简单的 outcome-level generator-verifier 交替训练就获得了 +4.7-5.9% 提升。如果 outcome-level 已足够，step-level co-evolving 的边际价值是否值得额外的工程和计算成本？

**缓解**：
1. CoVerRL 自身也发现了 "consensus trap" 问题——当 MV 错误时 verifier 无法纠正，只能减少错误放大。Step-level SPC 信号可以在 MV 错误时通过 process inconsistency 提供更早的预警
2. [[wiki/papers/rahman-2025-spark|SHAPE]] 已证明 step-level credit assignment 可以在提升 accuracy 的同时减少 30% token 消耗——效率提升是 step-level 的独有优势
3. 可以设计 CoVerRL + SPC 的组合实验，验证 step-level 的边际价值

## 变体方案：Dual-Network Outcome-Driven Co-Evolution

### 动机

上述三层自举架构（Layer 1-3）中，Layer 3 的 Verifier 校准方式是 **imitation-driven**：直接用 SPC 标签做回归，让 Verifier 对齐 SPC 的打分标准。这存在一个根本限制——**Verifier 的天花板被锁死在 SPC**。如果存在某种比 SPC 更有效的 step-level 评分策略（比如 SPC 无法区分的步骤之间，存在其他特征可以区分），imitation-driven 的 Verifier 永远发现不了。

一个自然的问题：**能不能让 Verifier 自由探索评分策略，只要最终 Model 变好就行？**

这引出了 **outcome-driven judge training** 的思路：不要求 Judge 和 SPC 打一样的分，只要求 Judge 的打分能引导 Model 朝 SPC/MV 认可的方向进步。

### 核心设计

#### 架构：同源分叉的 Dual-Network

```
初始化：
  Base_LLM (预训练模型，如 Qwen2.5-7B)
  Model = Base_LLM + LoRA_policy    ← 负责推理生成
  Judge = Base_LLM + LoRA_judge     ← 负责 step-level 打分
  共享 base weights，仅 LoRA adapters 独立更新
```

关键设计决策：
- **同源初始化**：Judge 从同一个预训练模型出发，天然具备与 Model 相当的语言理解能力
- **LoRA 分叉**：额外显存开销约 ~5%（而非 2x），工程可行
- **独立参数更新**：Model 的 policy gradient 不干扰 Judge 的评分能力，反之亦然。区别于 SPARK (Liu et al.) 的单网络多目标训练中 policy gradient 和 reward modeling gradient 可能冲突

#### 训练流程

```
日常 RL 训练（每步）：
  1. Model 对 batch 问题生成 responses
  2. Judge 对每个 step 打分 → {score_k}
  3. 用 step scores 计算 advantage → RL 更新 Model（仅 LoRA_policy）
  
Judge 校准（每 N_eval 步）：
  4. 记录 Model_v1 在校准 batch 上的表现：
     - MV 正确率 / SPC 分数作为 baseline_v1
  5. 用最近 N_eval 步 Judge 指导下训练的 Model_v2 重新生成：
     - MV 正确率 / SPC 分数作为 baseline_v2
  6. Judge reward = Δperformance = baseline_v2 - baseline_v1
  7. 用 Δperformance 更新 Judge（仅 LoRA_judge）

异常兜底：
  如果连续 M 次 Judge 更新后 Δperformance ≤ 0：
     → 回退到 SPC 直接标签校准（imitation-driven fallback）
     → 防止 Judge 持续给出无效甚至有害的评分
```

#### 与 Imitation-Driven 校准的本质区别

| 维度 | Imitation-Driven（已有设计） | Outcome-Driven（本变体） |
|------|---------------------------|------------------------|
| Judge 优化目标 | $\min \|Judge(step_k) - SPC(step_k)\|$ | $\max P(\text{Model improves} \| \text{Judge's feedback})$ |
| 类比 | Behavior Cloning | Reinforcement Learning |
| Judge 天花板 | = SPC | 理论上可 > SPC |
| 探索空间 | 无（纯监督） | 有（RL 搜索） |
| 训练稳定性 | 高（回归问题） | 较低（稀疏 RL signal） |
| 收敛速度 | 快（dense step-level labels） | 慢（trajectory-level reward） |
| "不准但有用"的 reward | 不可能（被强制对齐 SPC） | 可能（只要 Model 变好就行） |

#### 理论动机

在 reward shaping 文献中，存在两种评价 reward function 质量的标准：
- **Accuracy**：shaped reward 与 true reward 的接近程度
- **Utility**：shaped reward 导致的 policy improvement 幅度

一个"不准但有用"的 reward function 可能比"准但没用"的更好。例如，Judge 可能发现在某些步骤给极端信号能加速学习，虽然这个评分不反映 SPC 意义上的 semantic consistency，但能引导 policy 更快进步。Imitation-driven 的 Verifier 永远不会探索到这种策略。

### 风险与挑战

#### 挑战 1：信号稀疏

Judge 对 K 个步骤各输出一个分数（动作空间 $\mathbb{R}^K$），但训练信号是每 $N_{eval}$ 步一个 scalar 的 $\Delta\text{performance}$。这是一个高维动作空间 + 稀疏 reward 的 RL 问题，收敛会比 imitation-driven 慢得多。

**缓解**：
1. SPC 冷启动：先用 SPC 标签做 warm-up 训练 Judge（Phase A），给 Judge 一个好的初始评分策略，再切换到 outcome-driven 微调
2. 限制 Judge 的更新幅度（小 LoRA rank、低学习率），避免一次更新偏离太远

#### 挑战 2：进步信号的度量

"Model 是否变好"的度量选择影响信号质量：
- **MV 正确率**：简单题饱和后无信号，但噪声低
- **SPC 分数**：后期 sharpening 导致区分度下降
- **建议**：两者加权组合，SPC 有效期用 SPC，退化后切换到 MV

#### 挑战 3：Judge 更新频率 $N_{eval}$

- $N_{eval}$ 太小 → 单步 RL 改变微小，$\Delta\text{performance}$ 被采样噪声淹没
- $N_{eval}$ 太大 → Judge 长期冻结，退化成 frozen judge
- **建议范围**：$N_{eval} \in [50, 200]$，每次用 ~500 个样本评估

#### 挑战 4：Bi-level Optimization 稳定性

Model 用 Judge 的分数训练，Judge 用 Model 的进步训练——这是双层优化。类似 GAN 的训练动态，存在振荡风险。

**缓解**：
1. Judge 更新频率远低于 Model（$N_{eval} \gg 1$），类似 Target Network 的稳定化效果
2. 异常兜底机制：连续 M 次无进步 → 回退到 SPC 直接校准
3. 可借鉴 [[wiki/papers/wang-2026-self-guide|Self-Guide]] 的 Stage-wise Trust Schedule：warm-up → activation → full → annealing

### 与现有方案的关系

本变体是三层自举架构（Section "方案"）的 **Layer 3 替代实现**，Layer 1（TTRL outcome anchor）和 Layer 2（SPC step-level signal）不变。区别仅在于 Layer 3 的 Verifier 如何被训练：

```
三层架构（不变）：
  Layer 1: TTRL outcome anchor → response-level pseudo reward
  Layer 2: SPC step-level signal → expensive but accurate

Layer 3 变体 A（已有设计）：Imitation-Driven
  → Verifier 直接对齐 SPC 标签
  → 稳定、快速、天花板 = SPC

Layer 3 变体 B（本节）：Outcome-Driven
  → Verifier 通过 RL 自由探索，以 Model 进步为 reward
  → SPC 作为 fallback 兜底
  → 可能超越 SPC 天花板，但训练更难
```

**建议实验策略**：先实现变体 A 作为 baseline，再实现变体 B 做对照。如果 B > A → outcome-driven 确实发现了更好的评分策略 → 强 finding。如果 B ≈ A → 说明 SPC 已经是足够好的评分标准。如果 B < A → 信号稀疏的代价大于探索的收益。

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

### Exp 3: Imitation-Driven vs Outcome-Driven Judge Training（核心对照）

本实验直接检验 Dual-Network Outcome-Driven 变体的核心假说：**自由探索的 Judge 能否超越 SPC 蒸馏的 Judge？**

| 实验组 | Judge 架构 | Judge 训练方式 | 校准信号 |
|--------|-----------|--------------|---------|
| A: Frozen SPC | 无 Judge | 全程 SPC probing | — |
| B: Imitation-Driven | LoRA_judge | SPC 标签回归 | $\min \|J(s_k) - SPC_k\|$ |
| C: Outcome-Driven | LoRA_judge | Model 进步做 RL | $\Delta\text{performance}$ |
| D: Outcome + Fallback | LoRA_judge | Outcome-driven + SPC 兜底 | 混合 |
| E: SPARK-style 多目标 | 单 LoRA（共享） | Policy loss + judging loss 联合 | Verifiable reward |

评估指标：
1. **Judge 天花板**：Judge step-level score 与 oracle SPC 的 AUROC —— C 是否 > B？
2. **Policy 最终性能**：AIME/AMC/MATH500 accuracy
3. **训练效率**：达到同等 accuracy 所需的 wall-clock time
4. **Judge 发现新策略的证据**：分析 C 组 Judge 的打分分布 vs SPC 分布，是否存在系统性偏差且该偏差是有益的
5. **稳定性**：训练曲线是否有 collapse / 振荡

关键假说检验：
- 如果 C > B > A → **outcome-driven 探索发现了比 SPC 更好的评分策略** → 核心贡献
- 如果 C ≈ B > A → co-evolving 有效，但 SPC 已是足够好的标准 → 回退到 imitation-driven
- 如果 B > C → 信号稀疏的代价 > 探索收益 → outcome-driven 不可行
- 如果 D > C → fallback 机制有效，纯 outcome-driven 不稳定需要兜底

### Exp 4: 长期稳定性

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
- vs **V-Zero (Wang et al., 2601.10094)**：V-Zero 的 Questioner-Solver co-evolution 与我们的 SPC-Verifier co-evolution 架构相似，但 V-Zero 用 Dual-Track Reward（intuition vs reasoning），我们用 SPC（semantic rollout consistency）做校准信号。V-Zero 在 VLM 上验证 co-evolution 超越有监督，为我们的方案提供了跨模态的信心。
- vs **Meta-TTRL (Tan et al., 2603.15724)**：Meta-TTRL 的 Metacognitive Synergy（自评估 > 外部强评估）直接支持 SPC-anchored 自评估路线。我们的 Co-Evolving Verifier 本质上是 capacity-matched 的自评估加速器。
- vs **[[wiki/papers/pan-2026-coverrl|CoVerRL (Pan et al., 2026)]]**：CoVerRL 最接近我们的 co-evolving 理念，但它在 outcome-level 做 generator-verifier 交替训练；我们的方案在 step-level 做 SPC-anchored verifier 训练，提供更细粒度的 process-level 信号。CoVerRL 的 balanced training 发现值得直接借鉴。
- vs **[[wiki/papers/wang-2026-self-guide|Self-Guide (Wang et al., 2026)]]**：Self-Guide 的 policy-reward co-evolution 架构与我们最接近。关键差异：(1) Self-Guide 用于 agent 任务，我们用于数学推理 URLVR；(2) Self-Guide 的 internal reward 是 trajectory-level，我们的是 step-level；(3) Self-Guide 没有周期性校准机制，我们用 SPC 做周期性重校准。
- vs **[[wiki/papers/plesner-2026-imperfect-verifier|Imperfect Verifier (Plesner et al., 2026)]]**：Imperfect Verifier 为我们的方案提供了最直接的理论支撑——co-evolving verifier 不需要完美，只需 precision 足够高。15% 容错边界为 Phase C 的在线校准频率选择提供了量化依据。
- vs **Grad2Reward (Zhang et al., 2026)**：Grad2Reward 的 self-judging 证明了同等规模模型的判别能力足以提供有效 reward（1.5B self-judge ≈ 30B 外部 judge）。我们的 Dual-Network 变体进一步放开：不要求 Judge 对齐任何标准（SPC 或 GT），只要求 Judge 的评分能驱动 Model 进步。Grad2Reward 的 frozen self-judge 是我们 outcome-driven co-evolving Judge 的 lower bound。
