---
title: "Semantic Process Consistency: 无监督步骤级纠错研究方案"
type: synthesis
tags: [semantic-process-consistency, SPC, step-level, URLVR, SPAE, TTRL, CoVo, process-reward, research-proposal, reward-hacking, DCRL, DARE, T3RL, DCPO, PIPO, CLIPO, SARL, Meta-TTRL, V-Zero, contrastive-learning, gradient-conflict, CoVerRL, SCRL, PowerFlow, DistriTTRL, TTVS, OLR, AsymGRPO, DBB, CSRS, SHAPE, imperfect-verifier, Self-Guide, PRAISE, SLATE, STEP, prefix-rollout, hidden-states, variance-reduction, SPC-Stat, rollout-tree, consensus-entity, math-entity-extraction, statistical-process-divergence]
created: 2026-04-08
updated: 2026-04-15
sources: [wiki/papers/zuo-2025-ttrl.md, wiki/papers/wu-2026-spae.md, wiki/papers/zhang-2025-covo.md, wiki/papers/zhang-2025-empo.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/he-2026-urlvr-scale.md, wiki/concepts/process-reward-model.md, wiki/concepts/reward-hacking.md, wiki/papers/du-2026-dual-consensus.md, wiki/papers/du-2026-dare.md, wiki/papers/liao-2026-t3rl.md, wiki/papers/ma-2026-dcpo.md, wiki/papers/cui-2026-clipo.md, wiki/papers/wang-2026-pipo.md, wiki/papers/wang-2026-sarl.md, wiki/papers/tan-2026-meta-ttrl.md, wiki/papers/wang-2026-v-zero.md, wiki/papers/pan-2026-coverrl.md, wiki/papers/yan-2026-scrl.md, wiki/papers/chen-2026-powerflow.md, wiki/papers/yang-2026-distribttrl.md, wiki/papers/bai-2026-ttvs.md, wiki/papers/yang-2026-olr.md, wiki/papers/gu-2026-asymgrpo.md, wiki/papers/kim-2026-dbb.md, wiki/papers/yu-2026-csrs.md, wiki/papers/ai-2026-shape.md, wiki/papers/plesner-2026-imperfect-verifier.md, wiki/papers/wang-2026-self-guide.md, wiki/papers/zhang-2026-praise.md, wiki/papers/liang-2026-step.md, wiki/papers/chen-2026-slate.md, wiki/synthesis/spc-related-work-survey.md]
status: active
---

# Semantic Process Consistency: 无监督步骤级纠错研究方案

## 一句话总结

> 用 [[wiki/papers/wu-2026-spae|SPAE]] 的 step-level probing 在每个推理步骤后做短续写，检查这些续写导向的答案是否与该轨迹最终答案**语义一致**，据此构造 **Semantic Process Consistency (SPC)** 与 volatility 信号，再与 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的 outcome anchor 结合，做无监督的 step-level credit assignment。

## 研究动机

### 为什么旧的 SPAE-SE 方案不够

旧方案的核心是把 SPAE 中需要 GT 的 Correctness 替换成 Semantic Certainty。这个方向有价值，但存在三个问题：

1. **创新点偏弱**：本质还是「SPAE probing + semantic entropy」的组合，更像替换打分器，而不是提出新的 process signal。
2. **没有击中 TTRL 的核心失败模式**：如果一条轨迹从头到尾都稳定地指向同一个错误答案，纯 semantic certainty 仍可能给高分。
3. **仍偏向 certainty，而不是 correction**：它回答的是“模型是否越来越确定”，不是“推理过程是否真的稳定支持了当前答案”。

用户指出的关键问题是：**TTRL 最大的风险不是分歧，而是“一致地错”**。因此新的 step-level 信号必须能显式刻画“推理过程是否真的在支持最终答案”。

### 新方向的核心问题

给定一条最终答案为 $a_{final}$ 的推理轨迹，真正有价值的问题不是：

- 这一步之后模型是否更自信？
- 这一步之后模型是否更偏向某个答案 token？

而是：

- **从这一步继续推下去，模型实际会生成什么答案？这些答案是否语义上支持当前最终答案？**

这就是 **Semantic Process Consistency (SPC)** 的核心定义。

## 相关工作的启发

### TTRL 给了 outcome anchor，但没有 process correction

[[wiki/papers/zuo-2025-ttrl|TTRL]] 用 majority voting 提供无监督 pseudo-reward，解决了“没有 GT 怎么训练”的问题，但它只在**最终答案层**判断一条轨迹是否与 majority 一致，无法区分：

- 哪些步骤真正推动了解题
- 哪些步骤只是冗余验证
- 哪些轨迹虽然最终答案一致，但中间推理并不稳定

### SPAE 给了 probing 框架，但 Correctness 需要 GT

[[wiki/papers/wu-2026-spae|SPAE]] 已经给出一个极强的 step-level 框架：在每个步骤边界插入 probe prompt，观察模型从当前前缀继续生成时的行为。问题在于其 Correctness 依赖 GT force-feeding，因此不能直接用于 URLVR。

### CoVo 给了关键 insight，但仍停留在 likelihood 空间

[[wiki/papers/zhang-2025-covo|CoVo]] 的关键发现是：

1. 正确轨迹的中间状态更一致地指向自己的最终答案（high consistency）
2. 错误轨迹的偏离往往更晚暴露（high volatility）

但 CoVo 的 consistency 定义基于 **distance / log-likelihood**：

$$
Con(\tau) = \frac{1}{T} \sum_{i=0}^{T-1} \mathbb{I}(D[i,0] = \min_k D[i,k])
$$

这本质上是在问：

> “当前 state 在概率上最偏好自己的最终答案吗？”

而我们想进一步升级为：

> “当前 state 真正 rollout 出来的答案，是否语义上支持自己的最终答案？”

前者是**概率判断**，后者是**生成行为**。

## 核心 Idea

### Semantic Process Consistency (SPC)

对每个步骤前缀 $h_k = (q, \tau^{1:k})$，像 SPAE 一样做 $M$ 次短续写 probing：

$$
z_k^{(1)}, z_k^{(2)}, \dots, z_k^{(M)} \sim \pi_\theta(\cdot | h_k, p_{probe})
$$

从每条 probe continuation 中抽取其导向的答案 $a_k^{(m)}$，再与该轨迹最终答案 $a_{final}$ 做语义等价判断：

$$
e_k^{(m)} = \mathbb{I}[a_k^{(m)} \equiv_{sem} a_{final}]
$$

定义该步骤的 **SPC 分数**：

$$
SPC_k = \frac{1}{M} \sum_{m=1}^{M} e_k^{(m)}
$$

直觉上：

- $SPC_k \approx 1$：这一步之后，无论怎么短续写，模型都稳定走向与最终答案同义的结果
- $SPC_k \approx 0$：这一步之后，模型的后续答案与最终答案大多不一致，说明当前轨迹尚未稳定支持最终结论

### Volatility: 偏离暴露得有多晚

仅看平均一致性还不够。还需要看“不一致最后一次出现在哪里”。定义轨迹 volatility：

$$
Vol(\tau) = \frac{1}{T} \max \{k \mid SPC_k < \delta \}
$$

其中 $\delta$ 是一致性阈值（例如 0.5 或 0.6）。

- **低 volatility**：早期就稳定收敛到最终答案
- **高 volatility**：到了很后面还会出现与最终答案不一致的 rollout，说明过程不稳

这直接继承了 CoVo 的 insight，但把判断标准从 likelihood distance 换成了 semantic rollout consistency。

## SPC 与现有方法的本质区别

### vs CoVo：语义行为 vs 概率判断

| 维度 | CoVo | SPC |
|------|------|-----|
| 核心对象 | state 对 candidate answer 的 log-prob distance | state rollout 后实际得到的答案 |
| 判断方式 | “最偏好哪个答案” | “真正会生成什么答案” |
| 空间 | token likelihood | semantic equivalence |
| 对过度自信的鲁棒性 | 较弱，仍可能 sharpen 错误答案 | 更强，要求生成行为也稳定支持最终答案 |
| 复用 SPAE probing | 否 | 是 |

一句话概括：**CoVo 看模型怎么想，SPC 看模型怎么做。**

### vs TTRL：step-level correction 而不只是 outcome voting

TTRL 只知道最终答案是否与 group majority 一致。SPC 则进一步回答：

- 这条“正确/错误”轨迹的中间过程是否稳定？
- 哪些步骤开始真正锁定了答案？
- 哪些步骤虽然最终答案一致，但内部 rollout 仍高度分歧？

因此 SPC 的定位不是替代 TTRL，而是作为其 **process correction / credit assignment 模块**。

### vs SPAE：保留框架，去掉监督

SPAE 的 probing、saturation penalty、difference shaping 都可以保留；真正替换的是需要 GT 的 Correctness 信号。SPC 相当于给 SPAE 提供一个 URLVR 可用的 correctness proxy，但比旧的 semantic certainty 更直接，因为它测的是“过程是否支持答案”，而不是“过程是否更确定”。

## 方法设计

### 1. Step-level probing

复用 SPAE 的做法，在每个 step boundary 后插入 probe prompt，例如 `Final Answer: \\boxed{`，采样 $M=5$ 条短续写。

每条 probe 只需要生成到能抽取出一个明确答案即可，不必完整展开整条 CoT。

### 2. Answer extraction + semantic equivalence

对每个 probe continuation：

1. 抽取答案字符串 $a_k^{(m)}$
2. 与当前轨迹最终答案 $a_{final}$ 做语义等价判断

可选实现：

| 方法 | 优点 | 风险 | 建议 |
|------|------|------|------|
| exact / numeric match | 最便宜 | 只适合数学题 | 数学任务起步首选 |
| embedding cosine + threshold | 泛化强 | 阈值敏感 | 通用任务推荐 |
| NLI / entailment | 最语义化 | 推理成本高 | 离线分析使用 |

### 3. SPC-aware Step Potential

为了最大化复用 SPAE，可以把 $SPC_k$ 和 SPAE 的 Confidence 组合成新的 Step Potential：

$$
\Phi_{SPC}(\tau^k) = 1.5 \cdot SPC_k \cdot Conf_k + 0.5 \cdot SPC_k - Conf_k
$$

三种状态仍然成立：

| 状态 | SPC | Conf | $\Phi_{SPC}$ | 含义 |
|------|-----|------|-------------|------|
| 探索中 | 低 | 低 | 约 0 | 还没锁定方向 |
| 稳定收敛 | 高 | 高 | 接近 +1 | 过程和答案一致，且模型自信 |
| 自信但不稳 | 低 | 高 | 接近 -1 | 看似自信，但 rollout 不支持最终答案 |

这里最关键的是第三种状态。它正对应 TTRL 中最危险的 failure mode：**最终答案看起来稳定，但过程其实不支持它**。

### 4. Trajectory-level SPC summary

除了每步的 $SPC_k$，还可以定义轨迹级统计量：

$$
Con_{SPC}(\tau) = \frac{1}{T} \sum_{k=1}^{T} SPC_k
$$

$$
Vol_{SPC}(\tau) = \frac{1}{T} \max \{k \mid SPC_k < \delta\}
$$

用途：

1. 作为训练时的 diagnostics
2. 作为 CoVo 的直接对照基线
3. 作为 trajectory filtering / curriculum 的辅助信号

## 为什么 SPC 能缓解 TTRL 的“一致地错”

### TTRL 的核心问题

如果一个问题上，模型族群多数都输出同一个错误答案，那么 TTRL 会把这个错误 majority 当作 pseudo-label。此时：

- 纯 outcome reward 会把“跟错答案一致”的轨迹视为正样本
- 但这些轨迹内部未必真的稳定

### SPC 的纠错机制

设某条轨迹最终答案是错误的 $a_{wrong}$。如果它只是“碰巧”或“表面上一致”地到达该答案，那么在早中期步骤后做 probing 时，往往会出现：

- 有些 rollout 给出别的候选答案
- 有些 rollout 还停留在未决状态
- 不同步骤对 $a_{wrong}$ 的支持并不稳定

这会导致：

- $SPC_k$ 偏低
- $Vol_{SPC}$ 偏高
- $\Phi_{SPC}$ 无法持续饱和

因此即使这条轨迹在 TTRL 层拿到正的 outcome signal，它在 step-level 也拿不到强 shaping reward。相比之下，真正高质量的正确轨迹应该表现为：

- 早期开始出现高一致性
- 后续步骤持续支持最终答案
- volatility 更低

也就是说，**SPC 不能从根本上消除 wrong-majority，但它能显著减少“过程不稳却被当作正样本”的奖励泄露。**

## 相对 TTRL 的核心 Story

### TTRL 已经解决了什么

[[wiki/papers/zuo-2025-ttrl|TTRL]] 最大的贡献，是在完全无标注条件下构造出可用的 outcome-level reward。它依赖 majority voting 的 Lucky Hit 机制：即使 pseudo-label 不完全正确，只要错误预测足够分散，大多数错误轨迹仍会收到正确的负反馈，因此早期训练可以非常有效。

这解释了为什么 TTRL 在很多 benchmark 上提升巨大，也解释了为什么它是当前双层架构里不可替代的 **Layer 1 outcome anchor**。

### 但 TTRL 没有解决什么

TTRL 的 reward 是答案级的 binary reward，它隐含地把整条推理链视为一个不可分的整体。这会带来三个直接问题：

1. **没有关键步骤识别能力**：一条 100-step 轨迹中，第 23 步的真正突破和第 87 步的冗余检查共享同一个 reward。
2. **错误 credit assignment**：只要最终答案跟 pseudo-label 一致，整条链都会被正向更新，即使前面大部分步骤实际上是不稳定甚至误导性的。
3. **reward accuracy 退化时缺少保险丝**：当训练进入 sharpening 区间后，majority answer 会越来越集中，错误 majority 也会被不断放大。此时 TTRL 只能继续强化“最终一致”的轨迹，却无法区分其中哪些轨迹的过程本身已经出现崩坏征兆。

这正是 [[wiki/papers/he-2026-urlvr-scale|He et al. 2026]] 所说的核心问题：intrinsic URLVR 在大规模训练下，本质上更像 **amplification**，而不是 **correction**。

### 我们方案相对 TTRL 的本质增量

SPC 方案的关键，不是把 TTRL 推翻，而是在 TTRL 之上新增一个 **process-side correction layer**。

TTRL 回答的问题是：

> “这条轨迹的最终答案是否站在当前 group majority 这一边？”

SPC 回答的问题是：

> “这条轨迹在中间步骤上，是否真的逐步建立起了对该最终答案的稳定支持？”

因此相对 TTRL，SPC 额外提供了三种新能力：

1. **关键步骤定位**：哪些 step 真正让轨迹从不稳定转向稳定。
2. **伪正样本过滤**：最终答案虽然跟 pseudo-label 一致，但过程内部 rollout 仍然分裂的轨迹，应降权。
3. **回退检测**：轨迹如果从高一致性重新掉回低一致性，说明模型在 checking 阶段把本来解对的问题“想坏了”，应显式惩罚。

### 关键步骤分析为什么重要

TTRL 最大的问题，不是它完全错，而是它**太粗**。它只能说“这条链最终像不像好链”，却说不出“好链好在哪一步，坏链坏在哪一步”。

而在数学推理里，真正决定泛化能力的，往往不是最终答案 token，而是那些关键的中间状态转折：

- 某一步第一次把式子变形成正确方向
- 某一步第一次锁定了正确中间量
- 某一步开始后，后续 rollout 几乎都落到同一个答案

这些步骤才是真正应该被强化的地方。

如果没有 step-level 分析，RL 会把大量 credit 平均分配到整条轨迹上，结果就是：

- 关键步骤得不到足够强化
- 冗余 checking 也被一起奖励
- 错误但表面一致的长链会被不断放大

这也是为什么 outcome-only 方法更容易在后期进入“reward 还在升，但真实 reward accuracy 已经在掉”的状态。

### SPC 如何帮助减缓 reward accuracy 崩溃

这里要强调：SPC **不是**从理论上彻底打破 He et al. 的 sharpening 结论。它仍然属于 intrinsic 路线，所以不能宣称“完全避免 collapse”。

但 SPC 可以合理主张：它比纯 TTRL 更不容易过早进入 reward accuracy 崩溃，因为它减少了两类最危险的错误更新。

#### 1. 减少对伪正样本的放大

在 TTRL 中，只要轨迹最终答案等于 pseudo-label，就会整体得到正向更新。

但在 SPC 中，如果这条轨迹虽然最终和 pseudo-label 一致，却满足以下特征：

- 早期步骤 rollout 高度分裂
- 后期步骤仍频繁偏离最终答案
- consistency 很低、volatility 很高

那么它的 $\Phi_{SPC}$ 会偏低，最终能分到的正向 advantage 就会明显下降。

也就是说，SPC 做的不是“改写 pseudo-label”，而是**降低 pseudo-label 错误时的错误放大强度**。

#### 2. 减少对冗余 checking 的奖励

SPAE 已经指出，很多轨迹在已经解对后仍继续 checking，甚至发生 Right-to-Wrong。纯 TTRL 对这种后段 checking 没有辨别能力，因为最终 reward 只在答案层给一次。

SPC 继承 SPAE 的 saturation penalty 后，可以表达这样一个偏好：

- 在答案已经被稳定支持之后，继续冗余生成不应获得同等 credit
- 如果后续步骤让 consistency 下降，说明模型把本来稳定的解答重新带入不稳定区，应惩罚

这使 reward 不再平均撒到整条链上，而是更集中地压到“建立正确支持”的关键步骤上。

#### 3. 把 collapse 的早期征兆提前暴露出来

He et al. 用 `Reward Accuracy` 和 `MCS` 观测 collapse，但这些指标本质是 response-level 的。SPC 额外提供了 process-level 的早期预警信号：

- 正样本轨迹的平均 $Con_{SPC}$ 是否下降
- $Vol_{SPC}$ 是否越来越晚
- 高 reward 轨迹中，`SPC high -> low` 的回退是否增多

如果这些现象出现，往往意味着模型虽然还在追逐 pseudo-label，但过程内部已经在失稳。这类信号理论上应早于最终 reward accuracy 的全面崩溃。

所以 SPC 对 collapse 的贡献，不一定是“完全阻止”，更现实的表述是：

> 它通过更细粒度的 credit assignment，把错误更新限制在更少的步骤和更少的轨迹上，从而延缓 response-level reward accuracy 的系统性退化。

### 一句话总结这段 Story

> TTRL 解决了“无标注时怎么给结果打分”，但没有解决“结果 reward 应该分配给哪些步骤”。SPC 的核心增量，就是把 reward 从答案级 majority voting 推进到过程级 support analysis：只强化那些真正让轨迹稳定支持最终答案的关键步骤，抑制伪正样本和冗余 checking 的错误更新，从而为 intrinsic URLVR 提供比 TTRL 更细、更稳的 credit assignment。

## 双层无监督架构

### Layer 1: TTRL outcome anchor

对每个问题采样 group responses，做 majority voting，得到 response-level pseudo reward：

$$
R_i^{TTRL} = \mathbb{I}[a_i = a_{maj}]
$$

### Layer 2: SPC step-level shaping

对每条 response 的每个 step 计算：

- $SPC_k$
- $Conf_k$
- $\Phi_{SPC}(\tau^k)$
- $\Delta \Phi_{SPC}(\tau^k)$

然后沿用 SPAE 的两部分设计：

1. **Saturation Penalty**：在过程已经稳定后，抑制继续冗余 checking 的 credit
2. **Difference Shaping**：奖励关键跃迁，惩罚从稳定走向不稳定的回退

最终 token-level advantage 仍可写成：

$$
\hat{A}_{i,j} = \hat{A}^{Group}_i \cdot f(\Phi_{SPC}) + \xi \cdot g(\Delta \Phi_{SPC})
$$

这使整个系统只需 unlabeled questions 即可运行。

## 计算成本与实现可行性

### 为什么 SPC 比全量 process sampling 更现实

如果从每个步骤都完整 rollout 到最终答案，成本会爆炸。SPC 的可行性来自两个设计：

1. **复用 SPAE probing**：本来就要在 step boundary 采样短续写
2. **只抽取答案，不追求完整 CoT**：probe continuation 只要足够导出一个候选答案即可

### 成本对比

| 方案 | 每步额外成本 | 特点 |
|------|-------------|------|
| SPAE | 5 次短 probe + GT force-feed | 有监督上界 |
| CoVo | 每个 state 对多个 candidate answer 计算 likelihood | 需要 answer set |
| SPC | 5 次短 probe + semantic answer match | 无监督，直接复用 probing |
| 朴素 full rollout SPC | 多次完整 rollout | 成本过高，不推荐 |

在数学任务里，如果答案抽取和等价判断足够便宜，SPC 的额外成本应接近 SPAE，而显著低于“每步对全候选答案做全量比较”的方案。

## Probe Acceleration: 从 SEP 到 SPC Probe

旧方案里仍然有价值的一点，是把 sampling-based signal 蒸馏成 probe。

可以沿着 Semantic Entropy Probes (SEPs) 的思路，训练一个轻量预测器直接从 hidden state 预测：

- $SPC_k$
- 或者 $\mathbb{I}[SPC_k \geq \delta]$
- 或者 volatility contribution

三阶段路线：

1. **收集数据**：离线运行 SPC，得到 `(h_k, SPC_k, Vol label)`
2. **训练探针**：线性 probe / 小 MLP
3. **在线部署**：RL 训练时直接预测 SPC，减少 probing 成本

这部分不是首发版本必须做，但可以作为二阶段优化，保留成论文的效率扩展点。

## SPC-Stat: 零额外成本的统计实现

### 动机：SPC-Probe 的成本问题

原始 SPC 方案（上文）依赖于在每个步骤边界做 M=5 次短续写 probing。在实际训练环境中（A800 GPU, batch_size=32, rollout=64, 单轮训练 ~9 小时），假设每条轨迹平均 10 个步骤，则每个 training step 需要额外 32 × 64 × 10 × 5 = **102,400 次短续写**，这在工程上不可接受。

### 核心洞察：已有 rollout 本身就是信息金矿

TTRL 训练中，每个问题已经采样了 64 条完整轨迹，通过 majority voting 得到伪标签。这 64 条轨迹本身蕴含了丰富的 step-level 信息——如果正确组的轨迹在某个中间步骤都产出了相同的关键数学实体（数值、方程、代数表达式），而错误组在这一步产出了不同的实体，那么这一步就是**关键决策点**。

这一洞察来自 Rollout-Tree 方法族的启发（[[wiki/synthesis/spc-related-work-survey|文献调研报告]]），包括：
- [[2604.11037|RTMC]]：观察到 group rollouts 在关键步骤隐式形成树结构
- [[2505.10978|GiGPO]]：通过 anchor state grouping 实现跨轨迹的 step-level advantage
- [[2602.03719|BranPO]]：发现 shared prefix + divergent suffix 结构中蕴含 contrastive signal

但这些方法均面向 agentic/tool-use 场景（SWE-bench, ALFWorld, WebShop），其"状态匹配"依赖环境返回的离散状态（hash 签名、exact string match）。在数学推理中，64 条 CoT 在 token 级别几乎不可能共享前缀——即使两条轨迹在第 3 步都做了"两边同除以 2"，具体措辞也完全不同。

**SPC-Stat 的核心创新**：在自然语言数学推理中，用**数学实体的语义等价**替代 token 级别的状态匹配，实现零额外成本的 step-level credit assignment。

### 方法设计

#### Step 1: 数学实体抽取（Math Entity Extraction）

对每条轨迹 $\tau_i$ 的每个步骤 $s_k^i$，用正则表达式抽取数学实体集合：

$$E_k^i = \text{extract}(s_k^i)$$

实体类型包括：
- **数值**：整数、分数、小数（如 `24`, `3/2`, `0.75`），统一归一化为规范形式
- **boxed 答案**：`\boxed{...}` 内容
- **等式/方程**：`x = ...`, `y^2 + 2y = 5`
- **关键代数表达式**：经过简单标准化后的数学子串

实现上用一组正则表达式 + 简单归一化（如 `3/2 = 1.5` 统一表示）。

#### Step 2: 共识实体发现（Consensus Entity Discovery）

将 64 条轨迹按最终答案分组：

$$\mathcal{C} = \{\tau_i \mid a_i^{final} = a_{maj}\}, \quad \mathcal{E} = \{\tau_i \mid a_i^{final} \neq a_{maj}\}$$

统计正确组中所有实体的出现频率：

$$\text{freq}_{\mathcal{C}}(e) = \frac{|\{\tau_i \in \mathcal{C} \mid e \in \bigcup_k E_k^i\}|}{|\mathcal{C}|}$$

定义三类实体：

| 类型 | 定义 | 含义 |
|------|------|------|
| **正确共识实体** $\mathcal{G}^+$ | $\text{freq}_{\mathcal{C}}(e) \geq 0.5 \wedge \text{freq}_{\mathcal{E}}(e) < 0.2$ | 正确组多数都经过、错误组很少出现的关键中间结果 |
| **错误独有实体** $\mathcal{B}$ | $\text{freq}_{\mathcal{E}}(e) \geq 0.5 \wedge \text{freq}_{\mathcal{C}}(e) < 0.2$ | 错误组走入歧途的标志性中间结果 |
| **共享实体** | 两组都高频出现 | 问题本身的基本量，无区分能力 |

**优点：完全不需要步骤对齐**。不同轨迹步骤数不同、步骤边界不对齐，但实体发现只关心"这个实体在这条轨迹中出现了没有、在第几步出现"。

#### Step 3: Step-Level 打分

对轨迹 $\tau_i$ 的 step $k$：

**主信号：共识实体产出**

$$A_k^i = \begin{cases} +1 & \text{if } \exists e \in \mathcal{G}^+ : e \in E_k^i \wedge e \notin \bigcup_{j<k} E_j^i \quad \text{(首次产出正确共识实体)} \\ -1 & \text{if } \exists e \in \mathcal{B} : e \in E_k^i \wedge e \notin \bigcup_{j<k} E_j^i \quad \text{(首次产出错误独有实体)} \\ 0 & \text{otherwise} \end{cases}$$

**辅助信号：Token 4-gram Jaccard 相似度**

对无法抽取实体的步骤（如纯文字推理描述），使用归一化进度对齐后的 n-gram 统计：

$$B_k^i = \text{avg}_{j \in \mathcal{C}, j \neq i} J(s_k^i, s_k^j) - \text{avg}_{j \in \mathcal{E}} J(s_k^i, s_k^j)$$

其中 $J(\cdot,\cdot)$ 是 4-gram Jaccard 相似度。正值 = 与正确组表述更相似，负值 = 与错误组更相似。

**综合 Step Reward**：

$$r_{step}(k, i) = \alpha \cdot A_k^i + (1 - \alpha) \cdot B_k^i \cdot \mathbb{1}[A_k^i = 0]$$

建议 $\alpha = 0.8$，优先使用实体信号，实体信号缺失时回退到 n-gram 辅助。

#### Step 4: 与 TTRL Outcome Reward 整合

$$\hat{A}_{token}(t) = R^{TTRL}_i \cdot (1 + \lambda \cdot r_{step}(k(t), i))$$

其中 $k(t)$ 是 token $t$ 所属的步骤编号，$\lambda$ 控制 step shaping 的强度。

含义：
- $R^{TTRL} = 1$（正确）且 $r_{step} > 0$：**强正向更新**——关键正确步骤获得额外 credit
- $R^{TTRL} = 1$ 且 $r_{step} = 0$：正常正向更新——无可辨识实体的普通步骤
- $R^{TTRL} = 1$ 且 $r_{step} < 0$：**削弱正向更新**——正确轨迹中的可疑步骤（产出了错误组独有实体）
- $R^{TTRL} = -1$：依然是负向更新，但 step signal 区分"错在哪一步"

### Trajectory-Level 诊断指标

**锁定进度（Lock-in Progress）**：

$$L(\tau_i) = \frac{\min\{k \mid \forall k' \geq k, \exists e \in \mathcal{G}^+ : e \in E_{k'}^i\}}{T_i}$$

正确轨迹的 $L$ 应比错误轨迹更低（更早锁定到共识中间结果）。

**共识覆盖率（Consensus Coverage）**：

$$\text{Cov}(\tau_i) = \frac{|\mathcal{G}^+ \cap \bigcup_k E_k^i|}{|\mathcal{G}^+|}$$

正确轨迹应覆盖更多共识实体。

### SPC-Stat vs SPC-Probe vs Rollout-Tree 方法族

| 维度 | SPC-Stat (ours) | SPC-Probe (原始方案) | RTMC/GiGPO | Tree-GRPO |
|------|-----------------|---------------------|------------|-----------|
| 额外模型调用 | **零** | ~10 万次/step | 零 | 需要完整 rollout |
| 状态匹配方式 | 数学实体等价 + n-gram | 续写答案语义等价 | hash/exact string | prefix tree |
| 适用领域 | 数学推理 URLVR | 数学推理 URLVR | agentic (SWE/UI) | math + code |
| 需要 GT | **否** | 否 | 否 | 用 GT 做 reward |
| 步骤对齐 | **不需要** | probing point | 环境 turn 天然对齐 | prefix 天然对齐 |
| 信号粒度 | step-level | step-level | step-level | step-level |
| 理论支撑 | gradient entanglement (FPA) | CoVo consistency insight | MC estimation theory | shared prefix variance reduction (SLATE) |

### SPC-Stat 的风险与缓解

#### 风险 1：数学实体抽取质量不足

并非所有步骤都会产出可抽取的数学实体。纯文字推理步骤（如"考虑用配方法"）没有明确的数值或方程。

**缓解**：
1. n-gram 辅助信号作为 fallback
2. 实体抽取失败的步骤不给信号（precision-first 原则，参考 [[wiki/papers/plesner-2026-imperfect-verifier|Imperfect Verifier]]）
3. 离线统计实体抽取覆盖率，如果 > 60% 的步骤能抽到实体则可行

#### 风险 2：共识实体可能包含"共享错误"

如果大多数正确轨迹碰巧都经过同一个不必要的中间步骤，该实体也会被选为共识实体。

**缓解**：
1. 要求正确组独有（$\text{freq}_{\mathcal{E}} < 0.2$），排除两组都高频的实体
2. 实体贡献度加权——越晚出现的共识实体权重越低（因为越晚越可能是"回声"而非"驱动力"）

#### 风险 3：BranPO 发现分叉主要在尾部

BranPO 发现在 search agent 中，分叉主要发生在最后一步。如果数学推理也如此，那么 SPC-Stat 的"中间步骤分叉检测"的价值会打折。

**缓解**：
1. BranPO 的结论是 domain-specific 的（search agent），数学推理中一个早期逻辑错误可以 derail 整条轨迹
2. 实验中直接验证：统计数学推理中正确/错误组实体分叉的位置分布
3. 如果确实以尾部分叉为主，可以调整为"BranPO-style suffix contrastive"与 SPC-Stat 结合

## 实验设计

### Phase 1: 离线机制验证

目标：先证明 SPC 真的是一个比 semantic certainty 更强的 process signal。

实验：

1. 在 1000 个数学样本上，记录每步的 $SPC_k$、$Conf_k$、最终 correctness
2. 比较以下相关性：
   - $SPC_k$ vs 最终 correctness
   - semantic certainty vs 最终 correctness
   - CoVo consistency vs 最终 correctness
3. 检查错误轨迹上，SPC 是否能更早暴露不稳定步骤

关键指标：

- step-level AUROC
- trajectory-level Pearson / Spearman
- earliest error exposure step

#### Phase 1b: SPC-Stat 离线验证

目标：验证 SPC-Stat 的共识实体发现和分叉检测是否有效。

实验：

1. 对 1000 个数学样本各采样 64 条轨迹（复用 TTRL 训练数据）
2. 运行数学实体抽取，统计覆盖率（多少步骤能抽到 ≥1 个实体）
3. 运行共识实体发现，检查 $|\mathcal{G}^+|$ 和 $|\mathcal{B}|$ 的分布
4. 统计正确组 vs 错误组的分叉位置分布（验证 BranPO 的"尾部分叉"在数学推理中是否成立）
5. 比较共识实体首次出现步骤：正确组 vs 错误组的锁定进度 $L(\tau)$

关键指标：

- 实体抽取覆盖率（目标 > 60%）
- 共识实体数量及其区分能力（正确组 vs 错误组的频率差异）
- 分叉位置分布（归一化进度的直方图）
- 锁定进度 $L(\tau)$ 的组间差异（t-test）

### Phase 2: RL 训练对比

| 实验组 | Outcome Signal | Step Signal | 需要 GT? |
|--------|---------------|-------------|---------|
| TTRL | majority voting | 无 | 否 |
| TTRL + old SC | majority voting | semantic certainty | 否 |
| TTRL + CoVo-style shaping | majority voting | likelihood consistency | 否 |
| **TTRL + SPC (ours)** | majority voting | semantic process consistency | 否 |
| **TTRL + SPC-Stat (ours)** | majority voting | 共识实体 + n-gram 统计分叉 | 否 |
| **TTRL + SPC-Stat (entity only)** | majority voting | 仅共识实体信号（无 n-gram） | 否 |
| SPAE | GT binary | GT-based correctness | 是 |

评估数据集：AIME2024/2025、AMC23、MATH500。

重点回答三个问题：

1. SPC 是否优于纯 TTRL？
2. SPC 是否优于旧的 SC 替代方案？
3. SPC 是否优于 CoVo-style likelihood consistency？

4. SPC-Stat（零成本）是否接近 SPC-Probe（高成本）的效果？
5. n-gram 辅助信号是否真的有增益？（entity only vs full SPC-Stat）
6. 分叉检测在哪些类型的数学问题上最有效？（代数 vs 几何 vs 组合）

### Phase 3: 长期稳定性验证

重点验证 SPC 是否真的减弱了纯 intrinsic reward 的 reward hacking：

1. 跑长训练曲线（>500 steps）
2. 监控 rise-then-fall 是否被推迟
3. 监控 wrong-majority 样本上 SPC 的 filtering 效果
4. 监控输出长度和 R2W 比例

### Phase 4: Probe 近似

训练 SPC Probe，比较：

- sampling-based SPC
- probe-predicted SPC

看性能损失是否足够小，从而支撑更大规模实验。

## 理论与叙事

### Story line

1. TTRL 证明了无监督 outcome reward 可行，但缺 step-level correction。
2. SPAE 证明了 probing 能提取 step-level progress，但需要 GT correctness。
3. CoVo 进一步说明，中间状态与最终答案的一致性/波动性区分了好坏轨迹。
4. 我们把 CoVo 的 insight 从 **likelihood-space** 推到 **semantic rollout-space**。
5. 最终得到一个既无监督、又 step-aware、还能针对 TTRL 核心失败模式的一体化方案。

### 为什么这个故事比旧方案更强

旧方案的故事是“把 SE 塞进 SPAE”。

新方案的故事是：

> 现有 URLVR 要么只看最终答案一致性，要么只看 token-level probability consistency。我们提出 Semantic Process Consistency，直接检查每个推理前缀的后续生成行为是否语义上支持最终答案，从而为无监督 RLVR 提供真正面向 process correction 的 step-level signal。

这个定位明显更强，也更容易和 CoVo、TTRL、SPAE 形成清晰区分。

## 风险与缓解

### 风险 1：probe 太短，抽不出稳定答案

缓解：

1. 数学任务先用能快速落到 boxed answer 的 probe prompt
2. 对难样本动态延长 probe 长度
3. 离线先测 answer extraction 成功率

### 风险 2：最终答案本身就是错的，SPC 仍可能“稳定地支持错答案”

这是所有 internal signal 都无法彻底消除的问题。SPC 的目标不是完全解决 wrong-majority，而是**降低奖励泄露概率**。因此必须保留 TTRL outcome anchor，并在实验里专门分析 wrong-majority 子集。

### 风险 3：语义等价判断不稳定

缓解：

1. 数学任务优先 exact / symbolic equivalence
2. embedding threshold 只作为开放任务扩展
3. 用人工抽样校验 equivalence precision

### 风险 4：仍然属于 intrinsic reward，长期可能被 sharpen

[[wiki/papers/he-2026-urlvr-scale|He et al. 2026]] 的结论仍成立：任何内部信号都要警惕 sharpening。SPC 的价值不在于否认这个结论，而在于提出一个**更接近 process correction 的内部信号**，并验证它是否比 certainty-based signals 更耐用。

> **延伸方向**：为了从根本上缓解 intrinsic signal sharpening，我们设计了 [[wiki/synthesis/co-evolving-verifier-proposal|Co-Evolving Verifier]] 分支方案——用 SPC 标签周期性训练一个轻量级 PRM，日常 RL 中替代 probing，同时被 SPC 周期性校准，形成三层自举架构。详见该文档。

## Positioning

- vs [[wiki/papers/zuo-2025-ttrl|TTRL]]：我们补 process，不改 outcome。
- vs [[wiki/papers/wu-2026-spae|SPAE]]：我们保留 probing 框架，但去掉 GT 依赖。
- vs [[wiki/papers/zhang-2025-covo|CoVo]]：我们把 consistency 从 likelihood judgment 升级为 semantic rollout behavior。
- vs [[wiki/papers/zhang-2025-empo|EMPO]]：我们从 outcome-level semantic uncertainty 推进到 step-level process support。
- vs [[wiki/papers/ghimire-2026-prism|PRISM]]：我们仍使用 internal signal，但更直接面向“过程是否支持答案”，可作为更强的 process-side baseline。
- vs [[wiki/papers/du-2026-dual-consensus|DCRL]]：DCRL 改进 outcome-level 的 pseudo-label 质量（解决 spurious majority），SPC 改进 step-level 的 credit assignment。两者在不同层（outcome vs process）工作，互补而非竞争。
- vs [[wiki/papers/cui-2026-clipo|CLIPO]]：CLIPO 在 rollout-level 做跨轨迹对比学习（抑制 spurious reasoning），SPC 在 step-level 做语义一致性检查。CLIPO 可以直接集成到 SPC 框架中作为跨轨迹信号。
- vs [[wiki/papers/wang-2026-sarl|SARL]]：SARL 看推理拓扑结构（宏观），SPC 看语义一致性（微观）。两者评估推理质量的角度正交，可互补组合。
- vs [[wiki/papers/ma-2026-dcpo|DCPO]]：DCPO 解耦 accuracy 和 calibration 的梯度，SPC 需要借鉴类似思想解耦 outcome reward 和 step-level process signal 的梯度。
- vs [[wiki/papers/wang-2026-pipo|PIPO]]：PIPO 修复 GRPO 的梯度爆炸，SPC 实验应采用 PIPO 发现后的改进框架（DAPO 或 PIRL）以确保训练稳定性。
- vs [[wiki/papers/pan-2026-coverrl|CoVerRL]]：CoVerRL 验证了 generator-verifier co-evolution 在 outcome-level 的有效性；SPC 提供更细粒度的 step-level 信号，可作为 co-evolution 的内核升级。
- vs [[wiki/papers/yan-2026-scrl|SCRL]]：SCRL 的 negative pseudo-labeling 证明"知道什么是错的"比"知道什么是对的"在不确定场景更可靠。SPC 可以集成负信号——rollout 偏离最终答案的步骤提供 negative signal。
- vs [[wiki/papers/chen-2026-powerflow|PowerFlow]]：PowerFlow (Theorem D.1) 从理论上证明 MV-RLIF = extreme sharpening。SPC 旨在在 MV 之上叠加 process correction，缓解而非消除 sharpening。
- vs [[wiki/papers/yang-2026-distribttrl|DistriTTRL]]：DistriTTRL 改进 reward estimation 质量（GMM 置信度建模 + shift correction），SPC 改进 credit assignment 粒度。两者正交，可组合。
- vs [[wiki/papers/bai-2026-ttvs|TTVS]]：TTVS 增广数据（语义变体），SPC 增广信号（step-level consistency）。数据增广可以帮助 SPC 获得更多样化的 rollout。
- vs [[wiki/papers/yang-2026-olr|OLR]]：OLR 的 Early Correctness Coherence 为 SPC probing 提供理论基础——如果正确答案在训练早期就潜伏于模型中，step-level probing 就是在表面化这些潜伏信号。
- vs [[wiki/papers/gu-2026-asymgrpo|AsymGRPO]]：AsymGRPO 的 informative vs spurious entropy 分解可改进 SPC 的 confidence 信号设计——只保留 informative entropy 作为 Conf_k。
- vs [[wiki/papers/kim-2026-dbb|DBB]]：DBB 的 Beta-Bernoulli posterior 可作为 SPC 分数的校准信号，利用历史统计平滑当前 batch 估计，零额外计算。
- vs [[wiki/papers/ai-2026-shape|SHAPE]]：最直接竞品。SHAPE 做 segment-level solvability + token-level entropy 层次化分配。SPC 做 semantic rollout consistency。关键差异化：SHAPE 的 solvability 是启发式定义，SPC 的 consistency 是经验性 rollout 测量；SHAPE 需要 answer correctness，SPC 是无监督。
- vs [[wiki/papers/plesner-2026-imperfect-verifier|Imperfect Verifier]]：提供 SPC 不完美信号的容错理论——15% 噪声率仍鲁棒（仅降 2pp），SPC 只要保持高 precision 就能有效工作。
- vs [[wiki/papers/wang-2026-self-guide|Self-Guide]]：Self-Guide 的 policy-reward co-evolution loop 与 SPC 的 probing-based self-evaluation 高度同构。差异化：SPC 聚焦 step-level 语义一致性，Self-Guide 聚焦 trajectory-level internal reward。
- vs [[wiki/papers/zhang-2026-praise|PRAISE]]：**与 SPC 思想最接近的已有工作。** 两者都从 prefix 处做 rollout 得到 intermediate answer，用 answer 质量变化估计 step reward。关键差异化：(1) PRAISE 在 agentic search（multi-hop QA）中做，prefix 是 search turn 级别；SPC 在数学推理中做，prefix 是 reasoning step 级别。(2) PRAISE 用 GT answer scoring（$v_t = R(\hat{y}_t, y)$），SPC 与 final answer 对比（无监督）。(3) PRAISE 用 adjacent prefix gains（$r_t^{proc} = \alpha(v_t - v_{t-1})$），SPC 用 semantic equivalence consistency（$SPC_k = \frac{1}{M}\sum e_k^{(m)}$）。(4) PRAISE 用 PPO，SPC 用 GRPO/DAPO。总结：PRAISE 验证了"从 prefix 做 rollout 估计 step reward"这一范式的有效性，但它依赖 GT 且面向 agentic search；SPC 的核心卖点是**无监督 + 语义等价判断 + 面向数学推理**。
- vs [[wiki/papers/chen-2026-slate|SLATE]]：SLATE 也从 shared prefix 做 k 个 continuation 估计 step-level advantage，理论证明方差降低 T 倍。关键差异化：(1) SLATE 用 LLM judge 做 dense reward（需要外部 LLM），SPC 用 self-consistency 信号（无外部依赖）。(2) SLATE 的目标是 step-level advantage estimation 的方差降低，SPC 的目标是无监督 process correction。(3) SLATE 的 continuation 是完整 rollout 到最终答案，SPC 是短续写 probing。**可借鉴点**：SLATE 的方差降低理论（Theorem 1: $\text{Var} \propto 1/T$）为 SPC 的 prefix-based probing 提供理论支撑——共享前缀减少采样方差是有严格数学证明的。
- vs [[wiki/papers/liang-2026-step|STEP]]：STEP 用 hidden states 训练 lightweight MLP scorer 预测 trace quality，用于 inference-time pruning。关键差异化：(1) STEP 用于 test-time 推理加速（pruning），SPC 用于 training-time credit assignment。(2) STEP 的核心发现——"hidden states 在前 25% 步骤就能区分正确/错误 traces"——为 SPC probing 提供了重要佐证：如果 hidden states 在早期就编码了 trace quality 信息，那么 step-level probing（让模型从当前 state 短续写）应该也能在早期捕获过程质量。(3) STEP 的 2-layer MLP scorer（开销 < 10⁻⁶）可作为 SPC Probe Acceleration 阶段的参考架构。
- vs **Rollout-Tree 方法族** (RTMC/Tree-GRPO/BranchGRPO)：这些方法利用 group rollouts 在关键步骤隐式形成树结构，通过状态匹配实现 step-level credit assignment。**关键差异**：(1) 全部面向 agentic/tool-use 场景（SWE-bench, ALFWorld），状态匹配依赖环境返回的离散状态（hash 签名、exact string match）；数学推理中 64 条 CoT 在 token 级别几乎不可能共享前缀，需要语义级别的实体匹配。(2) SPC-Stat 用数学实体等价替代 token 状态匹配，是 Rollout-Tree 思想在自然语言数学推理中的首次实现。(3) SPC-Stat 完全无监督（TTRL pseudo-label），Rollout-Tree 方法族多数用 GT reward。
- vs [[2505.10978|GiGPO]]：GiGPO 的 anchor state grouping 与 SPC-Stat 的共识实体发现在思想层面高度同构——都是"跨轨迹找到相同状态，比较从该状态出发的不同动作的收益"。关键差异：GiGPO 用环境状态精确匹配（exact string / LMS ≥ 0.9），SPC-Stat 用数学实体语义等价。GiGPO 计算 discounted return 做 micro relative advantage，SPC-Stat 用共识实体首次出现做 binary credit。
- vs [[2604.11037|RTMC]]：RTMC 是 SPC-Stat 在 agentic 领域的近似实现——核心观察完全一致（rollouts 共享前缀→隐式树→分支发散→per-step Q-values）。差异化：(1) RTMC 用 hand-crafted state-action signatures（hash of edit content, view buckets），SPC-Stat 用 regex-based math entity extraction。(2) RTMC 面向 SWE-bench (code agent)，SPC-Stat 面向 URLVR 数学推理。(3) RTMC 需要 prior-based value smoothing（因为大多数状态只访问一次），SPC-Stat 的共识实体天然在正确组内有多次"访问"。
- vs [[2602.03719|BranPO]]：BranPO 发现在 search agent 中分叉主要在尾部，并在 shared prefix 上构建 contrastive suffix 训练。**重要参考**：BranPO 的尾部分叉结论是否适用于数学推理需要实验验证；如果数学推理中分叉确实在早中期（逻辑错误导致），则 SPC-Stat 的全步骤分析比 BranPO 的固定尾部分支更有价值。
- vs [[2602.09598|ELPO]]：ELPO 用 binary-search rollout tree 定位第一个不可恢复的错误步骤（irrecoverable step）。思想与 SPC-Stat 的分叉点检测一致，但 ELPO 需要额外 rollout（binary search），SPC-Stat 利用已有 64 条 rollout 零成本实现。
- vs [[2601.18984|VPPO]]：VPPO 将轨迹分为 verified correct prefix 和 erroneous suffix，前者正奖励、后者惩罚。SPC-Stat 的共识实体覆盖率（$\text{Cov}$）和锁定进度（$L$）提供了类似但更细粒度的分割——不是 binary 的 prefix/suffix，而是每步的 credit 量化。差异：VPPO 需要 GT + PRM，SPC-Stat 完全无监督。
- vs **FPA / Gradient Entanglement** ([[2509.19893|FPA]], [[2410.13828|Yuan et al. 2024]])：FPA 从理论上证明了正确/错误轨迹共享大量 token overlap 导致梯度冲突。SPC-Stat 的共识实体分析正是解决此问题的直接手段——通过精确定位正确/错误组的分叉实体，将 credit 集中在真正区分对错的步骤上，减少 shared token 上的梯度冲突。


## 2026 年新论文对 SPC 方案的系统性启发

> 2026 年 1-4 月发表的 10 篇新论文，从 Reward Estimation、训练稳定性、Label-Free RL 和多模态自进化四个方向，为 SPC 方案提供了重要的设计改进和验证信号。

### 一、Layer 1 (Outcome Anchor) 升级路径

SPC 三层架构的 Layer 1 当前使用 TTRL 的 naive majority voting。三篇新论文证明了这不是最优选择：

| 升级方案 | 论文 | 核心改进 | 对 SPC 的影响 |
|----------|------|----------|--------------|
| Dual Consensus | [[wiki/papers/du-2026-dual-consensus\|DCRL]] | Anchor+Explorer harmonic mean 解决 spurious majority | 减少 Layer 1 的伪标签错误率，SPC 基座更可靠 |
| Distribution-Aware | [[wiki/papers/du-2026-dare\|DARE]] | Uncertainty-normalized distribution 保留完整投票分布信息 | Information Collapse (Theorem 2.1) 理论与 SPC 的信息保留目标一致 |
| Tool Verification | [[wiki/papers/liao-2026-t3rl\|T³RL]] | Code execution 锚定 pseudo-label，N=16 超 TTRL N=64 | External anchor 使 SPC 的 intrinsic step signal 有更稳定的 outcome 基座 |

**实验建议**：SPC 正式实验中，Layer 1 应至少对比 naive TTRL vs DARE-enhanced TTRL 两个版本。T³RL 适合数学可编程验证的子集，DCRL 适合需要打破 spurious majority 的场景。

### 二、优化框架风险与解决方案

两篇论文揭露了 GRPO 的严重缺陷，直接影响 SPC 实验设计：

#### PIPO: GRPO 梯度爆炸 (arXiv:2604.00860)

[[wiki/papers/wang-2026-pipo\|PIPO]] 严格证明 GRPO 的 group-relative normalization 引入 gradient scaling factor：

$$\eta(p) \sim \frac{\sqrt{G-1}}{G \cdot p(1-p)} \to \infty \quad \text{when } p \to 0 \text{ or } p \to 1$$

**对 SPC 的影响**：
- SPC 在 step-level 做 shaping reward，如果某些步骤的 rollout 一致性接近 0 或 1，η(p) 会放大噪声
- Hard queries（模型初始正确率低）和 easy queries（正确率高）的梯度信号不稳定
- **建议**：SPC 实验应使用 DAPO（Dynamic Sampling 过滤全对/全错 groups）替代原始 GRPO，或集成 PIPO-style retrospective verification

#### DCPO: Accuracy-Calibration Gradient Conflict (arXiv:2603.09117)

[[wiki/papers/ma-2026-dcpo\|DCPO]] 首次证明 accuracy 优化和 calibration 优化在 Fisher 信息度量下梯度方向冲突：

$$\langle \nabla J_{acc}, \nabla J_{cal} \rangle_{F^{-1}} < 0$$

**对 SPC 的影响**：
- SPC 的 probing 信号本质上包含模型置信度信息。如果直接将 SPC step signal 与 TTRL outcome reward 相加，可能产生类似的梯度冲突
- **建议**：SPC 信号与 outcome reward 的整合方式应参考 DCPO 的解耦思想——不是简单加权求和，而是在梯度层面分别作用于不同 token 子集（如 reasoning tokens 接收 outcome gradient，transition tokens 接收 SPC gradient）

### 三、跨轨迹信号：CLIPO 直接解决 SPC 问题 #5

SPC 当前方案的第五个已知问题是"没有跨轨迹信息"——每条轨迹独立评估，不同正确轨迹之间的结构相似性被忽略。

[[wiki/papers/cui-2026-clipo\|CLIPO]] (arXiv:2603.10101, Alibaba Qwen) 提出的 InfoNCE 对比学习**直接解决此问题**：

- **核心机制**：在 group 内的 successful rollouts 上施加 contrastive loss，让正确推理路径的表示聚拢，偏离主流正确推理的 outlier 被推远
- **Reward augmentation 设计**：$r'_i = r_i + \max(-\lambda \cdot \mathcal{L}_{CL}(x, y_i), -0.5)$

**SPC 可以直接借鉴的设计**：
1. 在 SPC 框架中增加 step-level contrastive loss——比较不同轨迹在同一 step boundary 的 semantic rollout 一致性
2. SPC 的 reward 集成也可采用 CLIPO 的 augmentation 方式：$r'_i = r_{TTRL} + \max(-\lambda \cdot \Phi_{SPC}, -0.5)$ 而非直接替换 outcome reward
3. CLIPO 跨 4 种 RL 算法一致有效，证明 contrastive regularization 解决的是 RLVR 的通用问题

### 四、SARL 与 SPC 的互补关系

[[wiki/papers/wang-2026-sarl\|SARL]] (arXiv:2603.27977) 是另一种"看模型怎么做"的方法：

| 维度 | SARL | SPC |
|------|------|-----|
| 关注什么 | 推理链的**拓扑结构** | 推理步骤的**语义一致性** |
| 核心度量 | Small-world network: C(G) + 1/(1+L(G)) | Semantic rollout consistency: SPC_k |
| 是否需要答案 | 完全不需要 | 需要最终答案做对比 |
| 适用场景 | 数学 + 开放域 | 主要面向数学 |
| 惊人结果 | 无标注 +7.65 超越 GT RL +7.15 | — |

**互补设计方案**：SARL 评估宏观推理结构是否合理（高 C + 低 L），SPC 评估微观步骤是否语义连贯。两者可以组合为双信号 process reward。SARL 在开放域表现突出（WildBench +9.10 vs EMPO -0.71），如果 SPC 未来要扩展到非数学任务，SARL 的思路是天然候选。

### 五、Metacognitive Synergy: 支持 SPC 自评估路线

两篇多模态论文提供了自评估优于外部评估的证据：

1. **[[wiki/papers/tan-2026-meta-ttrl\|Meta-TTRL]]** 发现：7B 模型自我内省产生的 reward 信号 > 235B 外部模型评估。Capacity-matched signals > absolute evaluator strength
2. **[[wiki/papers/wang-2026-v-zero\|V-Zero]]** 发现：无监督 Questioner-Solver co-evolution (51.9) > 有监督 GRPO (50.8)

**对 SPC 的意义**：SPC 的核心路线是 probing-based self-evaluation（让模型自己短续写来评估步骤质量），而非训练额外的外部 PRM。Meta-TTRL 和 V-Zero 的结果为这条路线提供了信心——self-evaluation 不是退而求其次的选择，而可能是更优的设计。

### 六、V-Zero Dual-Track Reward 启发 SPC 信号设计

[[wiki/papers/wang-2026-v-zero\|V-Zero]] 的 Dual-Track Reasoning Reward 对比 intuition (System 1 快思考) vs reasoning (System 2 慢思考)：

- **一致 case**：给 ambiguity reward $r_d = \min(c, 1-c)$，奖励边界样本而非已确定的简单样本
- **分歧 case**：给 $r_d = 0.5 \cdot c$，推理推翻直觉时 reward 最高

**启发 SPC 信号设计**：
- 将 SPAE probing 的短续写答案视为 "intuition"（快速预测），将完整轨迹最终答案视为 "reasoning"
- SPC 在 consistency 已高的步骤适度降低 reward（类似 V-Zero 的 ambiguity reward 逻辑），在 consistency 低的步骤增大 reward（这些步骤是模型真正的提升空间）
- 这直接对接 SPAE 的 saturation penalty 设计

### 七、综合设计调整建议

基于以上分析，SPC 方案可以做以下具体改进：

| 编号 | 改进项 | 优先级 | 来源论文 |
|------|--------|--------|----------|
| A1 | Layer 1 从 naive MV 升级为 DARE distribution-aware | 高 | DARE |
| A2 | RL 优化框架从 GRPO 切换到 DAPO | 高 | PIPO |
| A3 | SPC 信号用 augmentation 方式集成（而非替换 outcome reward） | 高 | CLIPO |
| A4 | 新增 step-level contrastive loss 对比组 | 中 | CLIPO |
| A5 | SPC 与 outcome reward 梯度解耦 | 中 | DCPO |
| A6 | Layer 1 增加 DCRL dual consensus 对比组 | 中 | DCRL |
| A7 | Saturation penalty 参考 V-Zero 的 ambiguity reward 思路 | 低 | V-Zero |
| A8 | 开放域扩展考虑 SARL 拓扑 reward | 低 | SARL |
| A9 | OLR-style progressive refinement for SPC labels | 中 | OLR |
| A10 | Imperfect Verifier precision-first principle for SPC signal | 高 | Imperfect Verifier |
| A11 | SHAPE-style segment-level as SPC ablation baseline | 中 | SHAPE |
| A12 | DBB Beta posterior for SPC score smoothing | 低 | DBB |
| A13 | PRAISE-style adjacent prefix gains 作为 SPC 消融 baseline | 中 | PRAISE |
| A14 | SLATE 方差降低理论验证：比较 prefix-based vs trajectory-level SPC 方差 | 中 | SLATE |
| A15 | STEP-style MLP scorer 作为 SPC Probe Acceleration 的参考架构 | 低 | STEP |

### 八、第二批新论文的补充启发

> 第二批 6 篇新论文从 probing 理论基础、噪声容错、层次化竞品分析和低成本信号增强四个角度，进一步完善 SPC 的设计和定位。

#### 8.1 OLR: Early Correctness Coherence 为 SPC Probing 提供理论基础

[[wiki/papers/yang-2026-olr|OLR]] 发现模型在训练早期就已经"知道"正确答案——它们潜伏在模型中但未被表面化。这直接支持 SPC 的 probing 方法论：如果正确答案确实早期存在于模型中，那么 step-level probing 就是在表面化这些潜伏信号。同时，OLR 证明传统噪声处理方法在 RLVR 中灾难性失败（-17.4%），说明 SPC 不应套用 SFT 领域的噪声鲁棒技术。

**对 SPC 的具体启发**：
1. SPC probing 有理论依据——模型内部确实存在可以被 probing 表面化的"潜伏正确性"
2. SPC 的噪声处理策略应避免传统 label smoothing / confidence penalty 等 SFT 技术，转而采用 RLVR 原生的方法（如 precision-first filtering）
3. OLR 的 progressive refinement 思想可用于 SPC 标签的迭代改进——随训练推进，SPC 标签质量应逐步提升

#### 8.2 Imperfect Verifier: SPC 不完美信号的容错支撑

[[wiki/papers/plesner-2026-imperfect-verifier|Imperfect Verifier]] 证明 RLVR 对 15% 噪声率仍鲁棒（仅降 2pp）。这意味着 SPC 即使作为不完美的 step-level 信号，只要 precision 足够高，就能有效工作。

**关键原则："moderate accuracy + high precision"**——SPC 宁可在不确定的步骤上不给信号，也不要给错误信号。具体设计含义：
1. SPC 的阈值 δ 应偏保守——宁可漏掉一些关键步骤，也不错标非关键步骤
2. 当 M 次 probe 的答案高度分裂时（如 5 次 probe 给出 5 个不同答案），SPC_k 应直接输出"不确定"而非强行打分
3. Imperfect Verifier 的理论为 SPC 提供了一个可量化的容错上界——只要 SPC 的 step-level precision > 85%，整体训练效果就不会显著退化

#### 8.3 SHAPE: 层次化信用分配的直接竞品与互补

[[wiki/papers/ai-2026-shape|SHAPE]] 是 SPC 最直接的竞品，也是定位时必须仔细区分的对象。对比：

| 维度 | SHAPE | SPC |
|------|-------|-----|
| 段级信号 | Solvability potential（启发式） | SPC_k（经验性 rollout） |
| Token 级 | Entropy redistribution | Saturation penalty + difference shaping |
| 需要 GT | 是（correctness） | 否（无监督） |
| 效率 | 高（无额外采样） | 中（需要 probing） |
| 信号来源 | 分析轨迹内部结构 | 通过 rollout 经验测量 |

**关键差异化**：
- SHAPE 的 solvability 是启发式定义（"从当前段继续，问题还可能被解决的概率"），依赖于 answer correctness 信号来计算
- SPC 的 consistency 是通过实际 rollout 经验测量的（"从当前步继续，模型真正会生成什么答案"），不依赖外部标签
- **SPC 无监督是最大卖点**——SHAPE 需要知道最终答案是否正确，SPC 只需要最终答案本身

**实验设计建议**：
- 在 ablation 中加入 SHAPE-style segment-level solvability 作为 baseline
- 比较 SPC 的 empirical rollout vs SHAPE 的 heuristic solvability 在 step-level 信号质量上的差异
- 在无 GT 设定下（URLVR），SHAPE 不能直接使用，但可以用 TTRL pseudo-label 近似实现一个 SHAPE 变体作为对照

#### 8.4 Self-Guide 与 Co-Evolving Verifier 的同构性

[[wiki/papers/wang-2026-self-guide|Self-Guide]] 的 policy-reward co-evolution loop 与我们的 [[wiki/synthesis/co-evolving-verifier-proposal|Co-Evolving Verifier]] 方案高度同构。但 Self-Guide 用于 agent 任务（非数学推理），且其 internal reward 是 trajectory-level 而非 step-level。

**SPC + Co-Evolving Verifier 的差异化在于**：
1. **Step-level granularity**：SPC 提供逐步信号，Self-Guide 只有轨迹级 reward
2. **Semantic rollout consistency signal**：SPC 的信号来源是 probing-based rollout，Self-Guide 的信号来源是 policy 自身的 trajectory reward
3. **三层自举架构**：SPC → Co-Evolving PRM → RL policy 的三层结构比 Self-Guide 的双层 policy-reward 更稳定

**可借鉴之处**：Self-Guide 在 agent 任务上的成功为 SPC 向非数学任务扩展提供了路线参考——在 agent 场景中，SPC 的 probing 可以改为"从当前状态继续执行，观察任务完成情况"。

#### 8.5 DBB 与 AsymGRPO: 低成本信号增强

两篇论文为 SPC 提供了零或极低额外成本的信号增强手段：

**[[wiki/papers/kim-2026-dbb|DBB]] 的 Beta-Bernoulli posterior**：
- DBB 利用历史统计信息（每个问题的历史正确率）构造 Beta posterior，作为 reward 的贝叶斯平滑
- 可直接应用于 SPC：将每个步骤的历史 SPC_k 分布建模为 Beta 分布，用 posterior mean 替代单次 batch 的 SPC 估计
- **优势**：零额外计算开销，只需要维护历史统计；可以平滑 SPC 信号在不同 batch 间的方差
- **风险**：如果模型能力快速变化，历史统计可能过时。可参考 [[wiki/papers/yang-2026-distribttrl|DistriTTRL]] 的 shift correction 思想进行补偿

**[[wiki/papers/gu-2026-asymgrpo|AsymGRPO]] 的 informative vs spurious entropy 分解**：
- AsymGRPO 证明 entropy 中包含 informative（有用的探索性不确定性）和 spurious（无意义的噪声）两个成分
- SPC 当前使用 Conf_k（基于 entropy 的置信度）作为辅助信号，但没有区分 entropy 的来源
- **改进方案**：用 AsymGRPO 的分解方法，只保留 informative entropy 计算 Conf_k，过滤掉 spurious entropy 引入的噪声
- 这可以使 SPC 的 $\Phi_{SPC} = 1.5 \cdot SPC_k \cdot Conf_k + 0.5 \cdot SPC_k - Conf_k$ 中的 Conf_k 更准确

#### 8.6 其他补充：CSRS、DistriTTRL、TTVS 的间接启发

**[[wiki/papers/yu-2026-csrs|CSRS]] 的 retracing re-inference anchoring**：
- CSRS 提出在推理过程中回溯到关键决策点重新推理，作为一种 anchoring 技术
- 可应用于 SPC probing：当某步的 SPC_k 突然下降时，从该步之前的高 SPC 点重新 probe，确认下降是否由该步骤引起
- 这提供了一种更精确的 "关键步骤定位" 机制

**[[wiki/papers/yang-2026-distribttrl|DistriTTRL]] 的 GMM 置信度建模**：
- DistriTTRL 用 GMM 建模 reward 分布，实现更细粒度的置信度估计
- 可用于建模 SPC 分数的分布——当 SPC_k 在不同问题上呈多峰分布时，GMM 可以识别出不同的"步骤类型"（探索步骤 vs 锁定步骤 vs 冗余步骤）
- DistriTTRL 的 shift correction 可用于补偿 SPC 历史 rollout 聚合时的分布偏移

**[[wiki/papers/bai-2026-ttvs|TTVS]] 的语义变体增广**：
- TTVS 通过生成语义等价的问题变体来增广训练数据
- 对 SPC 的直接好处：更多样化的问题变体 → 更多样化的 rollout → 更可靠的 SPC 信号估计
- TTVS 的 hybrid IGE/CGE exploration 思想也可应用于 SPC probing 的温度调节——在训练早期用高温多样性 probing，后期用低温精确 probing

## 潜在标题

- *"Semantic Process Consistency: Step-Level Correction for Unsupervised RLVR"*
- *"From Outcome Voting to Process Support: Unsupervised Step-Level Credit Assignment via Semantic Rollouts"*
- *"Beyond Likelihood Consistency: Semantic Process Consistency for RLVR"*
