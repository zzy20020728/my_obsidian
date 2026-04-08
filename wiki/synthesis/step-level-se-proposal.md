---
title: "Semantic Process Consistency: 无监督步骤级纠错研究方案"
type: synthesis
tags: [semantic-process-consistency, SPC, step-level, URLVR, SPAE, TTRL, CoVo, process-reward, research-proposal, reward-hacking]
created: 2026-04-08
updated: 2026-04-08
sources: [wiki/papers/zuo-2025-ttrl.md, wiki/papers/wu-2026-spae.md, wiki/papers/zhang-2025-covo.md, wiki/papers/zhang-2025-empo.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/he-2026-urlvr-scale.md, wiki/concepts/process-reward-model.md, wiki/concepts/reward-hacking.md]
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

### Phase 2: RL 训练对比

| 实验组 | Outcome Signal | Step Signal | 需要 GT? |
|--------|---------------|-------------|---------|
| TTRL | majority voting | 无 | 否 |
| TTRL + old SC | majority voting | semantic certainty | 否 |
| TTRL + CoVo-style shaping | majority voting | likelihood consistency | 否 |
| **TTRL + SPC (ours)** | majority voting | semantic process consistency | 否 |
| SPAE | GT binary | GT-based correctness | 是 |

评估数据集：AIME2024/2025、AMC23、MATH500。

重点回答三个问题：

1. SPC 是否优于纯 TTRL？
2. SPC 是否优于旧的 SC 替代方案？
3. SPC 是否优于 CoVo-style likelihood consistency？

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

## 潜在标题

- *"Semantic Process Consistency: Step-Level Correction for Unsupervised RLVR"*
- *"From Outcome Voting to Process Support: Unsupervised Step-Level Credit Assignment via Semantic Rollouts"*
- *"Beyond Likelihood Consistency: Semantic Process Consistency for RLVR"*
