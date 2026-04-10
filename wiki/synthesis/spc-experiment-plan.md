---
title: "SPC 实验设计方案：从 TTRL 环境复现 SPAE 到无监督步骤级奖励"
type: synthesis
tags: [SPC, experiment-plan, TTRL, SPAE, URLVR, step-level, reward-shaping, implementation-guide, PIPO, DCPO, DARE, CLIPO, DAPO, gradient-conflict, contrastive-learning, CoVerRL, SCRL, DistriTTRL, TTVS, OLR, AsymGRPO, DBB, SHAPE, Imperfect-Verifier, Self-Guide, noise-robustness, data-augmentation, Bayesian-reward, hierarchical-credit]
created: 2026-04-08
updated: 2026-04-10
sources: [wiki/synthesis/step-level-se-proposal.md, wiki/papers/wu-2026-spae.md, wiki/papers/zuo-2025-ttrl.md, wiki/papers/zhang-2025-covo.md, wiki/papers/he-2026-urlvr-scale.md, wiki/papers/wang-2026-pipo.md, wiki/papers/ma-2026-dcpo.md, wiki/papers/du-2026-dare.md, wiki/papers/cui-2026-clipo.md, wiki/papers/du-2026-dual-consensus.md, wiki/papers/liao-2026-t3rl.md, wiki/papers/wang-2026-sarl.md, wiki/papers/pan-2026-coverrl.md, wiki/papers/yan-2026-scrl.md, wiki/papers/yang-2026-distribttrl.md, wiki/papers/bai-2026-ttvs.md, wiki/papers/yang-2026-olr.md, wiki/papers/gu-2026-asymgrpo.md, wiki/papers/kim-2026-dbb.md, wiki/papers/ai-2026-shape.md, wiki/papers/plesner-2026-imperfect-verifier.md, wiki/papers/wang-2026-self-guide.md]
status: active
---

# SPC 实验设计方案：从 TTRL 环境复现 SPAE 到无监督步骤级奖励

## 一句话目标

> 先在现有 [[wiki/papers/zuo-2025-ttrl|TTRL]] 训练环境里做出一个 **可运行、可观测、可对照** 的 [[wiki/papers/wu-2026-spae|SPAE]] 风格 step-level credit assignment 管线，再逐步把其中依赖 GT 的部分替换为无监督信号，最终得到 [[wiki/synthesis/step-level-se-proposal|SPC]] 方案。

## 给下一个 agent 的核心原则

这套实验不要一上来就做“最终版”。正确顺序是：

1. **先跑通工程骨架**：在 TTRL 环境里插入 step boundary、probe、step potential、token-level shaping。
2. **先做有监督上界**：先用 GT correctness 复现一个 SPAE-like 版本，确认代码链路是对的。
3. **再替换 reward source**：把 GT correctness 换成 pseudo-label / self-consistency / semantic signal。
4. **一次只改一个模块**：每个阶段只回答一个问题，避免调参时不知道是谁起作用。
5. **先做小规模短训练**：先看曲线和日志，再做长训练。

如果你是小白，可以把它理解成：

- `TTRL` 负责回答“**最终答案像不像对的**”
- `SPAE/SPC` 负责回答“**中间哪些步骤真的有用**”

## 总体研究问题

我们想验证 4 个问题：

1. 仅有 TTRL 的 outcome reward 时，是否存在明显的 step-level credit assignment 粗糙问题？
2. 在同一训练环境中，SPAE-style shaping 是否能先作为一个有效工程增强件工作？
3. 把 SPAE 中依赖 GT 的 correctness 换成无监督 proxy 后，是否仍然有效？
4. 不同无监督 step signal 中，哪一个最值得继续做大实验：pseudo-label correctness、confidence、semantic entropy、SPC？

## 整体技术路线

### 主线

`TTRL baseline -> TTRL + SPAE-GT -> TTRL + pseudo-label SPAE -> TTRL + confidence/SE baselines -> TTRL + SPC`

### 每一阶段的角色

| 阶段 | 目的 | 结论类型 |
|------|------|----------|
| Phase 0 | 跑通环境与日志 | 工程验证 |
| Phase 1 | 在 TTRL 环境复现 SPAE 骨架 | 上界 / sanity check |
| Phase 2 | 用 TTRL pseudo-label 替换 GT correctness | 第一版无监督 step reward |
| Phase 3 | 加简单模块做对照：confidence / semantic entropy | baseline 比较 |
| Phase 4 | 实现 SPC | 主实验 |
| Phase 5 | 长训练与稳定性分析 | 论文结论 |

## 为什么这样设计

这是最稳的顺序，因为它把问题拆成了三个层次：

1. **代码问题**：step 切分、probe、advantage shaping 能不能接进 TTRL。
2. **信号问题**：GT correctness 换成 pseudo correctness 后还有没有信息量。
3. **研究问题**：SPC 是否比 confidence / semantic entropy 更强。

如果跳过 Phase 1，直接做 SPC，一旦结果不好，你无法判断到底是：

- probe 写错了
- step mapping 写错了
- shaping 写错了
- pseudo-label 不准
- semantic equivalence 不准
- 还是 SPC 本身不 work

## Phase 0：环境接入与最小可运行版本

### 目标

在现有 TTRL 代码里，插入一个最小 step-level hook，但先**不改变训练逻辑**，只做日志。

### 需要实现的模块

1. **step splitter**
2. **probe sampler**
3. **answer extractor**
4. **step-to-token mapper**
5. **debug logger**

### 输入输出约定

给定一条 response：

- 输入：`question + full reasoning trajectory + final answer`
- 输出：
  - step 列表
  - 每个 step 的 prefix
  - 每个 prefix 的 probe continuations
  - 每个 probe continuation 抽取出的答案
  - 每个 step 对应的 token span

### 验收标准

满足下面 5 条就算 Phase 0 完成：

1. 能从一条 response 中稳定切出 steps
2. 每个 step 后都能发 probe prompt 并采样出短续写
3. 数学任务中，至少大部分 probe 都能抽出 boxed / numeric answer
4. 能把 step-level score 回填到 token span
5. 日志里能看到一条样本的 step-level 可视化结果

### 建议先做的 debug 样本数

- 先做 `16~32` 条样本
- 不要一开始就全量训练

## Phase 1：在 TTRL 环境中先做 SPAE-GT 上界

### 目标

先不追求无监督，先在你的 TTRL 环境里做一个 **SPAE-like oracle 版**，验证整条 credit assignment 链路是通的。

### 这里为什么允许用 GT

这一步不是最终方案，而是**工程校准**。

目的只有两个：

1. 证明“step probing + shaping”这套代码在你的环境里能工作
2. 给后续无监督版本提供一个上界参考

### 实现方式

保持 TTRL 的 rollout 和训练框架不动，只额外计算：

- `Conf_k`：沿用 SPAE 的 probe entropy confidence
- `Acc_k^GT`：用真实答案做 force-feed correctness
- `Phi_k^GT`：按 SPAE 公式合成 step potential
- `A_token^step`：按 saturation penalty + difference shaping 注入 token-level advantage

### 训练组

| 组别 | Outcome reward | Step signal | 用途 |
|------|----------------|-------------|------|
| G0 | TTRL | 无 | baseline |
| G1 | GT reward 或 TTRL | SPAE-GT | 工程上界 |

如果你暂时不方便切到 GT outcome reward，也可以继续保留 TTRL outcome reward，只把 step correctness 用 GT 计算。这样仍然足够做工程校验。

### 这一阶段主要看什么

1. 长度是否下降
2. checking tokens 是否下降
3. R2W 是否下降
4. 短训练下精度是否不低于 vanilla TTRL
5. 日志里 `Phi_k` 曲线是否符合直觉：探索期接近 0，稳定后升高，回退时下降

### 这一阶段的意义

如果 Phase 1 都跑不出合理趋势，后面不要急着做 SPC，先修代码。

## Phase 2：把 GT correctness 替换成 TTRL pseudo-label correctness

### 目标

做出第一版真正可训练的**无监督 SPAE 替代物**。

### 核心改动

把 SPAE 中的：

- `Acc_k^GT = prefix 对 ground-truth answer 的支持`

替换为：

- `Acc_k^Pseudo = prefix 对 TTRL majority pseudo-answer 的支持`

### 两种实现方式

#### 方案 A：force-feed pseudo answer

直接把 TTRL majority answer 当作“伪 GT”，沿用 SPAE 的 force-feeding 计算：

`Acc_k^Pseudo-FF`

优点：

- 改动最小
- 最适合第一版上线
- 能最大复用 SPAE 代码结构

缺点：

- 本质还是 likelihood-space
- 会继承 TTRL pseudo-label 的噪声

#### 方案 B：probe-match pseudo answer

不 force-feed，而是看 probe continuation 抽出的答案是否等于 pseudo answer：

`Acc_k^Pseudo-Probe = mean[ ans(probe_m) == a_maj ]`

优点：

- 更接近之后的 SPC
- 不再依赖 answer token force-feeding

缺点：

- 工程稍复杂
- 前期可能更不稳定

### 建议

先做 **方案 A**，因为它是最平滑的过渡版本。

一句话：

`先把 GT answer 换成 pseudo answer，再考虑把 force-feed correctness 换成 rollout-based correctness。`

### 训练组

| 组别 | Outcome reward | Step signal |
|------|----------------|-------------|
| G0 | TTRL | 无 |
| G1 | TTRL | SPAE-GT |
| G2 | TTRL | SPAE-Pseudo-FF |
| G3 | TTRL | SPAE-Pseudo-Probe |

### 这一阶段回答的问题

1. 只替换 GT 标签后，step shaping 还能不能工作？
2. pseudo-force-feed 和 probe-match 哪个更稳？
3. 无监督 step signal 是否至少优于纯 TTRL baseline？

## Phase 3：加简单 baseline，别直接跳 SPC

### 目标

在实现 SPC 之前，先做两个更简单的对照版本，帮助判断“到底是 process consistency 有用，还是任何 step-level 信号都行”。

### Baseline 1：Confidence-only

只用 probe entropy 做 `Conf_k`，不引入 correctness。

形式可以很简单：

- `Phi_k = Conf_k`
- 或者 `Phi_k = 2 * Conf_k - 1`

意义：

- 检验“只有模型自信度”是否足够
- 这是最弱但最容易实现的 baseline

### Baseline 2：Semantic Entropy step signal

对每个 step 的多次 probe continuation 做答案聚类，计算 semantic entropy：

- 高 entropy = 当前 step 后续答案分歧大
- 低 entropy = 当前 step 后续答案趋于一致

可以定义：

- `SC_k = 1 - SE_k / SE_max`

意义：

- 对照旧的 step-level semantic certainty 方案
- 方便判断 SPC 是否真的优于“只是看一致性/确定性”

### 训练组

| 组别 | Outcome reward | Step signal |
|------|----------------|-------------|
| G0 | TTRL | 无 |
| G2 | TTRL | SPAE-Pseudo-FF |
| G4 | TTRL | Confidence-only |
| G5 | TTRL | Step Semantic Entropy |

### 这一阶段不要做的事

1. 不要同时引入 CoVo vector aggregation
2. 不要同时训练 verifier
3. 不要同时做 curriculum
4. 不要同时改 RL 算法

先把对照做干净。

## Phase 4：实现 SPC 主方案

### 目标

把“correctness 是不是支持某个答案”从 **likelihood 判断** 升级成 **semantic rollout behavior 判断**。

### SPC 定义

对每个 step prefix 做 `M` 次短 probe，抽取每条 probe continuation 的答案 `a_k^(m)`，与该轨迹最终答案 `a_final` 做等价判断：

`SPC_k = mean[ a_k^(m) semantically matches a_final ]`

### 最小实现版

在数学任务上，先不要做复杂语义匹配，直接用：

1. exact string match
2. numeric normalization 后 exact match
3. boxed answer extraction

只要数学抽取稳定，第一版就够了。

### 第一版推荐公式

先不要搞很复杂，直接上最稳的两种之一：

#### 版本 1：只用 SPC

- `Phi_k = 2 * SPC_k - 1`

#### 版本 2：SPC + Conf

- `Phi_k = 1.5 * SPC_k * Conf_k + 0.5 * SPC_k - Conf_k`

建议先做 **版本 1**，因为更容易解释，也更容易 debug。

如果版本 1 有效果，再加版本 2。

### 轨迹级附加统计

除了 `SPC_k`，还要记录：

- `Con_SPC = mean_k SPC_k`
- `Vol_SPC = last step with SPC_k < delta`

这两个指标先用于日志和分析，不一定马上进 reward。

### 训练组

| 组别 | Outcome reward | Step signal |
|------|----------------|-------------|
| G0 | TTRL | 无 |
| G4 | TTRL | Confidence-only |
| G5 | TTRL | Step Semantic Entropy |
| G2/G3 | TTRL | pseudo-label SPAE |
| G6 | TTRL | SPC-only |
| G7 | TTRL | SPC + Conf |

## Phase 5：长期稳定性与 sharpening 分析

### 目标

不是只看短期涨不涨点，而是看哪个信号**更晚坏掉**。

### 要监控的曲线

1. train reward
2. validation accuracy
3. response length
4. majority label accuracy（如果能离线估）
5. wrong-majority 子集表现
6. `Con_SPC` / `Vol_SPC`
7. R2W rate

### 重点问题

根据 [[wiki/papers/he-2026-urlvr-scale|He et al. 2026]]，所有 intrinsic signal 最终都可能 sharpen，所以这里不要求“永不退化”，而是比较：

1. 谁起效更快
2. 谁 peak 更高
3. 谁退化更晚
4. 谁对 wrong-majority 更鲁棒

## 建议的数据与训练预算

### 工程调试阶段

- 训练数据：先用一个小子集
- 推荐：`128 ~ 512` 个样本
- 目的：快速看到日志，验证管线，不追求最终分数

### 机制验证阶段

- 推荐：`1k ~ 2k` 样本
- 目的：看不同 step signal 的相对趋势

### 正式实验阶段

- 再上更大训练集
- 优先看 `AIME24 / AMC23 / MATH500`

### 为什么先小后大

因为你的核心任务不是一开始刷榜，而是先知道：

- 信号有没有信息量
- 代码有没有 bug
- 曲线是不是符合预期

## 评估指标

### 结果指标

1. Pass@1 / Acc@1
2. Pass@k（如果已有评估脚本）
3. 平均输出长度

### 过程指标

1. 每条轨迹的 step 数
2. solve tokens / check tokens
3. R2W rate
4. `Phi_k` 时序曲线
5. `SPC_k` 时序曲线
6. answer extraction success rate

### 稳定性指标

1. rise-then-fall 是否出现
2. peak step 在哪里
3. collapse 前是否已有 process-level 预警

## 推荐的最小实验矩阵

如果算力有限，不要一下子全做，优先跑下面 6 组：

| 优先级 | 实验组 | 目的 |
|--------|--------|------|
| P0 | TTRL | 主 baseline |
| P1 | TTRL + SPAE-GT | 工程上界 |
| P2 | TTRL + SPAE-Pseudo-FF | 第一版无监督 step reward |
| P3 | TTRL + Confidence-only | 最简 baseline |
| P4 | TTRL + Step Semantic Entropy | 旧思路 baseline |
| P5 | TTRL + SPC-only | 主方法 |

如果这 6 组跑清楚，论文主线已经很完整了。

## 给下一个 agent 的实现优先级

### 第一优先级

1. 在 TTRL rollout 后拿到 step 边界
2. 实现 probe sampling
3. 实现 step-to-token mapping
4. 把 step-level `Phi_k` 转成 token-level shaping advantage
5. 先接 SPAE-GT

### 第二优先级

1. 接 pseudo-label force-feed correctness
2. 加日志和可视化
3. 做小规模训练 sanity check

### 第三优先级

1. 加 confidence-only
2. 加 step semantic entropy
3. 最后接 SPC-only

### 先不要做

1. 不要先做 co-evolving verifier
2. 不要先做复杂 semantic embedding matching
3. 不要先做多任务扩展
4. 不要先做论文级大规模 sweep

## 最容易踩坑的地方

### 坑 1：step 切分不稳定

如果 step boundary 不稳定，后面所有 step-level 信号都不可信。

### 坑 2：probe 抽不出答案

如果 probe continuation 经常没有明确答案，SPC 和 semantic entropy 都会变成噪声。

### 坑 3：token 映射错位

如果 `Phi_k` 没有正确映射回对应 token span，shaping reward 等于白加。

### 坑 4：一次改太多

最容易把自己绕晕。每次只改一个模块。

### 坑 5：只看最终分数，不看过程日志

这个方向的关键不是只有 final accuracy，而是要知道：

- 哪一步开始稳定
- 哪一步发生回退
- 哪些轨迹属于伪正样本

## 我建议的最终主叙事

论文/汇报时可以这样讲：

1. TTRL 解决了无监督 outcome reward 的问题，但 step-level credit assignment 很粗。
2. SPAE 提供了很好的 step-level probing 框架，但 correctness 依赖 GT。
3. 我们先在 TTRL 环境中复现 SPAE 骨架，验证 step shaping 的工程可行性。
4. 然后逐步把 GT correctness 换成 pseudo-label correctness、confidence、semantic entropy。
5. 最终提出 SPC：直接看某一步之后 rollout 出来的答案，是否语义上支持最终答案。

这条线非常自然，也非常适合你现在的工程条件。

## 一句话执行建议

> 下一个 agent 不要直接实现最终版 SPC。先用现有 TTRL 环境做出 `SPAE-GT -> SPAE-Pseudo-FF -> Confidence / Step-SE -> SPC-only` 这条渐进路线，每一步都保留日志、可视化和小规模 sanity check。

## 基于 2026 新论文的实验设计修订

> 2026 年 1-4 月发表的新论文揭露了 GRPO 的系统性缺陷、提供了更可靠的 TTRL anchor、以及新的信号集成方式。以下修订直接影响 SPC 实验的设计选择。

### 修订 1：优化框架从 GRPO 切换到 DAPO

**依据**：[[wiki/papers/wang-2026-pipo|PIPO]] (arXiv:2604.00860) 严格证明 GRPO 的 group-relative normalization 引入 gradient scaling factor η(p) ∝ 1/[p(1-p)]，在 p→0（全错组）或 p→1（全对组）时梯度爆炸→mode collapse。

**对 SPC 的具体影响**：
- SPC 的 step-level shaping reward 在 GRPO 下会被 η(p) 放大——hard queries 的 step signal 噪声会被无限放大
- 数学推理训练中必然存在 hard queries（p→0），GRPO 在这些 query 上的梯度信号不可靠

**修订建议**：
- 主实验全部使用 DAPO（已有 Dynamic Sampling 过滤全对/全错 groups）
- 保留一组 GRPO 作为对照，但不作为主要结论的基础
- 如果观察到训练不稳定，检查是否是 η(p) boundary explosion 导致

**实验矩阵更新**：

| 优先级 | 实验组 | 优化框架 | 目的 |
|--------|--------|----------|------|
| P0 | TTRL (DAPO) | DAPO | 主 baseline |
| P0-alt | TTRL (GRPO) | GRPO | GRPO 对照 |
| P1 | TTRL + SPAE-GT (DAPO) | DAPO | 工程上界 |
| P2 | TTRL + SPAE-Pseudo-FF (DAPO) | DAPO | 第一版无监督 |
| P3 | TTRL + Confidence-only (DAPO) | DAPO | 最简 baseline |
| P4 | TTRL + Step SE (DAPO) | DAPO | 旧思路 baseline |
| P5 | TTRL + SPC-only (DAPO) | DAPO | 主方法 |
| P6 | TTRL + SPC + CLIPO contrastive (DAPO) | DAPO | 跨轨迹信号对比 |

### 修订 2：SPC 信号集成方式——Augmentation 而非替换

**依据**：[[wiki/papers/cui-2026-clipo|CLIPO]] (arXiv:2603.10101) 的 reward augmentation 设计证明，将 process signal 作为 reward 调整项（而非替换 outcome reward）更稳定：

$$r'_i = r_i + \max(-\lambda \cdot \mathcal{L}_{CL}, -0.5)$$

**修订建议**：SPC 的 step-level reward 也应采用 augmentation 方式集成：

$$r'_i = r_{TTRL} + \alpha \cdot \max(\Phi_{SPC}, \text{lower\_bound})$$

而非之前计划的：

$$\hat{A}_{i,j} = \hat{A}^{Group}_i \cdot f(\Phi_{SPC}) + \xi \cdot g(\Delta \Phi_{SPC})$$

**好处**：
1. 保持 TTRL outcome reward 的主导地位，SPC 只做 bonus/penalty 调整
2. lower_bound 防止 SPC 信号过大导致 reward 崩溃
3. 与 CLIPO 的 contrastive loss 可以进一步叠加
4. 更简单，更容易 debug

**新增实验组**：

| 组别 | 集成方式 | 目的 |
|------|----------|------|
| G6a | SPC as augmentation ($r' = r_{TTRL} + \alpha \cdot \Phi_{SPC}$) | 新建议方式 |
| G6b | SPC as multiplicative ($r' = r_{TTRL} \cdot (1 + \beta \cdot \Phi_{SPC})$) | 乘法对照 |
| G6c | SPC as original shaping ($\hat{A} = \hat{A}^{Group} \cdot f(\Phi)$) | 原始方式对照 |

### 修订 3：Layer 1 Anchor 升级

**依据**：
- [[wiki/papers/du-2026-dare|DARE]] (ICML 2026) 证明 naive MV 存在 Information Collapse (Theorem 2.1)，distribution-aware reward AIME24 +25.3%
- [[wiki/papers/du-2026-dual-consensus|DCRL]] 证明 dual consensus 解决 spurious majority
- [[wiki/papers/liao-2026-t3rl|T³RL]] 证明 tool verification 使 N=16 > TTRL N=64

**修订建议**：在实验矩阵中加入 Layer 1 anchor 升级的对比：

| 组别 | Layer 1 Anchor | Step Signal | 目的 |
|------|---------------|-------------|------|
| G0 | TTRL (naive MV) | 无 | 原始 baseline |
| G0-dare | TTRL (DARE) | 无 | 升级 anchor baseline |
| G5-dare | TTRL (DARE) | SPC-only | SPC + 升级 anchor |

**实现优先级**：DARE 实现最简单（只改 reward 计算，不改框架），建议优先实现。DCRL 需要维护两个模型，T³RL 需要 code interpreter，放到后续阶段。

### 修订 4：SPC 与 Outcome Reward 的梯度解耦

**依据**：[[wiki/papers/ma-2026-dcpo|DCPO]] (arXiv:2603.09117) 证明 accuracy 和 calibration 在 Fisher 信息度量下存在 gradient conflict（内积 < 0）。

**对 SPC 的启示**：SPC 的 probing 信号包含模型置信度信息，与 outcome accuracy reward 可能存在类似的梯度冲突。

**修订建议**：
- 不直接合并 SPC gradient 和 TTRL gradient
- 参考 DCPO 的 masked gradient 思路：reasoning tokens 接收 outcome gradient，step transition tokens 接收 SPC gradient
- 这是一个可选的高级对照组，优先级中等

**新增实验组**：

| 组别 | 梯度方式 | 目的 |
|------|----------|------|
| G6-coupled | SPC + TTRL 梯度直接相加 | 基础方式 |
| G6-decoupled | SPC + TTRL 梯度解耦（DCPO-style mask） | 解耦对照 |

### 修订 5：新增 Contrastive Learning 对比组

**依据**：[[wiki/papers/cui-2026-clipo|CLIPO]] 的 InfoNCE 对比学习跨 4 种 RL 算法一致有效，且直接解决 SPC 方案问题 #5（没有跨轨迹信息）。

**修订建议**：在 Phase 4 的实验矩阵中新增 contrastive learning 对比组：

| 组别 | 信号组合 | 目的 |
|------|----------|------|
| G8 | TTRL + CLIPO contrastive only | Contrastive-only baseline |
| G9 | TTRL + SPC + CLIPO contrastive | SPC + 跨轨迹信号 |

### 修订后的最小实验矩阵（第三版）

如果算力有限，优先跑以下 11 组：

| 优先级 | 实验组 | 优化框架 | Outcome Signal | Step Signal | 目的 |
|--------|--------|----------|---------------|-------------|------|
| P0 | TTRL baseline | DAPO | naive MV | 无 | 主 baseline |
| P1 | TTRL + SPAE-GT | DAPO | naive MV | GT correctness | 工程上界 |
| P2 | TTRL + SPAE-Pseudo | DAPO | naive MV | Pseudo correctness | 第一版无监督 |
| P3 | TTRL + Confidence | DAPO | naive MV | Probe entropy | 最简 baseline |
| P4 | TTRL + Step SE | DAPO | naive MV | Semantic entropy | 旧思路 baseline |
| P5 | **TTRL + SPC-only** | **DAPO** | naive MV | **SPC (augmentation)** | **主方法** |
| P5b | TTRL + SPC (precision-first) | DAPO | naive MV | SPC (阈值过滤) | 精度优先变体 |
| P6 | DARE + SPC | DAPO | DARE distribution | SPC (augmentation) | 升级 anchor |
| P6b | DBB + SPC | DAPO | DBB Beta posterior | SPC (augmentation) | 零成本升级 anchor |
| P7 | TTRL + SPC + CLIPO | DAPO | naive MV | SPC + contrastive | 跨轨迹扩展 |
| P8 | SHAPE (GT) | DAPO | GT | Hierarchical | 有监督 step-level 对照 |

### 新增的最容易踩坑的地方

#### 坑 6：GRPO 梯度爆炸
如果使用 GRPO 而非 DAPO，hard/easy queries 的梯度信号会被 η(p) 无限放大。症状：训练曲线剧烈震荡、entropy 快速下降、loss 突然爆炸。解决：切换到 DAPO 或检查是否需要 clip η(p)。

#### 坑 7：SPC 信号过强导致 reward 崩溃
如果 SPC 信号直接替换 outcome reward（而非做 augmentation），step-level 噪声可能主导训练。症状：accuracy 不升反降、output 变短或变长。解决：使用 augmentation 方式 + lower_bound，确保 TTRL outcome reward 仍是主信号。

#### 坑 8：SPC 和 outcome reward 的梯度冲突
类似 DCPO 发现的 accuracy-calibration conflict，SPC 信号和 outcome reward 可能在某些 token 上方向相反。症状：训练停滞、两个 loss 交替上下。解决：尝试 DCPO-style gradient masking 或降低 SPC 信号权重。

#### 坑 9：不要使用传统噪声鲁棒方法
[[wiki/papers/yang-2026-olr|OLR]] 明确证明 small-loss selection、confidence learning 等 SFT 领域的经典噪声处理方法在 RLVR 中灾难性失败（-17.4%）。原因：RL 的 loss landscape 与 SFT 根本不同。如果 SPC 信号有噪声，应该用 precision-first threshold 过滤而非 small-loss selection。

#### 坑 10：SHAPE 对比实验需要公平设置
[[wiki/papers/ai-2026-shape|SHAPE]] 需要 GT correctness（有监督），SPC 不需要。对比时需要同时报告：(1) SHAPE with GT vs SPC without GT（公平性对比）；(2) SHAPE with pseudo-label vs SPC（同等设置对比）。不要只报告 SHAPE with GT 然后说 SPC 不如它。

## 基于第二批新论文的实验设计补充修订

> 第二批 2026 年新论文（CoVerRL, SCRL, DistriTTRL, TTVS, OLR, AsymGRPO, DBB, SHAPE, Imperfect Verifier, Self-Guide）进一步完善了 SPC 实验的对照设计、信号设计原则和零成本增强策略。

### 补充修订 6：新增 SHAPE 作为直接对照 Baseline

**依据**：[[wiki/papers/ai-2026-shape|SHAPE]] (arXiv:2604.06636) 提出 segment-level solvability potential + token-level entropy 重分配，是 SPC 最直接的 step-level credit assignment 竞品。

**修订建议**：
- Phase 4 实验矩阵必须包含 SHAPE 作为对照组
- 关键对比维度：SHAPE 需要 GT correctness，SPC 不需要；SHAPE 的 solvability 是启发式，SPC 的 consistency 是经验性测量
- 同时记录 token efficiency 指标（SHAPE 声称 -30% tokens）

**新增实验组**：

| 组别 | 方法 | 需要 GT | 目的 |
|------|------|--------|------|
| G10 | SHAPE (GT-based hierarchical) | 是 | Step-level 有监督上界对照 |
| G11 | SHAPE (pseudo-label variant) | 否 | SHAPE 无监督化对照 |

### 补充修订 7：OLR/Imperfect Verifier 指导 SPC 信号设计原则

**依据**：
- [[wiki/papers/yang-2026-olr|OLR]] 证明传统噪声鲁棒方法（small-loss selection）在 RLVR 中灾难性失败（-17.4%）
- [[wiki/papers/plesner-2026-imperfect-verifier|Imperfect Verifier]] 证明 15% 噪声率内仍鲁棒

**修订建议**：
- SPC 信号设计应遵循 "precision-first" 原则：宁可在不确定步骤不给信号，也不要给错误信号
- 不要套用 SFT 领域的噪声鲁棒技术（如 confidence learning, small-loss trick）到 SPC
- 新增 "SPC signal threshold" 超参扫描实验：只在 SPC_k 高于某阈值（如 0.8）或低于某阈值（如 0.2）时才给 shaping reward

**新增实验组**：

| 组别 | SPC 阈值策略 | 目的 |
|------|-------------|------|
| G12a | SPC 全量信号 (无阈值) | 基础方式 |
| G12b | SPC 阈值过滤 (只给 >0.8 或 <0.2) | Precision-first |
| G12c | SPC 渐进过滤 (早期全量 → 后期收紧阈值) | OLR-inspired progressive refinement |

### 补充修订 8：DBB Beta Posterior 作为零成本 Layer 1 增强

**依据**：[[wiki/papers/kim-2026-dbb|DBB]] 的 Beta-Bernoulli posterior 利用历史统计平滑 reward 估计，零额外计算/内存，OOD +12.49%

**修订建议**：
- DBB 可以直接集成到 TTRL Layer 1 作为零成本增强
- 替代 naive MV 的 binary reward，用 Beta posterior 均值作为连续 reward

**新增实验组**：

| 组别 | Layer 1 Reward | Step Signal | 目的 |
|------|---------------|-------------|------|
| G13 | DBB (Beta posterior) | 无 | DBB-only baseline |
| G14 | DBB (Beta posterior) | SPC (augmentation) | DBB + SPC |

### 补充修订 9：TTVS 数据增广策略

**依据**：[[wiki/papers/bai-2026-ttvs|TTVS]] 通过动态增广测试数据为语义等价变体，扩大 TTRL 的有效训练集

**修订建议**：
- TTVS 的数据增广可以提升 SPC 实验的 robustness
- 在 Phase 4 后期考虑：将 TTVS 的 variational synthesis 应用到 SPC 的 test data
