---
title: "Dual Consensus Reinforcement Learning for Unsupervised LLM Reasoning"
type: paper
tags: [URLVR, TTRL, majority-voting, dual-consensus, unlearning, spurious-majority, adaptive-sampling, Qwen3, Beihang]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2603.16223]
status: active
---

# DCRL: Dual Consensus 解决 TTRL 虚假多数问题

## 基本信息
- **作者**: Du et al.
- **机构**: Beihang University (北京航空航天大学)
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.16223)
- **链接**: https://arxiv.org/abs/2603.16223
- **Base Model**: Qwen3-8B

## 一句话总结
> 通过两阶段投票机制（anchor + explorer 双视角）解决 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的 spurious majority 问题：克隆 anchor → unlearn 一步得到 explorer → 双组 rollout → harmonic mean 选取 pseudo-label → 三级 reward + 自适应采样，在 Qwen3-8B 上 Avg 50.9 超越 TTRL 的 50.2。

## 摘要
[[wiki/papers/zuo-2025-ttrl|TTRL]] 利用 majority voting 在无标注数据上估计 pseudo-label 进行 RL 训练，但存在一个核心缺陷：**spurious majority（虚假多数）**——当模型对某些问题系统性地偏向错误答案时，majority voting 会持续给出错误的 pseudo-label，导致训练在错误方向上强化。DCRL（Dual Consensus Reinforcement Learning）通过引入一个"explorer"模型（对 anchor 做 unlearning 得到）来提供多样化的第二视角，利用两组模型的 **harmonic mean consensus** 选取更可靠的 pseudo-label，并设计三级 reward 和动态采样策略进一步提升训练效率。

## 核心贡献
1. **识别并解决 Spurious Majority 问题**: 首次明确指出 TTRL 的 majority voting 在模型系统性偏差时会产生虚假多数，并提出 dual consensus 解决方案
2. **Anchor-Explorer 双视角架构**: 通过 unlearning 机制从同一模型分化出两个具有不同 bias 的视角，用于交叉验证 pseudo-label
3. **Harmonic Mean 选择机制**: 比简单投票更稳健的 pseudo-label 选取方式，要求两个视角同时高置信才选取
4. **三级 Reward + 动态采样**: 精细化 reward 信号（1.0/0.5/0.0）并优先采样 anchor 与 explorer 不一致的"困难"样本

## 方法

### 问题定义
[[wiki/papers/zuo-2025-ttrl|TTRL]] 的 majority voting reward 存在固有缺陷：当模型对某类问题有系统性偏差时（比如对某个数学题总是算出同一个错误答案），majority voting 会持续选中这个错误答案作为 pseudo-label，并给予正 reward——形成**正反馈死循环**。随训练进行，这种错误偏差会被进一步放大，即 spurious majority 问题。

### 技术方案

#### 1. Anchor-Explorer 双模型构造

**Step 1 - 克隆 Anchor**: 将当前 policy $\pi_\theta$ 冻结为 anchor 模型 $\pi_0$

**Step 2 - Unlearning 生成 Explorer**: 对 anchor 做一步 unlearning 更新得到 explorer $\pi_1$：

$$\mathcal{L}_{\text{unlearn}} = -\log(1 - p_{\text{clip}})$$

其中 $p_{\text{clip}}$ 是 anchor 模型在训练数据上的 clipped 概率。Unlearning 的目的是让 explorer 在保持基本能力的同时，**打破 anchor 的系统性偏差**，从而提供不同的"思考角度"。

**Unlearning 学习率**: $\eta_{\text{unlearn}} = 3 \times 10^{-7}$（最优值，过大会破坏能力，过小则 explorer 与 anchor 过于相似）

#### 2. Dual Consensus Pseudo-Label 选择

两组模型分别对 query $x$ 进行 rollout 采样：
- Anchor $\pi_0$: 采样 $N$ 条 response，统计答案分布 $p_0(a)$
- Explorer $\pi_1$: 采样 $N$ 条 response，统计答案分布 $p_1(a)$

Pseudo-label 通过 **harmonic mean** 选取：

$$y^* = \arg\max_a \text{harmonic}(p_0(a), p_1(a)) = \arg\max_a \frac{2 \cdot p_0(a) \cdot p_1(a)}{p_0(a) + p_1(a)}$$

**Harmonic mean 的优势**: 只有当两个视角都对某答案有高置信时，harmonic mean 才会高。如果只有 anchor 偏好某答案但 explorer 不认同（即 spurious majority），harmonic mean 会很低，从而避免选中错误 pseudo-label。

#### 3. 三级 Reward 设计

不同于 TTRL 的 binary reward（0 或 1），DCRL 使用三级 reward：

| 条件 | Reward | 含义 |
|------|--------|------|
| Anchor 和 Explorer 都同意 | $r = 1.0$ | 高置信正确 |
| 仅 Anchor 同意 | $r = 0.5$ | 中等置信 |
| 两者都不同意 | $r = 0.0$ | 低置信/错误 |

这种设计比 binary reward 更精细，为模型提供更丰富的梯度信号。

#### 4. Dynamic Sampling（动态采样）

核心思想：**优先采样 anchor 和 explorer 不一致的"困难"query**。

具体机制：
- 计算每个 query 上 anchor 和 explorer 的答案分布差异
- 差异大的 query → 更高的采样概率
- 差异小的 query（两者已达成共识）→ 降低采样概率

动态采样的意义：将计算资源集中在模型最"困惑"的问题上，提高训练效率。

### 关键公式

$$\mathcal{L}_{\text{unlearn}} = -\log(1 - p_{\text{clip}})$$

$$y^* = \arg\max_a \frac{2 \cdot p_0(a) \cdot p_1(a)}{p_0(a) + p_1(a)}$$

$$r = \begin{cases} 1.0, & \text{anchor 和 explorer 都同意} \\ 0.5, & \text{仅 anchor 同意} \\ 0.0, & \text{两者都不同意} \end{cases}$$

### 实现细节
| 参数 | 值 |
|------|-----|
| Base Model | Qwen3-8B |
| Unlearning LR | $3 \times 10^{-7}$ |
| Unlearning Steps | 1 step |
| Rollout 采样数 | $N$ per group |
| Reward 级别 | 三级 (1.0 / 0.5 / 0.0) |
| RL 算法 | GRPO |

## 实验结果

### 主实验

| 方法 | AIME24 | Avg (多benchmark) |
|------|--------|-----|
| GRPO with GT | — | 53.5 |
| TTRL | — | 50.2 |
| **DCRL** | **+15.1% vs TTRL** | **50.9** |

- DCRL 在 Qwen3-8B 上 Avg 50.9，超越 TTRL 的 50.2
- TTA（Test-Time Adaptation）场景下 AIME24 提升 15.1%，说明 DCRL 有效缓解了 spurious majority
- 与 GRPO with GT（53.5）仍有差距，说明 unsupervised 信号仍有改进空间

### 消融实验

| 消融项 | Avg 变化 | 说明 |
|--------|---------|------|
| 移除 Dynamic Sampling | **-2.9** | 最关键组件 |
| 移除三级 Reward → binary | -1.5 | 精细 reward 有价值 |
| 移除 Explorer (单模型 MV) | -2.1 | 回退到 TTRL |
| Unlearn LR=1e-7 (太小) | -0.8 | Explorer 与 anchor 过于相似 |
| Unlearn LR=1e-6 (太大) | -1.2 | Explorer 能力退化 |

**关键发现**:
- **Dynamic Sampling 是最关键组件**（移除导致 -2.9 avg），说明将计算资源集中在困难样本上非常重要
- Unlearn LR = $3 \times 10^{-7}$ 是最优值，需要精确控制 anchor 与 explorer 的差异程度

## 与其他工作的关系

### 与 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的直接改进关系
- DCRL 是 TTRL 的直接改进版本，核心解决 spurious majority 问题
- 保留了 TTRL 的 online RL + pseudo-label 框架，但用 dual consensus 替代 naive majority voting
- TTRL 的 Lucky Hit 机制在错误分散时有效，但当错误系统性集中时失效——DCRL 的 explorer 视角打破这种集中

### 与 [[wiki/papers/wu-2026-self-judge|Self-Judge]] 的比较
- 两者都认识到 majority voting 的局限性，但解决思路不同
- Self-Judge: distributional modeling 保留分布形状信息
- DCRL: 引入第二视角（explorer）进行交叉验证

### 与 [[wiki/papers/zhang-2025-empo|EMPO]] 的比较
- EMPO 用语义聚类替代 majority voting 解决答案等价性问题
- DCRL 从不同角度解决 MV 问题——不是改进 voting 本身，而是引入多视角 consensus

### 与 [[wiki/papers/ghimire-2026-prism|PRISM]] 的关系
- PRISM 发现纯内部信号长期不可靠，DCRL 通过两个不同 bias 的模型交叉验证来提高信号质量
- 两者的核心思想都是"不要依赖单一信号源"

### 与 [[wiki/concepts/grpo|GRPO]] 的关系
- DCRL 使用 GRPO 作为底层 RL 算法，但用三级 reward 替代 binary reward
- 三级 reward 为 GRPO 的 group advantage 估计提供了更精细的信号

## 局限性与开放问题
1. **计算开销翻倍**: 需要维护两组模型（anchor + explorer）并分别 rollout，计算量约为 TTRL 的 2x
2. **Unlearning 超参敏感**: LR 需要精确调到 $3 \times 10^{-7}$，过大过小都影响效果
3. **仍然是 outcome-level**: 和 TTRL 一样只有最终答案层面的 reward，无 step-level credit assignment
4. **Explorer 退化风险**: 长期训练中 explorer 是否会逐渐失去多样性？论文未讨论
5. **与 GT 仍有差距**: Avg 50.9 vs GRPO with GT 53.5，说明 unsupervised pseudo-label 的天花板仍然存在

## 面试相关
> DCRL 涉及 TTRL 的核心改进，面试中可能作为 TTRL follow-up 被追问

- **Q: TTRL 的 Spurious Majority 问题是什么？DCRL 如何解决？** 🔴
- A: 当模型对某些问题有系统性偏差时（如总是算出同一个错误答案），majority voting 会持续选中错误答案作为 pseudo-label，形成正反馈死循环。DCRL 通过 unlearning 构造一个与原模型偏差不同的 explorer，两组模型 rollout 后用 harmonic mean 选取 pseudo-label——只有两个视角都高置信的答案才会被选中，有效过滤 spurious majority。

- **Q: 为什么用 harmonic mean 而不是算术平均或几何平均？** 🟡
- A: Harmonic mean 对低值更敏感——如果两个概率中有一个很低，harmonic mean 也会很低。这意味着只有 anchor 和 explorer 都对某答案高置信时才会被选中。如果只有一方偏好（如 spurious majority 情况下只有 anchor 高、explorer 低），harmonic mean 会有效压制，比算术平均更保守、更鲁棒。

- **Q: Dynamic Sampling 为什么是最关键的组件？** 🟡
- A: 因为训练中大部分"简单"query 两组模型很快达成一致，继续采样这些 query 是计算浪费。Dynamic Sampling 将资源集中在 anchor 和 explorer 不一致的"困难"query 上——这些正是模型最可能犯系统性错误的问题，也是最需要纠正的。移除 Dynamic Sampling 导致 -2.9 avg 下降，比移除 explorer 本身（-2.1）影响还大。

- **Q: Unlearning 在这里的作用是什么？为什么不用另一个独立训练的模型？** 🟢
- A: Unlearning 的目的是**最小代价地打破模型的系统性偏差**，同时保持大部分推理能力。一步 unlearning 比训练独立模型便宜得多，且确保 explorer 与 anchor 的差异是可控的。用独立模型会引入不可预测的差异，可能导致 consensus 信号噪声过大。

- **Q: DCRL 的三级 reward 比 TTRL 的 binary reward 好在哪里？** 🟢
- A: Binary reward（0/1）对所有正确答案等同对待，对所有错误答案也等同对待。三级 reward（1.0/0.5/0.0）区分了"双方都确认正确"和"仅一方确认正确"，为模型提供更精细的梯度信号。消融实验显示移除三级 reward → binary 会导致 -1.5 avg 下降。

## 个人笔记

### 与 SPC 研究方案的关系
DCRL 直接解决了 TTRL majority voting 的核心缺陷（spurious majority），与 SPC 研究方案高度相关：

1. **SPC 的 outcome-level anchor 可以用 DCRL 替代 naive MV**: 当前 SPC 方案在 outcome level 依赖 TTRL 的 majority voting，但这会受 spurious majority 影响。DCRL 的 dual consensus 提供了更可靠的 outcome anchor。

2. **"双视角"概念的启发**: DCRL 的 anchor + explorer 双视角与 SPC 的"多 rollout 视角一致性"理念有异曲同工之处——都是通过引入多样性来提高信号质量。SPC 可以考虑在 step-level 也引入类似的 dual perspective。

3. **Dynamic Sampling 可借鉴**: SPC 方案也面临计算资源分配问题。DCRL 的 dynamic sampling 思路（优先处理困难样本）可以直接应用于 SPC 的训练过程。

### 关键启示
- Spurious majority 是 TTRL 系统性弱点，任何基于 majority voting 的方法都需要考虑
- 引入多样化视角（通过 unlearning）是低成本打破偏差的有效方式
- Harmonic mean 作为 consensus 机制比简单投票更鲁棒
- Dynamic Sampling 的效果比改进 reward 本身更显著——"在哪里训练"可能比"怎么训练"更重要
