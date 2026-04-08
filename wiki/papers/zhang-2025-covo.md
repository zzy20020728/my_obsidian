---
title: "CoVo"
type: paper
tags: [URLVR, CoVo, consistency, volatility, self-improving-reasoning, process-signal, trajectory-evaluation, Zhejiang-University, Alibaba, NTU]
created: 2026-04-08
updated: 2026-04-08
sources: [https://arxiv.org/abs/2506.08745]
status: active
---

# CoVo

## 基本信息
- **作者**: Zhang et al.
- **机构**: Zhejiang University, Alibaba Group, Nanyang Technological University
- **年份**: 2025
- **会议/期刊**: arXiv preprint (arXiv:2506.08745)
- **链接**: https://arxiv.org/abs/2506.08745

## 一句话总结
> 提出用中间状态对最终答案的 **Consistency** 和 **Volatility** 评估整条推理轨迹，构造比 majority voting 更细粒度的无监督 reward，在数学与通用推理上超过 [[wiki/papers/zuo-2025-ttrl|TTRL]] / [[wiki/papers/zhang-2025-empo|EMPO]]，并在部分设置下逼近甚至超过 supervised RL（Section 4）。

## 摘要
CoVo 关注 URLVR 中一个核心问题：纯 outcome-level 的 self-reward 往往只能看见“最终答案是否一致”，却看不见中间推理是否真的稳定支持这个答案。论文提出一个过程感知的无监督信号：如果一条轨迹是正确的，那么它的中间状态通常会持续指向自己的最终答案，且一旦偏离，偏离往往较早暴露；错误轨迹则相反。为此，CoVo 基于中间状态到候选答案的 likelihood distance 构造 **Consistency** 与 **Volatility** 两个指标，再按答案分组做向量聚合，得到 trajectory/group-level reward，并结合 curiosity bonus 训练推理模型。

## 核心贡献
1. **Consistency + Volatility 框架**: 首次系统性把“中间状态是否持续支持最终答案”与“偏离暴露得有多晚”同时纳入无监督 reward 设计
2. **Distance Matrix 建模**: 对每个推理 prefix 与多个候选最终答案之间计算 likelihood distance，显式刻画 state-answer 关系
3. **Vectorial Aggregation Reward**: 不只对单条轨迹打分，而是先按最终答案分组，再用 consistency/volatility 向量聚合得到更鲁棒的 group reward
4. **理论分析**: Proposition 1 形式化说明 majority-voting self-reward 会走向 model collapse；Proposition 2 从变分优化视角解释 self-rewarding RL
5. **强实证结果**: 在 7 个 benchmark、多个 3B/7B instruct 模型上稳定优于 TTRL/EMPO，并匹敌甚至超过 supervised RL baseline

## 方法详解

### 1. 问题设定
给定问题 $x$，模型采样得到多条推理轨迹 $\tau_i$，每条轨迹对应一个最终答案 $y_i$。目标是在无 GT 的情况下，为每条轨迹构造更可靠的 reward。

### 2. Distance Matrix

对一条轨迹 $\tau$ 的每个中间状态 $s_i$，以及该问题上的候选最终答案 $y_k$，定义 distance：

$$
d(s_i, y_k) = -\frac{1}{|y_k|} \sum_j \log \pi_\theta(y_k[j] \mid s_i, y_k[:j])
$$

于是得到一个 distance matrix：

$$
D \in \mathbb{R}^{T \times K}
$$

其中 $T$ 是轨迹长度，$K$ 是候选最终答案个数。每一行表示“当前 prefix 对各候选答案的支持程度”。距离越小，说明当前状态越偏好该答案。

### 3. Consistency

对一条轨迹，记其自身最终答案为第 0 个候选答案，则 consistency 定义为：

$$
Con(\tau) = \frac{1}{T} \sum_{i=0}^{T-1} \mathbb{I}(D[i,0] = \min_k D[i,k])
$$

解释：中间状态中，有多少比例把“自己的最终答案”看作最接近的答案。

经验现象：

- 正确轨迹通常有更高的 consistency
- 错误轨迹通常 consistency 很低

论文报告中，正确与错误轨迹在该指标上差异非常显著。

### 4. Volatility

CoVo 进一步观察到，只看平均一致性还不够。很多错误轨迹并不是一直不一致，而是直到很晚才显露偏离。因此定义：

$$
Vol(\tau) = \frac{1}{T} \max \{ i \mid D[i,0] \neq \min_k D[i,k] \}
$$

解释：最后一次“当前状态不再支持自己最终答案”发生得有多晚。

- 正确轨迹：偏离若出现，往往更早消失，volatility 更低
- 错误轨迹：常常到很后面还在摇摆，volatility 更高

### 5. Vectorial Aggregation

CoVo 没有直接把 `Con - Vol` 作为 reward，而是把每条轨迹投射成二维向量：

$$
v_i = Con(\tau_i) \cdot [\cos(Vol(\tau_i)), \sin(Vol(\tau_i))]
$$

然后按最终答案分组，对同一答案组内的轨迹向量进行聚合，再用组向量范数构造 reward。这个设计兼顾了：

1. 组内一致答案的强化
2. 对晚期波动轨迹的惩罚
3. 对单条 noisy trajectory 的平滑

### 6. Curiosity Bonus

除主 reward 外，CoVo 还加入基于 token probability distribution 的 curiosity bonus，以鼓励探索，并配合 KL penalty 抑制策略过快坍缩。

## 关键理论结果

### Proposition 1: Majority Voting 的 Collapse 风险

论文形式化说明，在 majority-voting self-reward 下，若某个答案在当前模型分布下占优，那么 RL 更新会不断放大这个答案的概率，最终使：

$$
\pi_\theta(y^* \mid x) \to 1
$$

如果这个 $y^*$ 不是 GT，就会出现典型的 reward hacking / model collapse。这个结论与 [[wiki/papers/he-2026-urlvr-scale|He et al. 2026]] 的 sharpening 分析高度一致。

### Proposition 2: Variational Optimization 视角

CoVo 进一步把 self-rewarding RL 写成对 latent reasoning states 的变分优化问题，说明中间状态建模并不是启发式 trick，而是能从优化目标上得到解释。

## 实验结果

### 实验设置
- **模型**: Llama3.2-3B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct 等
- **任务**: MATH, GSM8K, AMC, Olympiad, MMLU-Pro, GPQA, CommonsenseQA
- **训练**: 无监督 RL，和 TTRL / EMPO / supervised RL 等对比

### 主要发现
1. **稳定优于 TTRL / EMPO**：说明只看 majority 或语义熵不够，中间状态与最终答案的一致性更有信息量
2. **接近甚至超过 supervised RL**：在若干 benchmark 上，CoVo 的性能和有 GT 的 RL 基线非常接近，部分场景还能更好
3. **正确/错误轨迹可分性强**：consistency 与 volatility 两个指标在正确和错误轨迹之间形成明显分离
4. **过程信息确实有增益**：不仅最终答案分组重要，中间状态演化轨迹本身就是有价值的 reward 线索

## 与其他工作的关系

- [[wiki/papers/zuo-2025-ttrl|TTRL]]: TTRL 只看最终答案是否与 majority 一致；CoVo 进一步看中间状态是否持续支持该答案
- [[wiki/papers/zhang-2025-empo|EMPO]]: EMPO 用 outcome-level semantic entropy 最小化；CoVo 改为 trajectory-level consistency/volatility 结构化建模
- [[wiki/papers/ghimire-2026-prism|PRISM]]: PRISM 证明纯内部 certainty signals 长期不稳；CoVo 是更强的 internal signal，但本质仍属于内部信号路线
- [[wiki/papers/he-2026-urlvr-scale|He et al. 2026]]: CoVo 的 distance 仍由模型 likelihood 给出，因此从统一理论看，依然可能落入 certainty/sharpening 范式
- [[wiki/papers/wu-2026-spae|SPAE]]: SPAE 提供了 step-level probing 框架；CoVo 提供了 state-to-answer consistency 的核心 insight，两者结合自然导向 [[wiki/synthesis/step-level-se-proposal|SPC 方案]]

## 局限性
1. **仍然基于 likelihood**: consistency 的底层判断是“state 对答案的 log-prob”，而不是实际 rollout 行为，因此可能继承过度自信问题
2. **长期稳定性未充分验证**: 论文训练步数有限，尚未像 PRISM / He et al. 那样系统验证 >1000 步下是否仍稳定
3. **主要在 instruct 模型验证**: 对更强 reasoning-specialized 模型或 base model 的适用性仍待看
4. **计算成本不低**: 每个中间状态都要对多个 candidate answer 计算 distance，状态数和答案数变大时成本上升明显

## 对当前研究方向的启发

CoVo 最大的价值，不是它已经彻底解决了 URLVR，而是它指出了一个非常关键的方向：

> 好的无监督 reward 不应该只看最终答案是否一致，而应该看中间推理过程是否持续支持这个答案。

这正是当前 [[wiki/synthesis/step-level-se-proposal|SPC 方案]] 的直接出发点。区别在于：

- CoVo 用 **likelihood consistency**
- SPC 改成 **semantic rollout consistency**

也就是从“模型判断哪个答案更像对的”升级到“模型从当前状态真正会生成什么答案”。

## 面试相关

- Q: CoVo 相比 TTRL 的核心改进是什么？
- A: TTRL 只看最终答案是否和 majority 一致，CoVo 进一步检查中间状态是否持续支持自己的最终答案，并引入 volatility 惩罚晚期偏离，因此 reward 更过程感知。

- Q: CoVo 为什么仍可能不彻底稳定？
- A: 因为它的 consistency 底层仍由 log-prob / likelihood distance 决定，本质还是内部 certainty-based 信号，只是比单纯 majority voting 更细。

- Q: CoVo 对你自己的研究有什么启发？
- A: 它说明“中间状态和最终答案的一致性”是有价值的，但 likelihood 判断还不够。我会进一步改成 semantic rollout consistency，也就是看模型实际续写会导向什么答案。

## 个人笔记

> *最值得吸收的不是向量聚合技巧，而是 consistency/volatility 这两个过程视角。后续方案应保留其 insight，但把判断空间从 log-prob 升级到语义行为。*
