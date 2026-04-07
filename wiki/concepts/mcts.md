---
title: Monte Carlo Tree Search (MCTS, 蒙特卡洛树搜索)
type: concept
tags: [搜索算法, 规划, RL, PRM, tree-search]
created: 2026-04-08
updated: 2026-04-08
sources: [wiki/papers/wang-2026-prorag.md]
status: active
---

# Monte Carlo Tree Search (MCTS, 蒙特卡洛树搜索)

## 定义
> Monte Carlo Tree Search，蒙特卡洛树搜索。一种通过随机模拟 (rollout) 在搜索树中平衡 exploration 和 exploitation 的决策算法。在 LLM 推理中，MCTS 可以探索多样的推理路径，通过 Q-value backpropagation 评估每个推理步骤的质量。在 URLVR 中，[[wiki/papers/wang-2026-prorag|ProRAG]] 用 MCTS 生成多样推理轨迹并为 [[process-reward-model|PRM]] 训练提供 contrastive labeling 数据。

## 关键性质
1. **探索-利用平衡**：PUCT 选择策略平衡已知好的节点（exploit）和未探索的节点（explore）
2. **Q-value Backpropagation**：终端节点的 reward 反向传播更新路径上所有节点的 Q 值
3. **Contrastive Labeling**：同一父节点的不同子节点（siblings）天然构成对比对，用于训练 PRM
4. **计算密集**：需要大量 rollouts 构建搜索树，但产出的数据质量高

## 直觉理解
> 下围棋时，你不可能穷举所有走法。MCTS 的策略是：(1) 选一个看起来有希望的走法（exploitation）；(2) 偶尔试试没走过的路（exploration）；(3) 把每种走法"随机下完"看看谁赢了（simulation）；(4) 记住每条路的胜率（backpropagation）。反复做这四步，就能在有限时间内找到好走法。在 LLM 推理中，"走法"变成了推理步骤，"下完"变成了继续推理到最终答案。

## 数学表达

### PUCT Selection（ProRAG 使用）

选择子节点的策略：

$$a^* = \arg\max_a \left[ Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$

其中：
- $Q(s, a)$ = 节点的 Q 值（历史平均 reward）
- $P(s, a)$ = policy prior（SFT 模型给出的初始概率）
- $N(s)$ = 父节点访问次数
- $N(s, a)$ = 子节点访问次数
- $c_{puct}$ = 探索系数

### Q-value Backpropagation（with decay）

$$Q(s_t) \leftarrow \text{reward} \cdot \gamma^{T - t}$$

$\gamma^{T-t}$ decay factor 确保靠前步骤的 Q 值反映其对最终结果的贡献（距离终端越远，折扣越大）。

### MCTS 四阶段循环

1. **Selection**：从根节点用 PUCT 选择到叶节点
2. **Expansion**：在叶节点扩展新的推理步骤
3. **Simulation/Rollout**：继续推理到终端（得到最终答案和 reward）
4. **Backpropagation**：将 reward 反向传播更新路径上所有 Q 值

## MCTS 在 URLVR/LLM 中的应用

### ProRAG 中的应用
1. 从 SFT policy 出发构建搜索树
2. 每个节点 = 一个推理步骤（包含 subquery + retrieval + subanswer）
3. 用 GPT-4o 对 sibling nodes 做 contrastive labeling（96% 与人工一致）
4. 生成 (good step, bad step) 对比对 → 训练 PRM

### OmegaPRM 中的应用
- 用 MCTS 探索推理路径，自动生成 step-level labels
- 复杂度 O(NlogN)，比 MathShepherd O(N²) 好但不如 [[wiki/papers/royer-2026-mcnig|MCNIG]] O(N)

### AlphaProof / AlphaGeometry 中的应用
- 数学证明搜索，MCTS + LLM policy = 探索证明空间

## MCTS vs 其他 PRM 数据生成方法

| 方法 | 策略 | 复杂度 | 数据质量 |
|------|------|--------|---------|
| MathShepherd | 每步重新采样 N 次 | O(N²) | 中等 |
| OmegaPRM | MCTS 探索 | O(NlogN) | 较高 |
| [[wiki/papers/royer-2026-mcnig\|MCNIG]] | 信息论度量 | O(N) | 较高 |
| [[wiki/papers/wang-2026-prorag\|ProRAG]] MCTS | MCTS + GPT-4o labeling | O(NlogN) | 高（96%一致性） |

## 相关论文
- [[wiki/papers/wang-2026-prorag|ProRAG (Wang et al., 2026)]] — MCTS-based PRM 训练，RAG 任务
- [[wiki/papers/royer-2026-mcnig|MCNIG (Royer et al., 2026)]] — 信息论替代方案，比 MCTS 更高效
- Silver et al., 2016 — AlphaGo（MCTS + deep RL 的经典之作）
- Luo et al., 2024 — OmegaPRM（MCTS 生成 PRM 训练数据）

## 面试常问点

- 🔴 Q: MCTS 在 LLM 推理中怎么用？
  - A: 把推理过程建模为搜索树，每个节点是一个推理步骤。用 PUCT 选择扩展哪个节点，rollout 到终端得到 reward，backpropagation 更新 Q 值。ProRAG 用 MCTS 探索 RAG 推理路径，生成 sibling nodes 做 contrastive labeling 训练 PRM。

- 🟡 Q: MCTS 生成的 PRM 数据和 self-consistency 方法（SPARK）的区别？
  - A: MCTS 通过 PUCT selection 有策略地探索搜索空间，sibling nodes 天然形成对比对，数据多样性更高。Self-consistency 是从每步重新采样 N 次看答案一致性，偏向随机探索。MCTS 数据质量更高但计算成本也更大。

- 🟢 Q: MCTS 的 PUCT 公式中各项含义？
  - A: Q(s,a) 是 exploitation（选历史 reward 高的节点），P(s,a)·√N(s)/(1+N(s,a)) 是 exploration（选访问次数少的节点），c_puct 控制两者的平衡。

## 与其他概念的关系
- 上位概念：搜索算法、规划 (Planning)
- 应用场景：[[process-reward-model|PRM]] 训练数据生成
- 对比方法：Beam Search（确定性搜索，不含随机 rollout）
- 相关算法：UCB (Upper Confidence Bound)、Alpha-Beta Pruning
