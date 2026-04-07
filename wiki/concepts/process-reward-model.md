---
title: Process Reward Model (PRM, 过程奖励模型)
type: concept
tags: [RL, reward-model, step-level, verification, URLVR]
created: 2026-04-07
updated: 2026-04-08
sources: [wiki/papers/rahman-2025-spark.md, wiki/papers/ghimire-2026-prism.md, wiki/papers/royer-2026-mcnig.md, wiki/papers/wang-2026-prorag.md]
status: active
---

# Process Reward Model (PRM, 过程奖励模型)

## 定义
> Process Reward Model，过程奖励模型。对推理过程中**每一步**提供 reward 信号的模型，区别于只看最终答案的 ORM（Outcome Reward Model）。PRM 能提供更细粒度的训练信号，帮助模型学习 "如何推理" 而非仅仅 "推理结果是否正确"。

## 关键性质
1. **Step-level 信号**：对每个推理步骤独立评估正确性/质量，提供密集的 reward
2. **比 ORM 信号更丰富**：不仅知道最终答案对不对，还知道哪步出了问题
3. **训练数据获取昂贵**：传统 PRM 需要人工标注每步的正确性（如 PRM800K 数据集）
4. **可合成训练数据**：[[wiki/papers/rahman-2025-spark|SPARK]] 证明用 [[self-consistency|self-consistency]] 生成的合成数据训练的 PRM 可以超过 ground-truth 训练的 PRM

## 直觉理解
> 考试批改的两种方式：ORM 像只看最终答案对不对就给分（答案对 = 满分，答案错 = 0 分）；PRM 像逐步批改（第一步对 ✓，第二步对 ✓，第三步计算错误 ✗...）。PRM 能告诉学生"你到底哪步出了问题"。

## 数学表达

### PRM 基本形式

对问题 q 和推理轨迹 o = (s_1, s_2, ..., s_S)：

$$R_{PRM}(q, o) = f\left(\{R_{step}(q, o, s)\}_{s=1}^{S}\right)$$

其中 $R_{step}(q, o, s) \in [0, 1]$ 是第 s 步的正确性评分，$f$ 是聚合函数。

### 聚合方式

**Min aggregation**（PRISM 使用，更保守）:
$$R_{PRM}(q, o) = \min_{s=1}^{S} R_{step}(q, o, s)$$

**Product aggregation**（SPARK 使用）:
$$R_{PRM}(q, o) = \prod_{s=1}^{S} R_{step}(q, o, s)^{1/S}$$

**Mean aggregation**:
$$R_{PRM}(q, o) = \frac{1}{S}\sum_{s=1}^{S} R_{step}(q, o, s)$$

### PRM 变体（SPARK）

| 变体 | 输入 | 输出 | 效果 |
|------|------|------|------|
| ORM | 完整轨迹 | 最终答案正确性 | 基线 |
| PRM | 逐步轨迹 | 每步 correct/incorrect | 较好 |
| **PRM-CoT** | 逐步轨迹 | 先分析推理再给 label | **最佳** |

## PRM 在 URLVR 中的角色

### 传统方式（需要标注）
人工标注每步正确性 → 训练 PRM → 用 PRM 做 RL reward

### SPARK 方式（无需标注）
1. 大规模采样 + step-level [[self-consistency|self-consistency]] → 生成合成 step-level labels
2. 在合成数据上 fine-tune → 得到 PRM
3. 冻结 PRM → 做 RL reward（stationary signal）

### MCNIG 方式（信息论，无需重新采样）
1. 对每步推理，计算模型对 correct/incorrect 答案集合的 log-probability 差异变化
2. MCNIG_i = NetInfo_i - NetInfo_0，衡量该步对"正确答案优势"的贡献
3. Label = MCNIG > threshold → 训练 PRM
4. **核心优势**: O(N) 复杂度（KV-caching），比 MathShepherd O(N²) 和 OmegaPRM O(NlogN) 高效 7 倍+
5. 详见 [[wiki/papers/royer-2026-mcnig|MCNIG (Royer et al., 2026)]]

### ProRAG 方式（MCTS + Contrastive Labeling）
1. 用 [[mcts|MCTS]] 从 SFT policy 出发探索多样推理路径
2. PUCT selection 平衡 exploration 和 exploitation，Q-value backpropagation with decay
3. 对 MCTS 树中 sibling nodes 用 GPT-4o 做 contrastive labeling（96% 与人工一致）
4. 生成 (good step, bad step) 对比对 → 训练 PRM
5. **特点**: 数据质量最高，但需要 MCTS 搜索和 GPT-4o 标注，计算成本较大
6. 详见 [[wiki/papers/wang-2026-prorag|ProRAG (Wang et al., 2026)]]

### PRISM 方式（混合信号）
- 用现成的 GenPRM-7B（已有的 generative PRM）
- 结合 self-certainty：$\hat{A} = \gamma \cdot \hat{A}_{SC} + \hat{A}_{PRM}$
- PRM 防止 overconfidence，self-certainty 防止格式崩溃

## PRM 训练数据生成方法对比

| 方法 | 策略 | 复杂度 | 需要 GT | 数据质量 | 论文 |
|------|------|--------|--------|---------|------|
| 人工标注 | 专家逐步标注 | — | 是 | 最高 | PRM800K |
| MathShepherd | 每步重新采样 N 次 | O(N²) | 是 | 中等 | Wang 2024 |
| OmegaPRM | MCTS 探索 | O(NlogN) | 是 | 较高 | Luo 2024 |
| **SPARK** | Step-level self-consistency | O(M×N) | 否 | 较高 | [[wiki/papers/rahman-2025-spark\|Rahman 2025]] |
| **MCNIG** | 信息论（log-prob 对比） | O(N) | 是* | 较高 | [[wiki/papers/royer-2026-mcnig\|Royer 2026]] |
| **ProRAG MCTS** | MCTS + GPT-4o labeling | O(NlogN) | 是 | 高（96%） | [[wiki/papers/wang-2026-prorag\|Wang 2026]] |

*MCNIG 需要 ground-truth 来分 correct/incorrect 答案集合，但不需要 step-level 标注。

## PRM 的已知问题

### 1. 训练数据获取成本（传统方式）
人工标注每步正确性极其昂贵。SPARK 的合成数据方案是突破。

### 2. 格式崩溃问题（PRISM 发现）
纯 PRM 做 reward → 模型忘记 boxed 格式 → verifier 无法提取答案。因为 PRM 只关注推理质量，不关注 instruction following。

### 3. Reward Hacking（SPARK 发现）
三种模式：
- **Step inflation**: 增加步骤数稀释错误步骤的权重
- **Step reduction**: 减少步骤数以最小化出错机会
- **Solution appending**: 正确答案后追加冗余内容

## Discriminative vs Generative PRM

| 维度 | Discriminative PRM | Generative PRM |
|------|-------------------|----------------|
| 输出 | 分类概率 (correct/incorrect) | 自由文本评估 + 分类 |
| 代表 | PRM800K (Lightman et al.) | GenPRM, SPARK PRM-CoT |
| 可解释性 | 低（只有分数） | 高（有评估理由） |
| 效果 | 较好 | 更好（CoT 提升判断） |
| 计算成本 | 低 | 高（需生成文本） |

## 相关论文
- Lightman et al., 2023 — "Let's Verify Step by Step"（PRM800K，PRM 经典论文）
- [[wiki/papers/rahman-2025-spark|SPARK (Rahman et al., 2025)]] — 无标注训练 PRM，PRM-CoT 超越 GT RLVR
- [[wiki/papers/ghimire-2026-prism|PRISM (Ghimire et al., 2026)]] — 发现纯 PRM 的格式崩溃问题，提出混合方案
- [[wiki/papers/royer-2026-mcnig|MCNIG (Royer et al., 2026)]] — 信息论自动生成 PRM 训练数据，O(N) 复杂度
- [[wiki/papers/wang-2026-prorag|ProRAG (Wang et al., 2026)]] — MCTS-based PRM 训练，RAG 任务上的 process supervision
- Wang et al., 2024 — "Math-Shepherd"（用自动标注的 step-level 数据训练 PRM）

## 面试常问点

- 🔴 Q: PRM 和 ORM 的区别？各自优缺点？
  - A: ORM 只评估最终答案（sparse signal），PRM 评估每个推理步骤（dense signal）。PRM 信号更丰富但训练数据获取更难。SPARK 证明用合成数据训练的 PRM 可以超过 ground-truth ORM。

- 🔴 Q: 如何在没有人工标注的情况下训练 PRM？
  - A: SPARK 方案：对每道题采样多个解法，用 step-level self-consistency（从每步开始重新求解 N 次，看答案一致性）作为 step-level label，在合成数据上 fine-tune 得到 PRM。效果超过用 ground-truth 标注训练的 PRM。

- 🟡 Q: PRM 做 RL reward 会有什么问题？
  - A: 两个问题：(1) PRISM 发现的格式崩溃——模型忘记 boxed 格式，PRM reward 上升但 accuracy 下降；(2) SPARK 发现的 reward hacking——step inflation/reduction/appending。对策：混合信号（PRISM）或 format constraints（SPARK）。

- 🟡 Q: Min aggregation vs Mean aggregation？
  - A: Min 更保守——只要有一步很差，整体分就低。Mean 会被其他好步骤拉高。在 URLVR 中 min 更抗 reward hacking（PRISM 的选择）。

## 与其他概念的关系
- 上位概念：[[reward-model|Reward Model]]、[[rlhf|RLHF]]
- 对比概念：ORM（Outcome Reward Model）、[[self-consistency|Self-Consistency]]（另一种信号来源）
- 应用：[[grpo|GRPO]]（PRM 做 reward 的 RL 框架）
- 风险：[[reward-hacking|Reward Hacking]]（PRM 被 exploit 的模式）
