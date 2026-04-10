---
title: "DARE: Distribution-Aware Reward Estimation for TTRL"
type: paper
tags: [URLVR, TTRL, distribution-aware, reward-estimation, uncertainty, exploration-bonus, information-collapse, ICML]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2601.21804]
status: active
---

# DARE: 分布感知的 TTRL Reward 估计

## 基本信息
- **作者**: Du et al.
- **机构**: 未明确标注
- **年份**: 2026
- **会议/期刊**: ICML 2026 (arXiv:2601.21804)
- **链接**: https://arxiv.org/abs/2601.21804
- **Base Model**: Qwen2.5-Math-1.5B

## 一句话总结
> 用完整 rollout 分布替代 TTRL 的 point-level majority voting，通过 uncertainty-aware 经验分布 + 探索 bonus + 分布修剪实现更可靠的 reward 估计，AIME24 相对 TTRL +25.3%。

## 摘要

[[wiki/papers/zuo-2025-ttrl|TTRL]] 使用 majority voting 选择 pseudo-label 作为 reward，但这种 point-level estimation 存在**信息坍塌**（information collapse）——将丰富的 rollout 分布压缩为单一投票结果，丢失了 uncertainty 信息。DARE 提出 **Distribution-Aware Reward Estimation**：(1) 用 uncertainty-normalized 经验分布替代 hard voting；(2) 为 under-sampled 答案添加探索 bonus；(3) 修剪明显噪声的分布尾部。理论分析证明 majority voting 在 Theorem 2.1 下存在 information collapse。

## 核心贡献

1. **Theorem 2.1: Information Collapse 理论**：从信息论角度证明 majority voting 丢失了 rollout 分布中的关键信息——多个答案的相对频率和 uncertainty 被压缩为"赢者通吃"
2. **Uncertainty-Aware Distribution**：$\hat{p}(\hat{y}) = n(\hat{y})/(u(\hat{y})+\varepsilon)$，用 uncertainty 归一化频率，低 uncertainty 的答案获得更高权重
3. **Exploration Bonus**：$b(y_i) = (1-n/M)(1-u)$，鼓励对 under-sampled 但 low-uncertainty 的答案给予额外关注
4. **Distribution Pruning**：移除概率过低的噪声答案，提高 reward signal 质量
5. **显著提升**：Qwen2.5-Math-1.5B AIME24 +25.3% relative over TTRL

## 方法

### 问题定义：为什么 Majority Voting 不够好？

在 TTRL 中，对一个 prompt 采样 N 个 rollout，每个给出答案 $a_i$。Majority voting 选择出现次数最多的答案作为 pseudo-label：

$$\tilde{y}^* = \arg\max_a \sum_{i=1}^N \mathbb{1}[a_i = a]$$

**问题**：
1. 如果答案 A 出现 6 次、答案 B 出现 4 次，MV 选 A、给 A 投票者 reward=1。但真实情况可能是 B 才是正确答案（40% 也不低）
2. 不同答案的 uncertainty 不同——高 uncertainty 的"多数"不如低 uncertainty 的"少数"可靠
3. 只看频率最高的答案，忽略了整个分布的形状

**Theorem 2.1 (Information Collapse)**：Majority voting 的 reward function 是 rollout distribution 的充分统计量的一个退化投影——它只保留了 argmax 信息，丢失了分布的形状、spread 和 uncertainty。

### 技术方案

#### Step 1: Uncertainty-Aware 经验分布

$$\hat{p}(\hat{y}) = \frac{n(\hat{y}) / (u(\hat{y}) + \varepsilon)}{\sum_{\hat{y}'} n(\hat{y}') / (u(\hat{y}') + \varepsilon)}$$

- $n(\hat{y})$：答案 $\hat{y}$ 的出现频率
- $u(\hat{y})$：答案 $\hat{y}$ 对应 rollouts 的平均 uncertainty
- $\varepsilon$：平滑项防止除零

**直觉**：frequency 高 + uncertainty 低 → 高分布权重（可靠的多数）；frequency 高但 uncertainty 也高 → 权重被打折（不确定的多数）。

#### Step 2: Exploration Bonus

$$b(y_i) = (1 - n/M) \cdot (1 - u)$$

- $n/M$：该答案的频率归一化
- $(1-u)$：低 uncertainty bonus

**直觉**：frequency 低但 uncertainty 也低的答案获得最高 exploration bonus——这些是"少数但自信"的答案，可能包含 majority voting 遗漏的正确答案。

#### Step 3: Distribution Pruning

移除 $\hat{p}(\hat{y}) < \delta$ 的答案，减少噪声干扰。

#### Step 4: Final Reward

$$r(y_i) = \tilde{p}(\hat{y}_i) + \alpha \cdot \tilde{b}(y_i)$$

其中 $\tilde{p}$ 和 $\tilde{b}$ 是 pruning 后归一化的分布和 bonus。

### 关键公式

#### Theorem 2.1 (Information Collapse)

Majority voting reward $R_{MV}(y) = \mathbb{1}[a(y) = \tilde{y}^*]$ 仅保留了 $\arg\max$ 信息，丢失了：
- 分布的相对频率（30% vs 70% 和 49% vs 51% 被同等对待）
- Uncertainty 信息（高置信 70% 和低置信 70% 被同等对待）
- 少数但可能正确的答案的信息

### 实现细节

| 参数 | 值 |
|------|-----|
| Base Model | Qwen2.5-Math-1.5B |
| Rollout 数 N | 任务相关 |
| Pruning 阈值 δ | 任务相关 |
| Exploration bonus 系数 α | 任务相关 |
| ε (smoothing) | 1e-8 |
| 优化框架 | GRPO |

## 实验结果

### 主实验: Qwen2.5-Math-1.5B

| 方法 | AIME24 | AMC | Avg |
|------|--------|-----|-----|
| TTRL | baseline | baseline | 41.5 |
| **DARE** | **+25.3% relative** | **+5.3%** | **44.2** |

**关键数据**：
- **AIME24**: +25.3% relative improvement over TTRL
- **AMC**: +5.3%
- **Average**: 44.2 vs TTRL 41.5（+2.7 absolute）

### 消融实验

| 组件 | 移除后的影响 |
|------|-------------|
| Uncertainty normalization | 较大下降（核心组件） |
| Exploration bonus | 中等下降 |
| Distribution pruning | 轻微下降 |

Uncertainty normalization 是最关键的组件——区分"可靠多数"和"不确定多数"是 DARE 的核心价值。

### 与其他 TTRL 改进方法对比

DARE 专注于 reward estimation 层面的改进，与 [[wiki/papers/du-2026-dual-consensus|DCRL]]（改进 voting 机制）、[[wiki/papers/liao-2026-t3rl|T³RL]]（引入外部工具验证）等方法正交——理论上可以组合使用。

## 与其他工作的关系

- **直接改进**: [[wiki/papers/zuo-2025-ttrl|TTRL]] 的 reward estimation
- **理论呼应**: [[wiki/papers/he-2026-urlvr-scale|He et al.]] 的 sharpening theorem — DARE 虽然改善了 reward quality，但本质仍是 intrinsic signal，受 sharpening 限制。但 DARE 的 uncertainty-aware 设计减缓了 sharpening 速度
- **方法正交**: 与 [[wiki/papers/du-2026-dual-consensus|DCRL]]（dual consensus voting）、[[wiki/papers/liao-2026-t3rl|T³RL]]（tool verification）在不同维度改进 TTRL
- **类似思想**: [[wiki/papers/liu-2025-ettrl|ETTRL]] 也利用 entropy 信息改进 TTRL，但 ETTRL 在 rollout generation 层面，DARE 在 reward estimation 层面

## 局限性与开放问题

1. **仍是 Intrinsic Signal**：虽然比 naive MV 好，但 DARE 的 reward 仍基于模型自身的 rollout 分布，受 [[wiki/papers/he-2026-urlvr-scale|He et al. sharpening theorem]] 限制
2. **Uncertainty 估计的准确性**：使用什么度量作为 $u(\hat{y})$？token-level entropy？sequence-level perplexity？选择不同度量可能效果差异大
3. **仅 1.5B 模型验证**：更大模型的分布特征可能不同，DARE 的改善幅度未知
4. **Exploration bonus 可能引入噪声**：鼓励 under-sampled 答案可能也会奖励纯粹的随机噪声
5. **Pruning 阈值需要调参**：太严格会丢失有价值的少数答案，太宽松保留噪声

## 面试相关
> 这篇论文可能被问到的面试问题

- **Q: TTRL 的 majority voting reward 有什么理论缺陷？**
- A: 存在 information collapse (Theorem 2.1)——将完整的 rollout 分布压缩为 argmax 一个点，丢失了分布形状、uncertainty 和少数答案的信息。30% vs 70% 和 49% vs 51% 在 MV 下被同等对待，但后者的 reward 信号应该弱得多。

- **Q: DARE 如何利用 uncertainty 改进 reward？**
- A: 用 uncertainty-normalized distribution 替代 hard voting——频率高且 uncertainty 低的答案权重最大（"自信的多数"），频率高但 uncertainty 高的答案被打折（"不确定的多数"）。额外加 exploration bonus 关注"低频但自信"的答案。

- **Q: DARE 与 [[wiki/papers/du-2026-dual-consensus|DCRL]] 都改进 TTRL 的 reward，区别是什么？**
- A: DCRL 改进 voting 机制本身（通过 dual consensus 引入两个视角），DARE 改进 reward 的计算方式（从 point estimation 升级到 distribution estimation）。二者正交，理论上可以组合。

- **Q: Exploration bonus 会不会鼓励错误答案？**
- A: 有这个风险，但 bonus 公式中的 $(1-u)$ 项保证只有低 uncertainty 的少数答案获得高 bonus。高 uncertainty 的随机噪声答案因为 $u$ 大，bonus 很小。配合 distribution pruning 进一步过滤噪声。

## 个人笔记

### 与 SPC 研究方案的关系

DARE 对 [[wiki/synthesis/step-level-se-proposal|SPC 方案]] 的三层架构有直接影响：

1. **Layer 1 升级**：SPC 的三层架构（TTRL anchor → SPC signal → Co-Evolving PRM）中，Layer 1 使用 TTRL 的 majority voting 作为 outcome-level anchor。DARE 可以直接替换 naive MV，提供更可靠的 outcome anchor。这意味着 SPC 的 "starting point" 质量就更高
2. **Information preservation 哲学一致**：DARE 的 Theorem 2.1 (information collapse) 说明 outcome-level 的 point estimation 会丢失信息。SPC 在 step-level 做 semantic rollout consistency 检查，本质上也是在保留 outcome MV 丢失的 process-level 信息——二者在信息论层面目标一致
3. **Uncertainty 信号可复用**：DARE 计算的 uncertainty $u(\hat{y})$ 可以直接作为 SPC 的辅助信号——高 uncertainty 的 pseudo-label 下，SPC 的 step-level 信号应该被赋予更大权重（因为 outcome anchor 不太可靠时，process-level 信号更重要）
4. **Exploration bonus 启示**：DARE 的 exploration bonus 鼓励"少数但自信"的答案，SPC 也应该对 majority 不一致但 step-level consistency 高的轨迹给予 bonus
