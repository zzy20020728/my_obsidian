---
title: Semantic Entropy (语义熵)
type: concept
tags: [信息论, 不确定性, URLVR, reward-signal]
created: 2026-04-07
updated: 2026-04-07
sources: [wiki/papers/zhang-2025-empo.md]
status: active
---

# Semantic Entropy (语义熵)

## 定义
> Semantic Entropy，语义熵。衡量模型输出在**语义层面**的不确定性。不同于标准 token-level entropy（衡量每个 token 的预测不确定性），semantic entropy 将语义等价的输出归为同一类，衡量的是"答案含义"的不确定性。在 URLVR 中被 [[wiki/papers/zhang-2025-empo|EMPO]] 用作无监督 reward 信号。

## 关键性质
1. **语义级别而非 token 级别**：将表述不同但含义相同的输出视为同一类
2. **与准确率强负相关**：实验验证，semantic entropy 越低，模型准确率越高
3. **可作为无监督 reward**：最小化 semantic entropy ≈ 鼓励模型收敛到一致（大概率正确）的答案
4. **依赖语义聚类质量**：计算准确性取决于能否正确判断两个输出是否语义等价

## 直觉理解
> 你问一个人同一个问题 10 次。如果他 8 次给出相同含义的答案，2 次给出不同的——semantic entropy 低，说明他很确定。如果 10 次答案的含义各不相同——semantic entropy 高，说明他完全不确定。关键是看"含义"而非"用词"。

## 数学表达

### 计算步骤

**Step 1**: 对问题 q 采样 G 个输出 {o_1, ..., o_G}

**Step 2**: 按语义等价性聚类为 meaning clusters {c_1, ..., c_K}
- 数学任务：正则表达式提取最终答案，精确匹配
- 自由形式：用 SLM 判断双向蕴含（bidirectional entailment）

**Step 3**: 计算每个聚类的概率
$$p(c_j|q) \approx \frac{|c_j|}{G}$$

**Step 4**: 计算语义熵
$$SE(q) = -\sum_{j=1}^{K} p(c_j|q) \log p(c_j|q)$$

### 作为 Reward

在 [[wiki/papers/zhang-2025-empo|EMPO]] 中，每条输出的 reward 等于其所属聚类的频率：
$$r_i = p(c_j|q) = \frac{|c_j|}{G}$$

最大化这个 reward 等价于最小化 semantic entropy。

### Entropy Thresholding

EMPO 引入双阈值过滤：
$$\delta_{low} < SE(q) < \delta_{high}$$

- $SE < \delta_{low}$：模型已很确定，继续训练边际收益小
- $SE > \delta_{high}$：模型完全随机，reward 信号太 noisy

## 与 Token-Level Entropy 的区别

| 维度 | Token-Level Entropy | Semantic Entropy |
|------|-------------------|-----------------|
| 粒度 | 每个 token | 完整输出的含义 |
| 等价性 | 相同 token 才等价 | 语义相同即等价 |
| "我有5元" vs "5块钱" | 不同 | 相同 |
| 计算成本 | 低（单次前向传播） | 高（多次采样+聚类） |
| URLVR 可靠性 | 容易 reward hack | 相对更鲁棒 |

## 相关论文
- [[wiki/papers/zhang-2025-empo|EMPO (Zhang et al., 2025)]] — 用 semantic entropy 做 URLVR 的 reward 信号
- [[wiki/papers/ghimire-2026-prism|PRISM (Ghimire et al., 2026)]] — 证明 entropy 类信号长期训练不可靠
- Kuhn et al., 2023 — "Semantic Uncertainty" 原始论文，首次定义 semantic entropy

## 面试常问点

- 🔴 Q: Semantic entropy 和普通 entropy 有什么区别？
  - A: 普通 entropy 在 token 或 sequence level 计算，相同含义不同表述的输出被视为不同。Semantic entropy 先做语义聚类，再在聚类上计算 entropy，衡量的是"答案含义"的不确定性。

- 🟡 Q: 如何实现语义等价性判断？
  - A: 数学任务用正则提取答案 + 精确匹配（简单可靠）；自由形式任务用 NLI 模型或 SLM 判断双向蕴含（如 EMPO 训练了 1.5B 的 General-Verifier）。

- 🔴 Q: Semantic entropy 做 reward 的局限性？
  - A: PRISM 实验表明，纯 entropy 信号长期训练会 reward hack——模型学会降低 entropy 的方式不一定是给出正确答案（如重复生成）。需要配合外部信号使用。

## 与其他概念的关系
- 上位概念：[[information-theory|信息论]]、[[uncertainty-estimation|不确定性估计]]
- 相关概念：[[self-consistency|Self-Consistency]]（也用采样一致性衡量不确定性）
- 应用概念：[[reward-hacking|Reward Hacking]]（semantic entropy 做 reward 的风险）
- 方法关联：[[grpo|GRPO]]（EMPO 基于 GRPO 框架）
