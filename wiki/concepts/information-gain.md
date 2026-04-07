---
title: Information Gain (信息增益)
type: concept
tags: [信息论, PRM, step-level, verification, URLVR]
created: 2026-04-08
updated: 2026-04-08
sources: [wiki/papers/royer-2026-mcnig.md]
status: active
---

# Information Gain (信息增益)

## 定义
> Information Gain (IG)，信息增益。衡量接收到新信息后，模型对某个事件的确信度变化量。在 URLVR 中，[[wiki/papers/royer-2026-mcnig|MCNIG]] 将其应用于 [[process-reward-model|PRM]] 训练数据的自动生成：衡量每步推理后，模型对正确/错误答案的 log-probability 变化，以此判断该步推理的质量。

## 关键性质
1. **衡量推理步骤的"贡献度"**：一步好的推理应该增加模型对正确答案的确信度
2. **基于 log-probability 变化**：不需要重新采样续写，只需计算 log-prob 差异
3. **单一答案 IG 在长答案任务上不可靠**：代码、SQL 等答案长且多样的任务，单一答案的 IG 容易被 subtle errors 误导
4. **Net Information Gain (MCNIG) 更鲁棒**：对比多个 correct 和 incorrect 答案的 IG 差异

## 直觉理解
> 你在解数学题，写了三步推理。第一步画了辅助线——此后你更确定答案是 42，也更不确定答案是 37。这就是正向的 Information Gain。如果某一步写了无关内容——对答案的确信度没变，IG ≈ 0。如果某步推理出错——你对错误答案更确信了，IG 为负。

## 数学表达

### 基础 Information Gain (IG)

对问题 q，推理链 (s_1, ..., s_T)，答案候选 y：

$$I_i(y) = \log P(y | q, s_1, ..., s_i)$$

$$IG_i(y) = I_i(y) - I_0(y)$$

$I_0(y)$ 是仅看问题 q 时对 y 的 log-probability。

### Net Information (NetInfo)

引入 correct set C 和 incorrect set W：

$$\text{NetInfo}_i = \max_{y \in C} I_i(y) - \max_{y \in W} I_i(y)$$

### Monte Carlo Net Information Gain (MCNIG)

$$\text{MCNIG}_i = \text{NetInfo}_i - \text{NetInfo}_0$$

**Label**: $\mathbb{1}[\text{MCNIG}_i > \tau]$

## IG vs MCNIG 对比

| 维度 | IG | MCNIG |
|------|-----|--------|
| 参考对象 | 单一答案 | correct set + incorrect set |
| 长答案鲁棒性 | 差（代码、SQL 失败） | 好（max over sets 设计） |
| 计算成本 | O(N) | O(N)（相同） |
| 适用场景 | 短答案数学题 | 通用（数学、代码、SQL、医学） |

## 相关论文
- [[wiki/papers/royer-2026-mcnig|MCNIG (Royer et al., 2026)]] — 用 MCNIG 自动生成 PRM 训练数据
- [[wiki/papers/zhang-2025-empo|EMPO (Zhang et al., 2025)]] — 用 [[semantic-entropy|semantic entropy]]（一种相关的信息论度量）做 reward

## 面试常问点

- 🔴 Q: 什么是 MCNIG？它和普通 IG 有什么区别？
  - A: IG 只看模型对单一答案的 log-prob 变化，在长答案任务（代码、SQL）上失败——因为 subtle errors 也可能增加特定答案的 log-prob。MCNIG 引入 NetInfo：同时对比多个 correct 和 incorrect 答案，用 max over sets 设计更鲁棒。MCNIG = NetInfo_i - NetInfo_0。

- 🟡 Q: MCNIG 为什么能实现 O(N) 复杂度？
  - A: 利用 KV-caching。Prompt 和 CoT 只需处理一次生成 KV-cache，对每个候选答案在相同 cache 上做 forward 计算 log-prob。所有答案的 scoring 互不依赖，可并行。

## 与其他概念的关系
- 上位概念：信息论（Information Theory）
- 相关概念：[[semantic-entropy|Semantic Entropy]]（另一种信息论度量）
- 应用概念：[[process-reward-model|PRM]]（MCNIG 用于生成 PRM 训练数据）
- 对比概念：[[self-consistency|Self-Consistency]]（另一种自动生成 step-level labels 的方法）
