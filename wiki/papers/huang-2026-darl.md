---
title: "DARL: Diversity-Aware Reinforcement Learning for General Reasoning"
type: paper
tags: [RLVR, diversity-reward, general-reasoning, dynamic-threshold, Qwen2.5, Xiamen-University, Kuaishou]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2601.14700]
status: active
---

# DARL: 多样性感知的通用推理 RL

## 基本信息
- **作者**: Huang, Lin et al.
- **机构**: Xiamen University (厦门大学) / Kuaishou (快手)
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2601.14700)
- **链接**: https://arxiv.org/abs/2601.14700
- **Base Model**: Qwen2.5-7B

## 一句话总结
> 提出动态多样性 reward，在标准 correctness reward 基础上鼓励模型探索多样化的正确推理路径，通用推理任务（逻辑、代码、常识）大幅提升。注意：需要参考答案，非纯 URLVR。

## 摘要

当前 RLVR 方法过度聚焦于数学推理，且 reward 设计仅奖励正确性——模型收敛到单一答案模式后停止探索。DARL 提出 **Dynamic Diversity Reward**：在 correctness reward 基础上，对提供"新颖但合理"答案的 rollout 给予额外 bonus。关键设计是**动态阈值**：diversity bonus 不能无限大，必须受限于参考答案的 reward 水平（防止 diversity hack）。实验在通用推理任务（AutoLogic、HumanEval、常识推理）上验证，Qwen2.5-7B 在 General benchmarks 上 +9.5 over RLPR。

## 核心贡献

1. **提出通用推理 RL 的 diversity 问题**：数学推理答案唯一，但通用推理（代码、逻辑、常识）可能有多个合理答案——标准 RLVR 只奖励一种，抑制了有价值的探索
2. **Dynamic Diversity Reward**：$r_{dynamic} = \alpha \cdot r(y^*) + \beta \cdot \Delta r \cdot \mathbb{1}[\Delta r \leq r(y^*)/\gamma]$，兼顾正确性和新颖性
3. **动态阈值防止 diversity hack**：diversity bonus 被参考答案的 reward 水平所约束，确保"新颖"不偏离"合理"
4. **通用推理验证**：不仅限于数学，在逻辑推理、代码生成、常识推理上均有效

## 方法

### 问题定义

标准 RLVR reward 是二值的：$r = 1$ (correct) / $r = 0$ (incorrect)。这导致：
1. 模型收敛到**单一解题模式**（mode collapse to one reasoning pattern）
2. 多个等价正确答案中，只有与 reference 匹配的被奖励
3. 通用推理任务（代码有多种正确实现、逻辑有多种推理路径）受影响更大

### 技术方案

#### Dynamic Diversity Reward

在每个 GRPO group 中：
1. 评估所有 rollout 的 correctness reward $r(y_i)$
2. 选择参考答案 $y^*$（最高 reward 的 rollout）
3. 对每个 rollout 计算与参考答案的 reward 差异 $\Delta r = r(y_i) - r(y^*)$
4. 如果差异在合理范围内（$\Delta r \leq r(y^*)/\gamma$），给予 diversity bonus

### 关键公式

#### 1. Dynamic Diversity Reward

$$r_{dynamic} = \alpha \cdot r(y^*) + \beta \cdot \Delta r \cdot \mathbb{1}[\Delta r \leq r(y^*)/\gamma]$$

- $\alpha$：correctness 权重
- $\beta$：diversity bonus 权重
- $\gamma$：diversity 阈值（相对于参考 reward 的比例）
- $\Delta r = r(y_i) - r(y^*)$：与参考答案的 reward 差异
- $\mathbb{1}[\cdot]$：indicator function，只有在合理范围内才给 bonus

**直觉**：奖励那些"与参考答案不同但质量接近"的回答，鼓励模型探索多样化的正确路径。阈值 $r(y^*)/\gamma$ 确保 diversity 不偏离 correctness 太远。

### 实现细节

| 参数 | 值 |
|------|-----|
| Base Model | Qwen2.5-7B |
| 优化框架 | GRPO |
| α (correctness weight) | 任务相关 |
| β (diversity weight) | 任务相关 |
| γ (threshold ratio) | 任务相关 |
| 评估任务 | AutoLogic, HumanEval, 常识推理 |
| 对比方法 | RLPR (标准 RL with Process Reward) |

## 实验结果

### 主实验: Qwen2.5-7B

| 方法 | General Avg | AutoLogic | HumanEval | Reasoning |
|------|------------|-----------|-----------|-----------|
| RLPR | baseline | baseline | baseline | baseline |
| **DARL** | **+9.5** | **+11.2** | **+18.3** | **+1.3** |

**关键数据**：
- **General 任务大幅提升**：AutoLogic +11.2, HumanEval +18.3
- **Reasoning 任务小幅提升**：+1.3（数学推理答案唯一，diversity bonus 空间有限）
- **HumanEval +18.3 是亮点**：代码生成任务天然存在多种正确实现，diversity reward 效果最显著

### Diversity 分析

DARL 训练的模型在 inference 时生成的推理路径多样性显著更高（entropy 更高），但 accuracy 不降反升——证明 diversity 和 accuracy 并不矛盾，适度的 diversity 是有益的。

## 与其他工作的关系

- **基于**: [[wiki/concepts/grpo|GRPO]] 优化框架
- **对比**: RLPR (RL with Process Reward), 标准 GRPO
- **区别于纯 URLVR**: DARL **需要 ground-truth answers** 来计算 $r(y^*)$，不适用于无标注场景
- **与 [[wiki/papers/wang-2026-sarl|SARL]] 对比**: SARL 完全 label-free 且关注推理拓扑结构，DARL 需要标注但关注答案多样性。应用场景不同
- **与 [[wiki/papers/cui-2026-clipo|CLIPO]] 互补**: CLIPO 让正确推理聚拢（减少 spurious reasoning），DARL 鼓励正确推理多样化——前者减少"假阳性"，后者增加"真多样性"

## 局限性与开放问题

1. **⚠️ 需要 Ground Truth**：依赖参考答案计算 diversity threshold，**不是纯 URLVR 方法**，无法直接用于无标注场景
2. **Diversity 定义粗糙**：当前通过 reward 差异 $\Delta r$ 衡量"新颖性"，但这不直接反映推理路径的多样性
3. **超参数较多**：$\alpha, \beta, \gamma$ 三个超参数需要逐任务调优
4. **仅在 7B 验证**：更大模型是否有同样的 diversity 问题未知
5. **数学推理提升有限**：数学推理答案唯一的本质限制了 diversity bonus 的空间，DARL 更适合开放域任务

## 面试相关
> 这篇论文可能被问到的面试问题

- **Q: DARL 的 dynamic diversity reward 如何防止 diversity hack？**
- A: 通过动态阈值 $r(y^*)/\gamma$——diversity bonus 只有在与参考答案的 reward 差异不超过阈值时才给予。这确保"新颖"的回答必须在"合理"的范围内，不会奖励偏离太远的答案。

- **Q: 为什么 DARL 在代码生成（HumanEval）上效果特别好？**
- A: 代码生成天然存在多种等价正确实现（不同算法、不同数据结构），标准 RLVR 只奖励与 reference 匹配的一种，压制了有效探索。Diversity reward 释放了这些等价解的训练价值。

- **Q: DARL 能否用于无标注 (URLVR) 场景？**
- A: 不能直接使用，因为 diversity reward 的阈值计算依赖 ground-truth。但可以借鉴其思想：用 majority voting pseudo-label 替代 GT 计算阈值，或用 [[wiki/papers/du-2026-dare|DARE]] 的 distribution-aware reward 估计代替。

## 个人笔记

### 与 SPC 研究方案的关系

DARL 虽然不是纯 URLVR（需要 GT answers），但其思想对 [[wiki/synthesis/step-level-se-proposal|SPC 方案]] 有参考价值：

1. **SPC 也应避免 overfitting to single majority answer**：在 [[wiki/papers/zuo-2025-ttrl|TTRL]] 框架中，majority voting 选出的 pseudo-label 成为唯一"正确答案"。如果模型所有 rollout 都收敛到同一推理路径，虽然 consistency 高但丧失了探索能力。DARL 的 diversity 思想提醒 SPC 需要平衡 consistency 和 diversity
2. **动态阈值设计可参考**：SPC 的 consistency signal 也应该有动态阈值——根据模型当前置信度水平调节 SPC reward 的强度，避免在模型已经高度确定时还强制探索
3. **通用推理的启示**：如果 SPC 未来扩展到非数学任务（代码、逻辑），diversity-aware 设计将更加重要
