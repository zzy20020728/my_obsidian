---
title: "DCPO: Decoupled Calibration and Reasoning in RLVR"
type: paper
tags: [RLVR, calibration, gradient-conflict, decoupled-optimization, confidence, ECE, AUROC, Qwen3, CAS]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2603.09117]
status: active
---

# DCPO: 解耦推理与置信度校准

## 基本信息
- **作者**: Ma et al.
- **机构**: Chinese Academy of Sciences (中国科学院)
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.09117)
- **链接**: https://arxiv.org/abs/2603.09117
- **Base Model**: Qwen3-8B

## 一句话总结
> 首次证明 RLVR 训练中 accuracy 和 calibration 存在 fundamental gradient conflict，提出 DCPO 将 reasoning tokens 和 confidence tokens 分开优化，accuracy 不降的同时 ECE 降低 71.6%。

## 摘要

RLVR 训练的 LLM 在推理准确率上大幅提升，但置信度校准（calibration）严重退化——模型对自己的回答过度自信或过度不自信。DCPO 首次从理论上证明 accuracy 优化和 calibration 优化在 Fisher 信息度量下存在**负梯度内积**（fundamental gradient conflict），即同时优化二者本质上相互矛盾。解决方案：将推理 token（思维链内容）和置信度 token（表达确信程度的内容）的梯度更新**解耦**——各自用不同的 reward signal 独立优化。

## 核心贡献

1. **理论证明 Gradient Conflict**：首次在 Fisher 信息度量空间中证明 accuracy 和 calibration 的梯度方向冲突（内积 < 0），解释了为什么 RLVR 训练后模型校准退化
2. **DCPO 算法**：通过 masked gradient optimization，将 reasoning tokens 和 confidence tokens 的优化解耦
3. **ECE 大幅改善**：Qwen3-8B ECE 从 0.435 降到 0.128（-71.6%），同时 accuracy 维持在 60.8%（与 GRPO 持平）
4. **高 AUROC**：AIME24 上 AUROC 达到 0.914，意味着模型能可靠区分自己会不会做某道题

## 方法

### 问题定义：为什么 RLVR 破坏校准？

RLVR（如 [[wiki/concepts/grpo|GRPO]]）优化策略使模型生成更多正确答案。但这个优化过程有一个副作用：模型同时学会了"对所有回答都表现得很自信"，因为自信的回答在 RL 训练中更容易获得高 reward。

**根本原因**：accuracy loss 鼓励模型在正确答案上加大概率（不管是否"确信"），calibration loss 要求模型在不确信时降低概率——二者在同一参数空间中拉扯。

### 技术方案

#### Step 1: 证明梯度冲突

定义 accuracy 目标 $J_{acc}$ 和 calibration 目标 $J_{cal}$，在 Fisher 信息矩阵 $F$ 下计算梯度内积：

$$\langle \nabla J_{acc}, \nabla J_{cal} \rangle_{F^{-1}} < 0$$

这意味着沿 accuracy 梯度方向更新参数，calibration 必然恶化，反之亦然。

#### Step 2: Token 分类

将模型输出的 tokens 分为两类：
- **Reasoning tokens**：思维链中的推理步骤（占大部分）
- **Confidence tokens**：表达置信度的 token（如 "I'm confident that..."、"The answer is likely..."、最终答案的表达方式等）

#### Step 3: Masked Gradient Optimization

- Reasoning tokens 仅接收 accuracy reward 的梯度
- Confidence tokens 仅接收 calibration reward 的梯度
- 通过 gradient mask 实现，不增加额外计算开销

### 关键公式

#### 1. Gradient Conflict（核心理论结果）

$$\langle \nabla_\theta J_{acc}(\theta), \nabla_\theta J_{cal}(\theta) \rangle_{F(\theta)^{-1}} < 0$$

其中 $F(\theta)$ 是 Fisher 信息矩阵。直觉：在自然梯度空间中，accuracy 和 calibration 的最优方向相反。

#### 2. Calibration Reward

$$R_c = -|conf - R_{IG}|$$

其中 $conf$ 是模型表达的置信度，$R_{IG}$ 是 instance-group accuracy（该问题在 group 中的正确率）。当模型的置信度与实际正确率匹配时 reward 最高。

#### 3. Masked Gradient Update

$$\nabla_\theta \mathcal{L} = M_{reason} \odot \nabla_\theta J_{acc} + M_{conf} \odot \nabla_\theta J_{cal}$$

其中 $M_{reason}$ 和 $M_{conf}$ 是互斥的 token-level mask。

### 实现细节

| 参数 | 值 |
|------|-----|
| Base Model | Qwen3-8B |
| 优化框架 | GRPO + DCPO (masking) |
| Confidence 提取 | 从模型输出的特定格式中解析 |
| Calibration 度量 | ECE (Expected Calibration Error) |
| 辨别度量 | AUROC |
| Accuracy reward | 标准 binary (correct/incorrect) |
| Calibration reward | $R_c = -\|conf - R_{IG}\|$ |

## 实验结果

### 主实验

| 方法 | Accuracy (%) | ECE (↓) | AUROC (↑) |
|------|-------------|---------|-----------|
| Qwen3-8B Base | — | — | — |
| GRPO | **61.2** | 0.435 | 0.721 |
| GRPO + Calibration Loss | 58.3 | 0.312 | 0.803 |
| **DCPO** | **60.8** | **0.128** | **0.914** |

**关键数据**:
- **ECE**: 0.435 → 0.128，降低 **71.6%**
- **Accuracy**: 60.8%，仅比 GRPO 低 0.4%（在误差范围内）
- **AUROC**: 0.914，模型能高度可靠地区分会做 vs 不会做的题
- 对比"GRPO + Calibration Loss"（朴素联合优化）：accuracy 降了 2.9%，ECE 也没有 DCPO 好——**印证了 gradient conflict 理论**

### AIME24 详细结果

| 方法 | AIME24 Score | AIME24 AUROC |
|------|-------------|-------------|
| GRPO | ~60% | 0.721 |
| **DCPO** | ~60% | **0.914** |

## 与其他工作的关系

- **基于**: [[wiki/concepts/grpo|GRPO]] 优化框架
- **理论联系**: 与 [[wiki/papers/wang-2026-pipo|PIPO]] 都发现了 GRPO 的隐含问题——PIPO 发现梯度爆炸，DCPO 发现 accuracy-calibration gradient conflict
- **应用互补**: [[wiki/papers/du-2026-dare|DARE]] 用分布级 reward 改善 reward estimation 的准确性，DCPO 改善模型对自身判断的校准性
- **与 URLVR 的交叉**: 校准良好的模型在 [[wiki/papers/zuo-2025-ttrl|TTRL]] majority voting 中能提供更可靠的投票——DCPO 训练的模型可能是更好的 TTRL base model

## 局限性与开放问题

1. **Token 分类的可靠性**：如何准确区分 reasoning tokens 和 confidence tokens？当前方法依赖输出格式，对自由格式输出可能不适用
2. **仅在数学推理验证**：其他任务（代码、常识推理）的 gradient conflict 程度可能不同
3. **Calibration reward 设计**：$R_{IG}$ 依赖 group-level accuracy，这在小 group 或极难/极易题目上可能噪声很大
4. **与其他 RL 算法的兼容性**：仅在 GRPO 上验证，DAPO/PPO 等算法的 gradient conflict 特征可能不同
5. **推理时 confidence 的利用**：DCPO 训练出校准良好的模型，但如何在推理时利用 confidence（如 rejection、routing）未深入探讨

## 面试相关
> 这篇论文可能被问到的面试问题

- **Q: 为什么 RLVR 训练会破坏模型的置信度校准？**
- A: 因为 accuracy 优化和 calibration 优化存在 fundamental gradient conflict（在 Fisher 信息度量下梯度内积 < 0）。直觉上，accuracy loss 鼓励模型"自信地给出答案"（无论对错），而 calibration loss 要求模型"不确定时承认不确定"，二者在参数空间中拉扯。

- **Q: DCPO 是如何解决这个问题的？**
- A: 将 token 分为 reasoning tokens 和 confidence tokens，通过 gradient mask 让 reasoning tokens 只接收 accuracy reward 的梯度，confidence tokens 只接收 calibration reward 的梯度。这样两类梯度在不同的 token 子集上独立优化，避免冲突。

- **Q: ECE 和 AUROC 分别衡量什么？**
- A: ECE (Expected Calibration Error) 衡量模型表达的置信度与实际正确率的匹配程度，越低越好。AUROC 衡量模型用置信度区分正确和错误回答的能力，越高越好。DCPO 在两个指标上都大幅领先。

- **Q: 这个发现对 LLM 部署有什么实际意义？**
- A: 校准良好的模型可以实现：(1) 可靠的 rejection（低置信度时拒绝回答）; (2) 路由到更强模型或人类专家; (3) 在 multi-agent 系统中提供可信的投票权重; (4) 更好的 uncertainty quantification。

## 个人笔记

### 与 SPC 研究方案的关系

DCPO 的发现对 [[wiki/synthesis/step-level-se-proposal|SPC 方案]] 有多层面启示：

1. **SPC probing 本质涉及模型置信度**：SPC 的短续写一致性检查实际上是在探测模型在每个步骤上的"方向确信度"。DCPO 证明置信度信号和 accuracy 信号存在 gradient conflict，意味着 **SPC 信号不应该与 outcome reward 直接相加**——需要某种解耦机制
2. **Decoupled optimization 可迁移到 SPC**：SPC 可以将 step-level consistency signal 和 outcome-level TTRL reward 在梯度层面解耦，类似 DCPO 的 masked gradient approach
3. **AUROC 0.914 的启示**：如果 DCPO 可以使模型高度可靠地预测自己是否正确，那么 SPC 的 probing 信号理论上也可以达到类似的辨别度
4. **更好的 TTRL base model**：DCPO 训练的校准模型可能是更好的 TTRL 起点——校准良好的模型在 majority voting 中能提供更可靠的"投票"
