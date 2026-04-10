---
title: "CLIPO: Contrastive Learning Improves Policy Optimization for LLM Reasoning"
type: paper
tags: [RLVR, contrastive-learning, InfoNCE, cross-trajectory, hallucination-suppression, Qwen, Alibaba, GRPO, DAPO]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2603.10101]
status: active
---

# CLIPO: 对比学习改进 RLVR 策略优化

## 基本信息
- **作者**: Cui et al.
- **机构**: Alibaba Qwen Team (阿里巴巴通义千问团队)
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.10101)
- **链接**: https://arxiv.org/abs/2603.10101
- **Base Model**: Qwen 系列

## 一句话总结
> 在 GRPO group 的 successful rollouts 上施加 InfoNCE 对比损失，捕获正确推理路径的不变结构，抑制 hallucination 和 answer-copying，跨 4 种 RL 算法一致提升。

## 摘要

当前 RLVR 方法（[[wiki/concepts/grpo|GRPO]] / DAPO 等）仅用 outcome reward 做策略优化，正确 rollout 之间的**结构相似性**被忽略——模型可能通过不同的（甚至不正确的）推理路径碰巧得到正确答案（answer-copying、hallucination reasoning 等）。CLIPO 提出在 group 内的 successful rollouts 之间施加 InfoNCE 对比学习损失，让模型学会正确推理路径的**不变表示**（invariant representation），从而抑制 spurious reasoning patterns。该方法作为 reward augmentation 组件即插即用，在 GRPO/DAPO/Dr.GRPO/PRIME 四种 RL 算法上均带来一致提升。

## 核心贡献

1. **识别 RLVR 的 spurious reasoning 问题**：即使 outcome 正确，不同 rollout 的推理路径质量差异很大——有的是"正确推理"，有的是"蒙对"
2. **InfoNCE 对比学习作为 regularization**：在 successful rollouts 间建立 contrastive loss，强制正确推理的表示空间收缩（聚拢），错误推理被推远
3. **Reward augmentation 设计**：将 contrastive loss 转化为 reward 调整项（而非额外 loss），无缝集成到任何 RL 算法
4. **跨算法普适性**：在 GRPO、DAPO、Dr.GRPO、PRIME 四种算法上均有效，证明这不是 algorithm-specific 的 trick

## 方法

### 问题定义

在 GRPO 的一个 group 中采样 G 个 rollouts，假设其中 $G^+$ 个得到正确答案。标准 RLVR 对这 $G^+$ 个正确 rollout 赋予相同的正 reward。

**问题**：这些正确 rollout 的推理质量可能天差地别：
- **高质量**：严谨的数学推导 → 正确答案
- **低质量**：跳步、错误中间步骤碰巧抵消、直接猜答案
- **Hallucination**：看似合理但逻辑链断裂的推理

如果不区分这些，模型可能强化 spurious patterns。

### 技术方案

#### Contrastive Head

在策略模型的最后一层 hidden state 上加一个轻量级 projection head（线性层），将 rollout 映射到 contrastive embedding space。

#### Contrastive Loss (InfoNCE)

对于一个 successful rollout $y_i$：
- **正样本** $\bar{y}_i$：同一 prompt 的其他 successful rollouts 的平均表示
- **负样本** $y_j$：同一 prompt 的 failed rollouts

拉近 $y_i$ 和 $\bar{y}_i$（正确推理聚拢），推远 $y_i$ 和 failed rollouts。

#### Reward Augmentation

将 contrastive loss 转化为 reward 调整项，而非直接加到 policy loss 上（更稳定）。

### 关键公式

#### 1. InfoNCE Contrastive Loss

$$\mathcal{L}_{CL} = -\log \frac{\exp(f(y_i, \bar{y}_i))}{\sum_{j} \exp(f(y_i, y_j))}$$

其中 $f(y_i, y_j) = \frac{\text{sim}(h_i, h_j)}{\tau}$，$h_i$ 是 rollout $y_i$ 经 contrastive head 后的表示，$\tau$ 是温度参数。

#### 2. Reward Augmentation

$$r'_i = r_i + \max(-\lambda \cdot \mathcal{L}_{CL}(x, y_i), -0.5)$$

**解读**：
- $r_i$ 是原始 outcome reward
- $-\lambda \cdot \mathcal{L}_{CL}$：contrastive loss 越小（与其他正确 rollout 越相似），bonus 越大
- $\max(\cdot, -0.5)$：设置下界，防止 contrastive penalty 过大导致 reward 崩溃
- $\lambda$：超参数控制 contrastive regularization 强度

#### 3. Cosine Similarity

$$f(y_i, \bar{y}_i) = \frac{h_i \cdot \bar{h}_i}{\|h_i\| \cdot \|\bar{h}_i\|} \cdot \frac{1}{\tau}$$

### 实现细节

| 参数 | 值 |
|------|-----|
| Contrastive head | 单层线性投影 |
| 温度 τ | 任务相关 |
| λ (contrastive weight) | 任务相关 |
| Reward 下界 | -0.5 |
| 兼容算法 | GRPO, DAPO, Dr.GRPO, PRIME |
| Track I 训练集 | GSM8K |
| Track II 训练集 | MATH |

## 实验结果

### Track I: GSM8K 训练 → 8 Benchmarks 评估

| 算法 | Baseline Avg | +CLIPO Avg | Δ |
|------|-------------|-----------|---|
| GRPO | — | — | +1.12 |
| DAPO | — | — | +提升 |
| Dr.GRPO | — | — | +提升 |
| PRIME | — | — | +提升 |

### Track II: MATH 训练 → Competition-Level 评估

| 算法 | Baseline Avg | +CLIPO Avg | Δ |
|------|-------------|-----------|---|
| GRPO | — | — | +1.35 |

### 跨算法一致性

**关键发现**：CLIPO 在所有 4 种 RL 算法上都带来正向提升，证明 contrastive regularization 解决的是 RLVR 的**通用问题**（spurious reasoning），而非某种特定算法的缺陷。

### Hallucination 抑制

CLIPO 显著减少了"正确答案但错误推理"的比例——通过强制正确 rollout 在表示空间中聚拢，偏离主流正确推理模式的 outlier（hallucination reasoning）的 reward 被压低。

## 与其他工作的关系

- **基于**: [[wiki/concepts/grpo|GRPO]] 框架，作为 reward augmentation 组件
- **解决的问题**: 与 [[wiki/papers/wang-2026-pipo|PIPO]] 解决 GRPO 梯度问题不同，CLIPO 解决的是 reward signal 的信息缺失（不区分 reasoning quality）
- **理论联系**: [[wiki/papers/he-2026-urlvr-scale|He et al.]] 的 sharpening 理论 — CLIPO 的 contrastive loss 提供了额外的 structural constraint，可能减缓 sharpening（因为模型被约束在正确推理的 manifold 上）
- **互补方法**: 可与 [[wiki/papers/du-2026-dual-consensus|DCRL]]、[[wiki/papers/du-2026-dare|DARE]] 等 reward improvement 方法叠加使用

## 局限性与开放问题

1. **Contrastive head 需要额外训练**：虽然轻量，但增加了超参数（τ, λ）和训练复杂度
2. **依赖足够的 successful rollouts**：如果 group 中 $G^+$ 很小（极难题），contrastive loss 可能不稳定
3. **"正确推理"的定义模糊**：contrastive loss 只能保证 successful rollouts 的表示聚拢，但聚拢的方向不一定是"真正正确的推理"
4. **未探索 step-level contrastive**：当前在 rollout-level 做对比，step-level contrastive 可能效果更好但成本更高
5. **Reward 下界 -0.5 的选择**：缺乏理论指导，可能需要逐任务调参

## 面试相关
> 这篇论文可能被问到的面试问题

- **Q: CLIPO 解决了 RLVR 的什么核心问题？**
- A: 正确答案 ≠ 正确推理。标准 RLVR 仅用 outcome reward，无法区分"严谨推理得到正确答案"和"蒙对/hallucination reasoning"。CLIPO 通过 contrastive learning 让正确 rollout 的表示聚拢，偏离正确推理模式的 outlier 被自然惩罚。

- **Q: 为什么用 reward augmentation 而不是直接加 contrastive loss 到 policy loss？**
- A: 直接加 loss 会改变优化目标的数学性质（如 PPO/GRPO 的 clip/trust-region 保证），可能导致训练不稳定。转化为 reward 调整项后，整个 RL 优化框架保持不变，contrastive 信号仅通过 reward 间接影响策略更新。

- **Q: CLIPO 与 self-consistency 方法（如 [[wiki/papers/zhang-2025-covo|CoVo]]）有什么区别？**
- A: Self-consistency 方法用 voting/consistency 判断答案是否正确（outcome-level），CLIPO 用 contrastive learning 判断推理路径是否"正统"（process-level）。前者解决"答案对不对"，后者解决"推理正不正"。

- **Q: InfoNCE loss 中温度 τ 的作用是什么？**
- A: τ 控制对比学习的"严格程度"——τ 小时模型被强制学习非常细粒度的区分（hard negatives 权重大），τ 大时更宽容。在 RLVR 中 τ 太小可能导致正确 rollout 之间的微小差异被过度放大。

## 个人笔记

### 与 SPC 研究方案的关系

CLIPO 对 [[wiki/synthesis/step-level-se-proposal|SPC 方案]] 的启发非常直接：

1. **解决 SPC 问题 #5 "没有跨轨迹信息"**：SPC 当前只看单条轨迹的 step-level consistency，CLIPO 提供了跨轨迹视角的方法论——可以在 SPC 框架中增加 contrastive loss，让"步骤语义一致"的轨迹在表示空间中聚拢
2. **Reward augmentation 设计可借鉴**：SPC 的 step-level reward 也可以采用类似的 augmentation 方式（而非替换 outcome reward），通过 $r' = r_{TTRL} + \max(-\lambda \cdot \Phi_{SPC}, \text{lower\_bound})$ 集成
3. **Step-level contrastive 是自然延伸**：CLIPO 在 rollout-level 做对比，SPC 可以在 step-level 做对比——比较不同轨迹在同一 step boundary 的 semantic rollout 一致性
4. **抑制 spurious majority**：CLIPO 抑制 hallucination reasoning，与 [[wiki/papers/du-2026-dual-consensus|DCRL]] 抑制 spurious majority 形成互补——SPC 可以同时利用两种信号
