---
title: "V-Zero: Zero-Annotation Multimodal Self-Evolution via Questioner-Solver Co-Evolution"
type: paper
tags: [URLVR, multimodal, co-evolution, questioner-solver, dual-track-reward, zero-annotation, VLM, Zhejiang-University, Qwen2.5-VL]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2601.10094]
status: active
---

# V-Zero: 零标注多模态自进化

## 基本信息
- **作者**: Wang et al.
- **机构**: Zhejiang University (浙江大学)
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2601.10094)
- **链接**: https://arxiv.org/abs/2601.10094
- **Base Model**: Qwen2.5-VL-7B

## 一句话总结
> 零标注多模态自进化框架：Questioner 生成 MCQ + 直觉答案，Solver 用 CoT 采样 + majority voting 获得 pseudo-label，Dual-Track Reasoning Reward 对比 intuition 与 reasoning 提供训练信号，无监督 Solver 超越有监督 GRPO。

## 摘要

视觉语言模型（VLM）的推理增强通常依赖大量标注数据（如 RLHF、SFT）。V-Zero 提出完全零标注的多模态自进化方案：模型自身扮演两个角色——**Questioner**（出题者）和 **Solver**（解题者），形成 co-evolution 循环。Questioner 从图像生成多选题（MCQ）并给出**直觉答案**（fast, intuitive answer without reasoning）；Solver 用 CoT 推理采样多次 + majority voting 获得 pseudo-label。关键创新是 **Dual-Track Reasoning Reward**：对比直觉答案（System 1 快思考）和推理答案（System 2 慢思考）的一致/分歧情况，生成差异化 reward。实验证明无监督 Solver (51.9) 超越有监督 GRPO (50.8)。

## 核心贡献

1. **Questioner-Solver Co-Evolution**：两个角色来自同一模型，无需外部数据或评估器，实现完全自进化
2. **Dual-Track Reasoning Reward**：利用 System 1 (intuition) vs System 2 (reasoning) 的对比作为训练信号——一致时奖励 ambiguity（鼓励探索），分歧时奖励高置信推理（推理能力超越直觉说明推理有价值）
3. **零标注超越有监督**：Qwen2.5-VL-7B 无监督 Solver (51.9) 超越 supervised GRPO (50.8)，证明 co-evolution 在多模态场景的有效性
4. **多模态 URLVR 新范式**：首次在 VLM 上实现完全无标注的 RL 自进化

## 方法

### 问题定义

VLM 推理能力提升的标准路径是 RLHF/SFT，需要大量标注。能否让模型自己出题、自己做题、自己提升？

**关键挑战**：
1. 没有 ground truth 如何提供 reward？
2. 自我生成的训练数据质量如何保证？
3. 如何避免 [[wiki/papers/he-2026-urlvr-scale|He et al.]] 指出的 sharpening/co-sharpening？

### 技术方案

#### Co-Evolution 循环

```
Round t:
  1. Questioner_t: image → MCQ + intuitive answer (a_fast)
  2. Solver_t: MCQ → CoT reasoning × K → majority voting → pseudo-label (â)
  3. Dual-Track Reward: compare â vs a_fast → r_d
  4. GRPO update: 用 r_d 更新 Solver → Solver_{t+1}
  5. (Optional) 用 Solver 的反馈更新 Questioner → Questioner_{t+1}
```

#### Questioner 设计

Questioner 从图像生成多选题（MCQ），同时给出一个**直觉答案**（fast answer）——不经过 CoT 推理，直接凭"第一感觉"选择。这个 fast answer 代表 System 1 的判断。

#### Solver 设计

Solver 对 MCQ 进行 CoT 推理，采样 K 次，用 majority voting 获得 pseudo-label â 和置信度 c。

#### Dual-Track Reasoning Reward

**核心设计**：对比 intuition (a_fast) 和 reasoning (â) 的结果：

**Case 1: 一致 (â = a_fast)**
- 直觉和推理都选择了同一答案
- Reward: $r_d = \min(c, 1-c)$
- **直觉**：答案一致时给 ambiguity reward——置信度越接近 0.5 reward 越高。为什么？因为"一致且自信"的题太简单，不需要额外训练；"一致但不自信"的题 boundary case，训练价值更高

**Case 2: 分歧 (â ≠ a_fast)**
- 推理得到的答案与直觉不同
- Reward: $r_d = 0.5 \cdot c$
- **直觉**：当推理推翻了直觉，说明推理发现了直觉没看到的东西——这正是推理的价值。置信度 c 越高，说明推理越"确信"自己的判断，reward 越高

### 关键公式

#### 1. Consistency Case (â = a_fast)

$$r_d = \min(c, 1-c)$$

最大值在 $c=0.5$ 时取得（$r_d = 0.5$），$c \to 0$ 或 $c \to 1$ 时趋近 0。

#### 2. Divergence Case (â ≠ a_fast)

$$r_d = 0.5 \cdot c$$

$c$ 越大（推理越自信地推翻直觉），reward 越高。

#### 3. Pseudo-label via Majority Voting

$$\hat{a} = \arg\max_a \sum_{k=1}^{K} \mathbb{1}[a_k = a], \quad c = \frac{\max_a \sum_k \mathbb{1}[a_k = a]}{K}$$

### 实现细节

| 参数 | 值 |
|------|-----|
| Base Model | Qwen2.5-VL-7B |
| MCQ 选项数 | 4 |
| CoT 采样次数 K | 任务相关 |
| 优化框架 | GRPO |
| Co-evolution 轮数 | 多轮迭代 |
| 评估 Benchmarks | MMMU, MathVerse, 其他 VLM 推理 |

## 实验结果

### 主实验: Qwen2.5-VL-7B

| 方法 | MMMU | MathVerse | Avg |
|------|------|-----------|-----|
| Base | — | — | — |
| Supervised GRPO (with GT) | — | — | 50.8 |
| **V-Zero (unsupervised)** | **+3.9** | **+3.0** | **51.9** |

**核心发现**：无监督 V-Zero Solver (51.9) 超越有监督 GRPO (50.8)！

### 各 Benchmark 详细

| Benchmark | Improvement |
|-----------|------------|
| MMMU | +3.9 |
| MathVerse | +3.0 |
| 其他 VLM benchmarks | 一致正向提升 |

### Dual-Track Reward 分析

- **Consistency case** 占大多数（简单题，直觉和推理一致），ambiguity reward 防止模型在简单题上过度训练
- **Divergence case** 占少数但信号最强（推理推翻直觉的题目是最有训练价值的"hard examples"）

### Co-Evolution 动态

随着训练轮次增加：
- Solver 推理能力持续提升
- Questioner 生成的题目逐渐变难（因为 Solver 变强了）
- 形成正向螺旋（constructive co-evolution）

## 与其他工作的关系

- **理论框架**: [[wiki/papers/he-2026-urlvr-scale|He et al.]] 的 sharpening theorem 预测纯 intrinsic signal 会 sharpen。V-Zero 的 Dual-Track reward 通过 ambiguity reward 主动抑制过度 sharpening
- **Co-evolution 对比**: [[wiki/papers/chen-2025-mae|MAE]] 也是多角色 co-evolution，但 V-Zero 用 intuition vs reasoning 对比而非 Judge 评估。V-Zero 更适合多模态，MAE 更通用
- **TTRL 扩展**: [[wiki/papers/tan-2026-meta-ttrl|Meta-TTRL]] 将 TTRL 扩展到 T2I，V-Zero 将 co-evolution 扩展到 VLM 推理——两者都探索多模态自进化
- **多样性鼓励**: Dual-Track reward 的 ambiguity reward 与 [[wiki/papers/huang-2026-darl|DARL]] 的 diversity reward 思想类似——防止模型收敛到单一模式

## 局限性与开放问题

1. **MCQ 格式限制**：Questioner 生成 MCQ（多选题），但很多 VLM 任务是开放式的（描述、生成、推理），MCQ 格式限制了适用范围
2. **Intuition 可靠性**：fast answer（直觉）可能在某些任务上系统性偏差，导致 Dual-Track reward 信号不准
3. **Questioner 质量**：如果 Questioner 生成的题目质量差（太简单/太难/题目本身有错），整个 co-evolution 受限
4. **仅 7B 模型验证**：更大 VLM 的 co-evolution 动态可能不同
5. **Co-sharpening 风险**：虽然 Dual-Track reward 缓解了 sharpening，但长期训练中 Questioner 和 Solver 是否会陷入 mutual sharpening 未充分分析

## 面试相关
> 这篇论文可能被问到的面试问题

- **Q: V-Zero 的 Dual-Track Reasoning Reward 设计的核心直觉是什么？**
- A: 对比 System 1（直觉，不经推理的快速回答）和 System 2（CoT 推理后的答案）。如果二者一致，说明题目相对简单，给 ambiguity reward 鼓励边界探索。如果推理推翻了直觉，说明推理发现了直觉看不到的东西——这正是"推理能力"的价值，高置信度推理获得高 reward。

- **Q: 为什么无监督 V-Zero 能超越有监督 GRPO？**
- A: 两个原因：(1) Co-evolution 产生了自适应的训练分布——Questioner 根据 Solver 当前能力动态调整题目难度，形成 curriculum learning 效果；(2) Dual-Track reward 提供的信号比 binary GT reward 更 fine-grained——它区分了"简单一致"、"困难一致"、"推理推翻直觉"等多种情况。

- **Q: V-Zero 与 MAE 的 co-evolution 有什么区别？**
- A: MAE 有三个角色（Proposer-Solver-Judge），V-Zero 有两个（Questioner-Solver）。MAE 的 Judge 做 response-level 评估，V-Zero 用 Dual-Track reward 做 intuition-reasoning 对比。MAE 是纯文本，V-Zero 是多模态。V-Zero 的 MCQ 格式使 pseudo-label 更可靠（选择题 majority voting 比开放题更稳定）。

- **Q: Dual-Track reward 的 consistency case 为什么给 ambiguity reward（$\min(c, 1-c)$）而不是 high confidence reward？**
- A: 因为直觉和推理都一致且自信的题太简单，训练价值低——模型已经"会了"。真正有价值的是 boundary cases（c 接近 0.5）——模型有点不确定，正是提升空间所在。这与 [[wiki/papers/liu-2025-ettrl|ETTRL]] 的"高熵 token 分叉"洞察类似。

## 个人笔记

### 与 SPC 研究方案的关系

V-Zero 对 [[wiki/synthesis/step-level-se-proposal|SPC 方案]] 和 [[wiki/synthesis/co-evolving-verifier-proposal|Co-Evolving Verifier 方案]] 都有重要启发：

1. **Questioner-Solver co-evolution ↔ Co-Evolving Verifier**：V-Zero 的 Questioner 类似于 Co-Evolving 方案中的 lightweight PRM——都是与 policy 一起进化的辅助组件。V-Zero 证明了 co-evolution 在多模态场景可以超越有监督学习，为 Co-Evolving Verifier 提供了信心
2. **Dual-Track reward → SPC consistency**：V-Zero 的 "intuition vs reasoning" 对比可以启发 SPC 的 "early-step prediction vs final answer" 一致性度量——将 SPAE probing 的短续写答案视为 "intuition"（快速预测），将完整轨迹最终答案视为 "reasoning"，二者的一致性就是 SPC 信号
3. **Ambiguity reward 防止 sharpening**：V-Zero 在一致 case 给 ambiguity reward（奖励不确定性），这启发 SPC 也应该在 consistency 高的步骤适度降低 reward（避免过度 sharpening），在 consistency 低的步骤增大 reward（这些步骤是模型真正的提升空间）
4. **MCQ 格式的可靠性启示**：V-Zero 选择 MCQ 而非 open-ended QA 来保证 pseudo-label 质量。SPC 在选择 anchor 机制时也应优先考虑 "structured, verifiable" 的信号形式
