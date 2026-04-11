---
title: "STEP: Step-level Trace Evaluation and Pruning for Efficient Test-Time Scaling"
type: paper
tags: [test-time-scaling, hidden-states, trace-pruning, inference-efficiency, reasoning, step-scorer, KV-cache]
created: 2026-04-11
updated: 2026-04-11
sources: ["https://arxiv.org/abs/2601.09093"]
status: active
---

# STEP: Step-level Trace Evaluation and Pruning for Efficient Test-Time Scaling

## 一句话总结

训练一个 **轻量级 2-layer MLP** 作为 step scorer，利用 LLM 在推理步骤边界的 **hidden states** 预测 trace 质量，结合 **GPU 内存感知的 pruning 策略**，在 self-consistency 基础上减少 45%–70% 端到端推理延迟，同时提升准确率。

## 基本信息

- **作者**: Zhixiang Liang†, Beichen Huang†, Zheng Wang, Minjia Zhang
- **机构**: University of Illinois Urbana-Champaign (UIUC)
- **arXiv**: 2601.09093
- **代码**: https://github.com/Supercomputing-System-AI-Lab/STEP

## 核心贡献

1. **Hidden states at step boundaries encode reasoning quality early**：即使只看前 25% 的步骤，hidden state score 已经能有效区分 correct vs incorrect traces
2. **Lightweight step scorer**：2-layer MLP (input→512→1)，计算开销 < 10⁻⁶ 相对于 LLM 前向传播
3. **GPU memory-aware pruning**：当 KV cache 饱和时触发 pruning，彻底消除 waiting queue，比 token-level 节省更多延迟
4. **45%–70% latency reduction + accuracy improvement**：在 AIME-25、HMMT-24/25、GPQA-Diamond 上跨 3 个模型验证

## 方法

### Step Scorer

**Step representation**: 提取 `<think>...</think>` 之间的内容，用 `\n\n` 分割为步骤。每步的 representation 是该步 last token 的 **last-layer hidden state** $\mathbf{h}_n \in \mathbb{R}^d$。

**Label construction**: 将 trace-level correctness label $y \in \{0, 1\}$ 传播到所有步骤作为 pseudo-labels：
$$\tilde{y}_n = y, \quad \forall n \in \{1, \ldots, N\}$$

**Model architecture**: 2-layer MLP + sigmoid：
$$\hat{y}_n = \sigma(\mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{h}_n + \mathbf{b}_1) + b_2)$$

**Training loss**: Weighted BCE（补偿错误 trace 更长产生更多 negative steps 的不平衡）

**训练数据**: HMMT 2012–2023 数学题，每题 64 次采样，选 5000 correct + 5000 incorrect traces。

### GPU Memory-Aware Pruning

传统方法用固定阈值或时间表触发 pruning。STEP 发现**推理延迟的主要瓶颈不是 token 数量，而是 KV cache 饱和后的 waiting queue**（占 59% 时间）。

**策略**：当 GPU memory 满时，prune trace-level score 最低的 trace，释放其 KV cache。

**Trace-level score**: 所有已有 step scores 的均值：
$$score_t = \frac{1}{n} \sum_{i=1}^{n} \hat{y}_i^t$$

**Final answer**: 完成的 traces 按 trace-level score 做 weighted voting。

## 实验结果

### 主实验 (N=64, Table 1)

| 方法 | AIME-25 Acc | HMMT-25 Acc | Latency 降低 |
|------|------------|------------|-------------|
| **DeepSeek-R1-0528-Qwen3-8B** | | | |
| Self-Consistency | 83.3% | 70.0% | baseline |
| Slim-SC | 83.3% | 69.2% | 11% |
| DeepConf | 81.7% | 71.7% | ~35% |
| **STEP** | **85.0%** | **73.3%** | **~60%** |
| **Qwen3-4B-Thinking-2507** | | | |
| Self-Consistency | 86.7% | 65.0% | baseline |
| **STEP** | 88.3% | 70.0% | **~60%** |

### Hidden State 判别能力 (Figure 5)

Pairwise Ranking Accuracy（区分 correct vs incorrect traces）：
- Hidden state scorer @ 25% steps: ~63–65%
- Hidden state scorer @ 50% steps: ~67–70%
- Hidden state scorer @ 75% steps: ~70–73%
- Token confidence (baseline): ~55–58%

**关键发现**：Hidden states 在早期步骤就已经编码了推理质量信息，且远优于 token-level confidence。

### Latency Breakdown (Table 2, DeepSeek-8B on HMMT-25)

| 方法 | Wait Time (s) | Decode Time (s) |
|------|:---:|:---:|
| SC | 1526 | 1256 |
| DeepConf | 69+194 | 680+726 |
| Slim-SC | 1155 | 983 |
| **STEP** | **0** | **1024** |

STEP 完全消除了 waiting time。

### GPU Memory 鲁棒性 (Table 3)

| Memory Utilization | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|:---:|:---:|:---:|:---:|:---:|
| Accuracy | 70.0 | 69.1 | 70.0 | 68.3 | 73.3 |

即使 memory 限制到 0.5（更早触发 pruning），accuracy 仍然稳定（70.1±1.8%）。

## 🔴 与 SPC 研究的关系

### STEP 验证了 SPC 的核心假设

STEP 的核心发现——**hidden states 在早期推理步骤就能区分 correct/incorrect traces**——直接支持 SPC 的 probing 方法论。如果 hidden states 已经编码了推理质量，那么 SPC 的 step-level probing（通过短续写观察行为）本质上是在用**生成行为**来表面化这些 hidden state 中的信号。

### 关键区别

| 维度 | STEP | SPC |
|------|------|-----|
| **信号来源** | Hidden states (internal representation) | Rollout behavior (generative probing) |
| **用途** | Inference-time pruning | Training-time reward signal |
| **需要 GT?** | 是（训练 scorer 需要 correctness labels） | 否（与 final answer 对比） |
| **Scorer 架构** | 2-layer MLP on hidden states | 无额外模型，直接 semantic match |
| **粒度** | Step-level score → trace-level average | Step-level SPC score → advantage shaping |

### 借鉴意义

1. **SPC 的替代信号**：如果 hidden states 已经这么强，SPC 可以考虑同时用 hidden state probe 和 rollout probe，或用 hidden state probe 做 SPC 的快速近似（对应 SPC 方案中的 "SPC Probe 加速" 部分）
2. **训练数据**：STEP 用 5000 correct + 5000 incorrect traces 训练 MLP，数据需求非常低。SPC Probe 的训练数据策略可以参考
3. **Weighted voting**：STEP 的 trace-level score weighted voting 可以与 TTRL 的 majority voting 结合

## 面试 Q&A

- Q: 为什么 hidden states 能在早期步骤就预测推理质量？🟡
- A: LLM 的 hidden states 是上下文的压缩表示。在推理早期，如果模型已经"走在正确方向上"（正确理解题意、选择了正确的方法），其 hidden state 编码的信息模式会与最终成功的 trace 更相似。这与 probing 文献的发现一致——模型内部比输出层面更早"知道"答案是否正确。

- Q: STEP 的主要延迟瓶颈是什么？🟡
- A: 不是 token 生成数量，而是 KV cache 饱和后的 **waiting queue**。当 N=64 条 trace 并行生成时，KV cache 很快超出 GPU memory，导致 vLLM 需要 preempt traces → 排队等待 → 重建 KV cache。这个 waiting 占 59% 的总时间。STEP 在 memory 饱和时 prune 最差 trace，释放 KV cache，彻底消除 waiting。
