---
title: "How Far Can Unsupervised RLVR Scale LLM Training?"
type: paper
tags: [URLVR, 综述, intrinsic-reward, external-reward, sharpening, model-collapse, ICLR-2026]
created: 2026-04-08
updated: 2026-04-08
sources: [https://arxiv.org/abs/2603.08660]
status: draft
---

# How Far Can Unsupervised RLVR Scale?

> **STATUS: DRAFT** — 基于 abstract 和已知信息编写，全文内容待补充。HTML 版本返回 404，PDF > 5MB 无法直接获取。请提供论文全文以完善此页面。

## 基本信息
- **作者**: He et al.
- **机构**: Tsinghua University
- **年份**: 2026
- **会议/期刊**: **ICLR 2026**（已接收）
- **链接**: https://arxiv.org/abs/2603.08660

## 一句话总结
> URLVR 领域最重要的综合分析论文：建立统一分类体系（intrinsic vs external），证明所有 intrinsic 方法本质上都在做 "sharpening the model's initial distribution"，发现 rise-then-fall pattern 及 Model Collapse Step (MCS) 概念。

## 摘要
本文对 URLVR 方法进行系统性综合分析。将方法分为 **intrinsic**（内部信号）和 **external**（外部信号）两类。提出统一理论框架，证明所有 intrinsic 方法本质上都在做 **sharpening the model's initial distribution**（锐化模型初始分布）。关键发现：sharpening 在 initial confidence 与 correctness 对齐时有效，但 misaligned 时会 **catastrophically fail**。所有 intrinsic 方法都呈现 **rise-then-fall pattern**（先升后降），collapse 时间由 model prior 决定而非工程选择。提出 **Model Collapse Step (MCS)** 作为衡量 model prior 和 RL trainability 的实用指标。初步探索 external reward 方法（基于 computational asymmetries），证据表明可以 escape confidence-correctness ceiling。

## 核心贡献（基于 Abstract）
1. **统一分类体系**：将 URLVR 方法分为 intrinsic（内部信号）vs external（外部信号）两大类
2. **Sharpening 理论**：证明所有 intrinsic 方法的统一机制——锐化模型初始分布
3. **Rise-then-fall pattern**：发现所有 intrinsic 方法的共同训练动态——先上升后崩溃
4. **Model Collapse Step (MCS)**：提出量化 model prior 质量和 RL trainability 的实用指标
5. **Confidence-Correctness Alignment**：揭示 intrinsic 方法成功/失败的根本条件
6. **External reward 探索**：初步证据表明 computational asymmetries 可以突破 intrinsic 方法的天花板

## 方法（基于 Abstract，待补充全文细节）

### 统一理论框架：Sharpening Mechanism

**核心论点**：所有 intrinsic reward methods（包括 [[wiki/papers/zhang-2025-empo|EMPO]] 的 semantic entropy、[[wiki/papers/ghimire-2026-prism|PRISM]] 的 self-certainty 等）本质上都在做同一件事——**sharpening the model's initial distribution**。

直觉理解：
- 模型对每个问题有一个"初始答案分布"（来自预训练）
- Intrinsic rewards 鼓励模型更确定地输出其"最有信心"的答案
- 如果初始分布中最有信心的答案恰好是正确的 → sharpening 有效 → accuracy 提升
- 如果初始分布中最有信心的答案是错误的 → sharpening 灾难性失败 → accuracy 下降

### Confidence-Correctness Alignment

$$\text{Sharpening 有效} \iff P_{init}(\text{correct answer}) > P_{init}(\text{any incorrect answer})$$

当 confidence 与 correctness aligned（模型对正确答案最有信心）时，sharpening 提升 accuracy。

当 misaligned 时，sharpening 让模型更确定地输出错误答案 → **catastrophic failure**。

### Rise-then-Fall Pattern

所有 intrinsic 方法的训练曲线呈现相同模式：
1. **Rise phase**: 模型在 confidence-correctness aligned 的样本上快速提升
2. **Fall phase**: 随着训练继续，模型开始在 misaligned 样本上过度 sharpen → accuracy 急剧下降

这个 pattern 与 [[wiki/papers/ghimire-2026-prism|PRISM]] 的实验发现完全一致（100 步后崩溃）。

### Model Collapse Step (MCS)

**定义**：训练曲线从 rise 转为 fall 的拐点步数。

**关键发现**：
- MCS 由 **model prior**（模型预训练质量）决定，而非工程选择（学习率、batch size 等）
- 更好的 base model → 更高的 MCS → 可以训练更久
- MCS 可以作为 RL trainability 的实用指标：MCS 高 = 模型适合用 intrinsic rewards 训练

### External Rewards：突破天花板

**Computational Asymmetries 方法**：
- 利用某些任务中验证比生成更容易的特性（如数学证明：验证一个证明比写一个证明容易）
- 这类 external reward 不受 confidence-correctness alignment 限制
- 初步证据表明可以 escape intrinsic 方法的天花板

## 与其他工作的关系
- **直接验证 [[wiki/papers/ghimire-2026-prism|PRISM]]**：PRISM 发现的"纯内部信号长期不可靠"在这里被理论化——sharpening mechanism + rise-then-fall pattern
- **理论解释 [[wiki/papers/zhang-2025-empo|EMPO]]**：EMPO 的 semantic entropy minimization 本质上就是 sharpening。Entropy thresholding 是对 misaligned 样本的启发式过滤
- **解释 [[wiki/papers/rahman-2025-spark|SPARK]]**：SPARK 的 trained PRM 属于 external reward（stationary signal），所以不受 rise-then-fall pattern 限制
- **与 [[wiki/papers/wu-2026-self-judge|Self-Judge]]**：Self-Judge 的 frozen Judge 属于 external，SC 频率属于 intrinsic。混合方案可能部分缓解 sharpening 的问题
- **与 [[wiki/papers/royer-2026-mcnig|MCNIG]]**：MCNIG 训练的 PRM 属于 external reward，理论上不受 sharpening 限制
- **与 [[wiki/papers/tan-2026-ctrl-rag|CTRL-RAG]]**：CLR 利用文档对比，可以视为一种 computational asymmetry（有文档 vs 无文档），可能属于 external reward

## 面试相关

- Q: URLVR 领域的 intrinsic vs external reward 有什么区别？🔴
- A: Intrinsic rewards（如 EMPO 的 semantic entropy、self-certainty）完全来自模型自身，不依赖外部模型或标注。External rewards（如 SPARK 的 trained PRM）来自冻结的外部模型。He et al. 2026 (ICLR 2026) 证明所有 intrinsic 方法都在做 "sharpening"——锐化模型初始分布，当 confidence 与 correctness 对齐时有效，misaligned 时灾难性失败。

- Q: 什么是 rise-then-fall pattern？为什么所有 intrinsic 方法都会出现？🔴
- A: 训练曲线先上升后下降。Rise phase：模型在 confidence-correctness aligned 的样本上提升。Fall phase：模型在 misaligned 样本上过度 sharpen，输出更确定的错误答案。根本原因是 sharpening mechanism 无法区分正确和错误的 high-confidence outputs。

- Q: Model Collapse Step (MCS) 是什么？有什么实用价值？🟡
- A: MCS 是训练曲线从 rise 转为 fall 的拐点步数。它由 model prior（预训练质量）决定，不受工程超参数影响。实用价值：MCS 越高说明 base model 越适合用 intrinsic rewards 训练。可以用 MCS 预估训练应该在什么时候停止。

- Q: 如何突破 intrinsic rewards 的天花板？🔴
- A: 两条路线：(1) 混合信号——像 PRISM 一样结合 external PRM 和 intrinsic signal，利用 external 信号防止 sharpening 失败；(2) Computational asymmetries——利用"验证比生成容易"的任务特性构建 external reward。He et al. 初步证据表明 external rewards 可以 escape confidence-correctness ceiling。

## 个人笔记
> 这是整个 URLVR 领域最重要的综合分析工作（ICLR 2026）。Sharpening mechanism 为前面四篇论文提供了统一的理论解释：
>
> - EMPO 的 semantic entropy minimization = sharpening
> - PRISM 发现的 rise-then-fall = sharpening 的必然结果
> - SPARK 的 PRM 不崩溃 = external reward 不受 sharpening 限制
> - Self-Judge 的混合方案部分缓解 = frozen Judge 提供 external 校准
>
> MCS 概念特别有实用价值——可以用来预判一个 base model 适不适合做 URLVR，以及应该训练多少步。
>
> **待补充**：全文中一定有大量关键实验细节（具体的 sharpening 证明、MCS 的计算方法、external reward 的具体实现等）。需要获取全文后完善此页面。
>
> **获取全文的方式**：(1) 用户直接提供 PDF；(2) 从 ICLR 2026 官方页面获取；(3) Semantic Scholar 或 Papers With Code 可能有 HTML 版本。
