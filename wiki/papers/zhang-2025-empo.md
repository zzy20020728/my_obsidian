---
title: "Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization"
type: paper
tags: [URLVR, semantic-entropy, unsupervised-RL, GRPO, reasoning, reward-free]
created: 2026-04-07
updated: 2026-04-07
sources: [https://arxiv.org/abs/2504.05812]
status: active
---

# EMPO: Entropy-Minimized Policy Optimization

## 基本信息
- **作者**: Qingyang Zhang et al.
- **机构**: Tianjin University & Tencent AI Lab
- **年份**: 2025
- **会议/期刊**: arXiv preprint (arXiv:2504.05812)
- **链接**: https://arxiv.org/abs/2504.05812
- **代码**: https://github.com/QingyangZhang/EMPO

## 一句话总结
> 完全无监督的 LLM 推理激励方法，通过最小化语义熵（semantic entropy）构建 reward，仅需问题 {q} 不需要答案 {a}，即可显著提升推理能力。

## 摘要
现有 RLVR 方法依赖 ground-truth 答案做 reward（rule-based verification），限制了其在无标注数据上的应用。EMPO 提出完全无监督的方案：对每个问题采样一组输出，按语义等价性聚类，将所属聚类的频率占比作为 reward 信号，配合 entropy thresholding 过滤极端样本，基于 [[grpo|GRPO]] 框架优化。核心洞察是 **语义熵与模型准确率有强负相关**，因此最小化语义熵等价于鼓励模型收敛到正确答案。

## 核心贡献
1. **完全无监督**：首次实现仅用 {q}（无需 {a}）的 LLM 推理 RL，打破了 RLVR 对 ground-truth 的依赖
2. **语义熵作为 reward**：提出基于语义聚类频率的 reward 信号，理论上等价于最小化 semantic entropy
3. **Entropy Thresholding**：用双阈值（δ_low, δ_high）过滤语义熵极端的样本，防止 reward hacking
4. **General-Verifier SLM**：对自由形式任务，训练 1.5B 的小模型做双向蕴含判断，实现语义等价性聚类

## 方法

### 问题定义
标准 RLVR 需要 (q, a) 对做 reward verification。在很多实际场景（开放域问答、复杂推理），获取 ground-truth 答案成本极高。能否在完全没有答案标注的情况下，仍然通过 RL 提升 LLM 推理能力？

### 技术方案

#### 1. 语义聚类与 Reward 构建
对每个问题 q，采样 G 个输出 {o_1, ..., o_G}，按语义等价性将输出聚类为 meaning clusters {c_1, ..., c_K}：

$$r_i = p(\hat{c}_j | q) \approx \frac{|c_j|}{G}$$

其中 $c_j$ 是 $o_i$ 所属的语义聚类。直觉：**被更多独立采样支持的答案更可能正确**。

#### 2. 语义等价性判断
- **数学任务**：用正则表达式提取最终答案，字符串精确匹配
- **自由形式任务**：训练 1.5B 的 General-Verifier SLM，判断两个输出是否双向蕴含（bidirectional entailment）

#### 3. Entropy Thresholding
计算每个问题的语义熵：

$$H(q) = -\sum_{j=1}^{K} p(c_j|q) \log p(c_j|q)$$

仅保留 $\delta_{low} < H(q) < \delta_{high}$ 的样本用于训练：
- **过滤低熵**（δ_low）：模型已经很确定的问题，继续训练收益小且可能 overfit
- **过滤高熵**（δ_high）：模型完全随机的问题，reward 信号太 noisy 无法学习

#### 4. EMPO 目标函数

### 关键公式

基于 [[grpo|GRPO]] 框架：

$$\mathcal{L}_{EMPO}(\theta) = -\mathbb{E}_{q \sim D, \{o_i\} \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(r_t^i(\theta) \hat{A}_i, \; \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right]$$

其中 advantage $\hat{A}_i$ 是基于语义聚类 reward 的归一化值。

## 实验结果

### 数学推理

| Benchmark | Qwen2.5-Math-7B Base | + EMPO | Qwen2.5-Math-7B Instruct |
|-----------|---------------------|--------|--------------------------|
| GSM8K | 82.5% | **88.7%** | 91.6% |
| MATH500 | 52.6% | **70.4%** | 75.2% |
| Minerva Math | 19.9% | **35.5%** | 34.0% |
| GaoKao | 42.4% | **64.8%** | 66.1% |
| OlympiadBench | 19.2% | **30.1%** | 33.9% |
| 平均 | 30.7% | **48.1%** | 49.4% |

EMPO 在无需任何标注的情况下，让 Base 模型接近 Instruct 版本的效果。

### 通用推理（Qwen2.5-7B Base）

| Benchmark | Base | + EMPO |
|-----------|------|--------|
| MMLU-Pro | 32.1% | **50.1%** |
| GPQA | 25.4% | **31.3%** |
| ARC-C | 52.8% | **58.5%** |

### 消融实验
- **去掉 entropy thresholding**：性能明显下降，高/低熵样本引入 noise
- **用 majority voting 代替 semantic clustering**：自由形式任务效果差，因为表述不同但语义相同的答案被错误区分
- **General-Verifier SLM vs LLM-as-Judge**：SLM 更高效且准确率接近

## 关键发现
1. **语义熵与准确率的强负相关**：实验验证在多个 benchmark 上，semantic entropy 下降伴随 accuracy 提升
2. **EMPO 不学习新能力，而是"找到更好的路径"**：效果来自更好地选择预训练中已有的强推理路径（evidence: 训练前后模型的 pass@k 变化模式）
3. **完全无监督 ≈ 有监督 RLVR 的 ~97%**：在数学任务上，EMPO 达到 ground-truth RLVR 的 97% 效果

## 与其他工作的关系
- **基于**: [[grpo|GRPO]] 框架做策略优化
- **核心概念**: [[semantic-entropy|语义熵 (Semantic Entropy)]]
- **对比**: 标准 RLVR（需要 ground-truth）、[[wiki/papers/ghimire-2026-prism|PRISM]]（也用内部信号但发现纯内部信号不可靠）
- **互补视角**: [[wiki/papers/rahman-2025-spark|SPARK]] 用 [[self-consistency|self-consistency]] 训练 PRM，EMPO 直接用语义聚类频率做 reward
- **关键分歧**: [[wiki/papers/ghimire-2026-prism|PRISM]] 证明纯内部信号（包括 entropy）长期训练会 [[reward-hacking|reward hack]]，EMPO 通过 entropy thresholding 缓解但未完全解决

## 面试相关
> 高频 URLVR 面试题

- Q: EMPO 的 reward 是怎么构建的？为什么不需要 ground-truth？🔴
- A: 对每个问题采样 G 个输出，按语义等价性聚类，reward = 所属聚类的频率占比。直觉是"越多独立采样支持的答案越可能正确"。语义熵与准确率强负相关，所以最小化语义熵等价于鼓励正确答案。

- Q: EMPO 和直接用 majority voting 做 reward 有什么区别？🟡
- A: Majority voting 基于字符串精确匹配，会将表述不同但语义相同的答案视为不同。EMPO 用语义聚类（数学用正则提取、自由形式用 SLM 判断双向蕴含），能正确处理同义异形的答案。

- Q: Entropy thresholding 的作用是什么？🟡
- A: 过滤两种极端：(1) 低熵样本——模型已经很确定，继续训练收益小；(2) 高熵样本——模型完全随机，reward 信号太 noisy。只保留中等不确定性的样本，这些是最有学习价值的。

- Q: EMPO 有什么局限性？🔴
- A: PRISM 的实验表明，纯内部信号（包括 entropy/self-certainty）长期训练会 reward hack。EMPO 的 entropy thresholding 是一种缓解但非根本解决。此外，语义聚类的质量依赖于聚类方法（数学用正则、自由形式用 SLM），如果聚类不准 reward 就不准。

## 个人笔记
> EMPO 是 URLVR 中"纯内部信号"流派的代表。核心假设：语义熵是准确率的可靠 proxy。这个假设在短期训练内成立，但 PRISM 的实验显示长期可能失效。实际应用中可能需要结合外部信号（如 PRM）做 stabilization。
>
> 值得注意的发现：EMPO 不是在教模型新能力，而是帮它"找到已有的好路径"。这对理解 RL 在 LLM 中的作用非常重要。
