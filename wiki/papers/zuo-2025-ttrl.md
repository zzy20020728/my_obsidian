---
title: "TTRL: Test-Time Reinforcement Learning"
type: paper
tags: [URLVR, test-time-training, majority-voting, self-evolution, GRPO, unsupervised-RL, Tsinghua, PRIME-RL]
created: 2026-04-08
updated: 2026-04-08
sources: [https://arxiv.org/abs/2504.16084, https://github.com/PRIME-RL/TTRL]
status: active
---

# TTRL: Test-Time Reinforcement Learning

## 基本信息
- **作者**: Yuxin Zuo*, Kaiyan Zhang* (equal contribution, project leads: Kaiyan Zhang & Ganqu Cui), Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding†, Bowen Zhou†
- **机构**: Tsinghua University & Shanghai AI Lab
- **年份**: 2025 (arXiv: Apr 22, 2025; v3: Jun 30, 2025)
- **会议/期刊**: arXiv preprint (arXiv:2504.16084)
- **链接**: https://arxiv.org/abs/2504.16084
- **代码**: https://github.com/PRIME-RL/TTRL

## 一句话总结
> 利用 majority voting 在无标注测试数据上估计 pseudo-label 并计算 rule-based reward，通过 online RL (GRPO) 实现模型 test-time self-evolution，Qwen2.5-Math-7B 在 AIME 2024 上 pass@1 提升 211%，且超越初始模型的 maj@n 上界，逼近使用 ground-truth 直接训练的性能。

## 摘要
TTRL 研究在**没有显式标注**的条件下用 RL 训练 LLM 进行推理任务。核心挑战：推理时如何在无 ground-truth 的情况下估计 reward？TTRL 发现 TTS（Test-Time Scaling）中的 majority voting 可以产生出人意料的有效 reward 信号。通过 repeated sampling → majority voting → rule-based reward → GRPO 更新的循环，TTRL 使 LLM 在 unlabeled test data 上实现自我进化。关键特性：(1) 性能可超越初始模型的 maj@n 上界；(2) 逼近 RL (leakage)——直接在测试数据上用 GT 训练的性能；(3) 泛化到 OOD 任务。

## 核心贡献
1. **Test-Time RL 范式**: 首次将 RL 与 Test-Time Training 结合，在无标注测试数据上通过自身 majority voting 估计 reward 进行 RL 训练
2. **Majority Voting Reward Function**: 极其简洁——对 N 条采样取 majority answer 作为 pseudo-label，匹配的得 reward=1，不匹配得 reward=0
3. **Lucky Hit 机制**: 理论分析为什么即使 label estimation 不准确，reward 准确率仍然很高——错误预测高度分散时，wrong label ≠ wrong prediction → 仍能给出正确的负 reward
4. **超越 maj@n 上界**: TTRL 虽以 maj@n 为监督信号，但 online RL 的动态特性使训练中 voting 质量持续提升，最终 pass@1 > 初始 maj@n
5. **广泛适用性**: 跨 4 个模型家族（Qwen/LLaMA/Mistral/DeepSeek）、base/instruct/LRM 模型、1.5B-32B 规模、3 种 RL 算法（GRPO/PPO/PRIME）均有效

## 方法详解

### 1. 问题定义
在 test time 对 pre-trained model 进行 RL 训练，**不访问 ground-truth labels**：
- 输入：一组 unlabeled questions $\{x_1, ..., x_M\}$
- 目标：通过 RL 更新 $\theta$ 使模型在这些问题上的推理能力提升

### 2. TTRL Pipeline
三阶段循环：

**Stage 1 - Label Estimation（标签估计）**:
给定 question $x$，用当前策略 $\pi_\theta$ 采样 $N$ 条 response：$\{\hat{y}_1, ..., \hat{y}_N\}$，提取答案后进行 majority voting 得到 pseudo-label $y^*$。

**Stage 2 - Reward Calculation（奖励计算）**:
$$R(\hat{y}_i, y^*) = \begin{cases} 1, & \text{if } \hat{y}_i = y^* \\ 0, & \text{otherwise} \end{cases}$$

极简的 binary rule-based reward：答案与 majority vote 一致得 1，否则得 0。

**Stage 3 - Policy Optimization（策略优化）**:
使用 GRPO 更新策略：
$$\max_{\theta} \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(y, y^*)]$$
$$\theta \leftarrow \theta + \eta \nabla_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[r(y, y^*)]$$

关键：这是 **online** 的——每次更新后重新采样、重新 voting，label 质量随模型提升而提升。

### 3. Vote-then-Sample 策略
- 先采样 64 条 response 用于 voting-based label estimation
- 然后 downsample 32 条用于实际训练
- 这种策略有效降低计算成本同时保持强性能

### 4. 超参数关键点
- **Temperature**: 0.6（通用）或 1.0（Math base models & LRMs）——更高 temperature 促进更好的 exploration
- **Episodes**: 根据数据集大小和难度调整——MATH-500 用 10 episodes，AMC 用 30，AIME 2024 用 80
- **Learning rate**: cosine schedule, peak $5 \times 10^{-7}$, AdamW optimizer
- **Max generation length**: 3,072 tokens（通用），32,768 tokens（LRMs）

## 关键理论洞察

### Lucky Hit 机制（为什么 TTRL 在 AIME 等困难任务上也有效）

当 label estimation 准确率仅 37% 时（如 AIME 2024 初期），reward accuracy 仍高达 92%。原因：

> 模型能力弱时，错误预测高度分散（最频繁答案仅占 16.6%），因此即使 pseudo-label 错误，大多数 wrong predictions 也与 wrong pseudo-label 不同 → 仍然收到正确的 negative reward (0)。

形式化：设 pseudo-label $\hat{l}$ 错误（$\hat{l} \neq l^*$），对于一条错误预测 $\hat{y}_i \neq l^*$：
- 如果 $\hat{y}_i \neq \hat{l}$（概率很高，因为预测分散）→ $R(\hat{y}_i, \hat{l}) = 0$ = 正确的 reward
- 如果 $\hat{y}_i = \hat{l}$（概率低）→ $R(\hat{y}_i, \hat{l}) = 1$ = 错误的 positive reward（false positive）

**关键推论**：模型越弱 → 预测越分散 → false positive 越少 → reward accuracy 越高 → self-improvement 越稳定。这构成了一个自稳定的 bootstrap 机制。

### 超越 maj@n 上界的机制

传统 self-training（STaR 等）的上界是初始模型的 maj@n。TTRL 超越这个上界的原因：
1. **Online RL vs Offline SFT**: TTRL 每步重新采样 + 重新 voting，voting 准确率随模型提升而提升
2. **RL 的泛化优势**: RL 不像 SFT 那样记忆训练数据，而是学习更 generalizable 的策略 (Chu et al., 2025)
3. **自强化循环**: better model → better voting → more accurate reward → even better model

## 实验结果

### 主实验（Table 1 & 2）

| 模型 | AIME 2024 | AMC | MATH-500 | GPQA | Avg |
|------|:---------:|:---:|:--------:|:----:|:---:|
| Qwen2.5-Math-1.5B | 7.7 | 28.6 | 32.7 | 24.9 | 23.5 |
| + TTRL | **15.8** (+105%) | **48.9** (+71%) | **73.0** (+123%) | 26.1 (+5%) | **41.0** (+74%) |
| Qwen2.5-Math-7B | 12.9 | 35.6 | 46.7 | 29.1 | 31.1 |
| + TTRL | **40.2** (+211%) | **68.1** (+91%) | **83.4** (+79%) | 27.7 (-5%) | **54.9** (+77%) |
| Qwen2.5-7B | 7.9 | 34.8 | 60.5 | 31.8 | 33.8 |
| + TTRL | **23.3** (+195%) | **56.6** (+63%) | **80.5** (+33%) | **33.6** (+6%) | **48.5** (+44%) |
| Qwen2.5-32B | 7.9 | 32.6 | 55.8 | 33.2 | 32.4 |
| + TTRL | **24.0** (+204%) | **59.3** (+82%) | **83.2** (+49%) | **37.7** (+14%) | **51.1** (+58%) |
| LLaMA3.1-8B-Instruct | 4.6 | 23.3 | 48.6 | 30.8 | 26.8 |
| + TTRL | **10.0** (+117%) | **32.3** (+39%) | **63.7** (+31%) | **34.1** (+11%) | **35.0** (+31%) |

### LRM 模型结果

| 模型 | AIME 2024 | AMC | MATH-500 |
|------|:---------:|:---:|:--------:|
| DeepSeek-R1-LLaMA-8B | 51.7 | 81.6 | 89.6 |
| + TTRL | **69.2** (+17.5) | **88.9** (+7.3) | **90.9** (+1.3) |
| Qwen3-8B (thinking) | ~高基线 | | |
| + TTRL | +~10pts | +~10pts | +~10pts |

### 关键发现
- **AIME 2024**: 所有模型至少 105% 提升
- **跨家族泛化**: Qwen/LLaMA/Mistral/DeepSeek 全部有效
- **LRM 仍可提升**: 即使是 DeepSeek-R1-LLaMA-8B 这样已经过昂贵 post-training 的模型，TTRL 仍能在 AIME 上提升 17.5 点
- **Scaling 自然发生**: 模型越大 → voting 越准 → 提升越大（1.5B→7B→32B 递增）

### TTRL vs RL (Leakage)
在 MATH-500 上，TTRL 的性能曲线几乎逼近使用 ground-truth labels 直接训练的 RL (leakage)，说明 majority voting reward 的有效性接近 perfect reward。

### 难度分级分析（MATH-500）

| Level | Backbone | + TTRL | 提升 |
|-------|:--------:|:------:|:----:|
| L1 (最简单) | 25.9 | 71.2 | +175% |
| L2 | 33.0 | 76.2 | +131% |
| L3 | 36.3 | 76.3 | +110% |
| L4 | 32.5 | 58.7 | +80% |
| L5 (最难) | 22.3 | 39.2 | +75% |

提升随难度递增而递减——prior knowledge 不足以支撑高难度问题的 self-improvement。

### OOD 泛化
在单一 benchmark 上 TTRL 后，在其他 benchmark 上也有实质性提升，说明 TTRL 学到的是 generalizable reasoning skills 而非 overfitting。

## RL 算法兼容性
TTRL 兼容多种 RL 算法，在 MATH-500 上测试：
- **GRPO** (default): Group Relative Policy Optimization
- **PPO**: Proximal Policy Optimization
- **PRIME**: Process-level RL with Implicit Rewards

三者性能轨迹高度一致，说明 TTRL 的核心贡献在于 reward estimation，而非特定 RL 算法。

## 失败场景分析

### 1. 不当超参数
- Temperature 过低（如 0.6 用于 Math base model）→ exploration 不足 → entropy 居高不下 → 训练发散
- Episodes 不足（小数据集 + 高难度）→ 探索不充分

### 2. Prior Knowledge 不足
- 模型在目标任务上的初始能力太弱 → voting 质量过差 → reward 信号噪声过大
- MATH-500 L5 问题上提升（+75%）远低于 L1（+175%），验证了 prior knowledge 的关键作用
- **无 curriculum learning 机制**：TTRL 不做数据过滤，直接面对所有难度的数据

## 与其他工作的关系

### 对比其他 URLVR 方法
| 方面 | TTRL | [[wiki/papers/zhang-2025-empo|EMPO]] | [[wiki/papers/wu-2026-self-judge|Self-Judge]] |
|------|------|------|------------|
| Reward 信号 | Majority voting (binary) | Semantic entropy minimization | Group-wise distributional reward |
| 标注需求 | 仅需 {q} | 仅需 {q} | 仅需 {q} |
| Reward 粒度 | Outcome-level | Outcome-level | Outcome-level |
| RL 算法 | GRPO/PPO/PRIME | [[wiki/concepts/grpo|GRPO]] | GRPO |
| 模态 | Text | Text | Multi-modal |
| 关键优势 | 极简、稳定、可超越 maj@n | 信息论基础、entropy thresholding | 无需对齐分布 |
| 关键局限 | 无 step-level credit | 可能 reward hack 到 low entropy | 计算开销大 |

### 与 [[wiki/papers/wu-2026-spae|SPAE]] 的互补关系
- **TTRL**: outcome-level reward（判断最终答案对错），简洁但无 step-level 信息
- **SPAE**: step-level credit assignment（区分关键步骤与冗余步骤），精细但需 GT
- **互补方案**: TTRL 做 outcome anchor + SPAE 的无监督变体做 step-level scoring → 详见 [[wiki/synthesis/step-level-se-proposal|Step-Level SE 研究方案]]

### 与 [[wiki/papers/ghimire-2026-prism|PRISM]] 的关系
- PRISM 发现纯内部信号（token entropy / trajectory entropy / self-certainty）长期训练不可靠
- TTRL 的 majority voting 是一种**跨样本 consensus 信号**，理论上比单样本内部信号更鲁棒
- 但 TTRL 的 Lucky Hit 机制隐含假设：随训练提升，预测分散度会降低 → reward accuracy 可能下降 → 是否需要额外机制维持？

### TTRL 的 Follow-up 工作（16+篇）
- **SPINE**: 将 TTRL 扩展到半监督设置
- **DARE**: 在 TTRL 基础上引入 difficulty-aware reward estimation
- **SCRL**: Self-Consistency RL，self-consistency 替代 majority voting
- **ETTRL**: Efficient TTRL，减少 sampling 开销
- **Self-Harmony**: 在 TTRL 训练中引入 harmony 正则化

## 面试相关 Q&A

### Q1: TTRL 的核心思想是什么？如何在无标注数据上做 RL？
**A**: TTRL 利用 majority voting 作为 pseudo-label estimation。对 N 条 response 取最频繁答案作为 estimated label，答案匹配得 reward=1，否则得 0。关键洞察：即使 label 不准确，由于错误预测高度分散（Lucky Hit），reward accuracy 仍然很高（92% when label accuracy is only 37%）。

### Q2: 为什么 TTRL 能超越初始模型的 maj@n 上界？
**A**: 传统 self-training (STaR) 用 majority voting 选择好的 CoT 做 SFT，上界就是 maj@n。TTRL 用 **online RL** 而非 offline SFT——每次更新后重新采样和 voting，voting 准确率随模型提升而提升，形成自强化循环：better model → better voting → more accurate reward → even better model。

### Q3: TTRL 什么情况下会失败？
**A**: (1) 模型在目标任务上 prior knowledge 严重不足（如 MATH L5 提升远小于 L1）；(2) 超参数不当（temperature、episodes）；(3) 无 curriculum learning——不做数据难度筛选，直接面对所有难度。

### Q4: TTRL 与 Test-Time Scaling (TTS) 的关系？
**A**: TTRL 同时结合了 TTS（majority voting 提升推理质量）和 TTT（在 test data 上更新参数）。TTS 是 TTRL 的子组件——用于 reward estimation，而 TTT 是 TTRL 的训练范式。

### Q5: TTRL 的 reward 信号有什么局限？
**A**: Binary outcome-level reward（对/错），**无法提供 step-level credit assignment**。模型无法知道哪个推理步骤是关键的、哪个是冗余的。这正是 [[wiki/synthesis/step-level-se-proposal|TTRL + SPAE-SE 双层架构]] 试图解决的问题。

## 个人笔记

### 对研究方案的意义
TTRL 是 [[wiki/synthesis/step-level-se-proposal|双层无监督架构]] 的 **Layer 1 (Outcome)**：
- 提供 outcome-level anchor：majority voting → 判断最终答案对错 → Group Advantage
- 缓解 step-level reward hacking：即使 step-level 信号被 hack，outcome reward 确保错误答案链被惩罚

### 关键启示
1. **简洁有效**: majority voting 这么 trivial 的信号能驱动如此大的提升，说明 pre-trained model 中已有足够的 prior knowledge，只需合适的训练信号就能激发
2. **Online 的重要性**: online RL vs offline SFT 是超越 maj@n 的关键——动态更新 reward 质量
3. **Lucky Hit 的局限**: 随训练进行，模型变强 → 预测更集中 → majority vote 更准但 diversity 降低 → 是否需要额外 exploration 机制？
4. **Step-level 缺失是最大弱点**: TTRL 只有 outcome signal，对长 CoT 推理中间步骤的 credit 是均匀分配的——这正是 SPAE-SE 要解决的问题

### 计算开销
- 64 条采样 + majority voting + 32 条训练，计算量约为标准 GRPO 的 2x
- 在 8×A100 80GB 上实现
- Vote-then-sample 策略有效降低开销

### 与 EMPO 的互斥性
- TTRL reward: 答案是否与 majority vote 一致
- EMPO reward: 语义熵是否降低
- 两者的 reward 信号**正交互补**：TTRL 看「对不对」，EMPO 看「确不确定」
- 理论上可以组合：R = α·R_TTRL + β·R_EMPO，但需要验证是否会 conflict
