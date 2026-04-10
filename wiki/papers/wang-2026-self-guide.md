---
title: "Co-Evolution of Policy and Internal Reward for Language Agents"
type: paper
tags: [RLVR, co-evolution, internal-reward, self-guidance, step-level-reward, policy-reward-loop]
created: 2026-04-10
updated: 2026-04-10
sources: ["https://arxiv.org/abs/2604.03098"]
status: active
---

# Self-Guide: Co-Evolution of Policy and Internal Reward for Language Agents

## 一句话总结

提出 Self-Guide——由模型自生成的**内部奖励信号**，同时用于推理时引导和训练时 step-level reward，通过**策略-奖励共演化循环**在三个 agent 基准上超越环境奖励 baseline 约 8%。

## 基本信息

- **作者**: Xinyu Wang, Hanwei Wu*, Jingwei Song*, Shuyuan Zhang, Jiayi Zhang, Fanqi Kong, Tung Sum Thomas Kwok, Xiao-Wen Chang, Yuyu Luo, Chenglin Wu, Bang Liu†
- **机构**: McGill University, McMaster University, HKU, HKUST(GZ), Peking University, UCLA, DeepWisdom, Université de Montréal, Mila
- **发表日期**: 2026-04-03
- **arXiv**: 2604.03098

## 摘要

LLM agent 通过与环境交互学习，但长时序训练受限于稀疏延迟奖励。现有方法通常通过事后信用分配或外部奖励模型应对，但推理时缺乏引导。本文提出 Self-Guide，一种自生成的内部奖励，同时支持推理时引导和训练时监督。Agent 在每步生成简短的自引导信号，指导下一步动作（推理时），同时转换为 step-level 内部奖励（训练时）。这创建了共演化循环：更好的策略产生更好的引导，更好的引导进一步改善策略。在 ALFWorld、ScienceWorld 和 WebShop 三个基准上，Self-Guide 结合 GRPO 比仅用环境奖励的 baseline 平均提升 8%。

## 核心贡献

1. **Self-Guide**: 新型自生成内部奖励，统一推理时引导与训练时奖励，无需外部奖励模型
2. **Stage-wise Trust Schedule**: 梯形调度策略稳定共演化训练
3. **共演化机制验证**: 证明 policy 和 internal reward 必须在线同步进化（offline distillation 不可迁移）

## 方法

### 3.1 问题设定

每个 episode 表示为 $\tau = \{(o_t, z_t, a_t)\}_{t=1}^T$，其中 $o_t$ 为环境观测，$z_t$ 为自引导信号，$a_t$ 为动作。环境仅在终端返回稀疏奖励 $R_{\text{env}}(\tau) \in \{0, 1\}$。

### 3.2 推理时自引导

在每步动作前，模型先生成自引导信号 $z_t$：

$$z_t \sim \pi_\theta(\cdot | h_{t-1}, o_t)$$

然后基于自引导生成动作：

$$a_t \sim \pi_\theta(\cdot | h_{t-1}, o_t, z_t)$$

$z_t$ 是自然语言评估（positive/neutral/negative），判断当前轨迹是否朝正确方向前进。

### 3.3 自引导作为内部奖励

将语言评估映射为标量奖励：

$$r_t^{\text{sg}} = g(z_t)$$

映射规则: positive → +0.1, neutral → 0, negative → -0.1

组合奖励：

$$R(\tau; u) = R_{\text{env}}(\tau) + \lambda(u) \sum_{t=1}^{T} r_t^{\text{sg}}$$

### 3.4 联合优化

使用标准 GRPO，组内归一化优势：

$$\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon}$$

由于 $z_t$ 和 $a_t$ 都由同一 $\pi_\theta$ 生成，单一目标同时训练模型产生更好的引导和更好的动作。

### 3.5 Stage-Wise Trust Schedule（核心设计）

梯形调度策略解决 bootstrap 问题：

$$\lambda(u) = \begin{cases} 0 & \text{Phase I: guidance-only warm-up} \\ 0 \to 1 & \text{Phase II: reward activation (step 40-50)} \\ 1 & \text{Phase III: full internal reward (step 50-70)} \\ 1 \to 0 & \text{Phase IV: late annealing (step 70-80)} \end{cases}$$

**设计原理**:
- **Phase I**: 自引导仅用于动作条件化，不作为奖励。模型先学会生成和使用引导
- **Phase II**: 渐进激活内部奖励，避免优化冲击
- **Phase III**: 共演化循环全力运转
- **Phase IV**: 逐渐退出。因为 self-guidance reward 不是 potential-based 的，长期保留会偏离真实环境目标

## 实验结果

### 主实验

**Qwen3-4B 结果**:

| 方法 | ALFWorld | SciWorld Score | WebShop Score |
|------|----------|----------------|---------------|
| ReAct (prompting) | 20.3 | 12.4 | 31.1 |
| ReAct w/ SG (prompting) | 58.6 | 13.1 | 42.7 |
| GRPO | 86.7 | 51.4 | 84.3 |
| GRPO w/ SG | 91.4 (+4.7) | 60.4 (+9.0) | 87.0 (+2.7) |
| **GRPO w/ SG & GR** | **96.9** (+10.2) | **61.6** (+10.2) | **87.8** (+3.5) |

**Qwen3-1.7B 结果**:

| 方法 | ALFWorld | SciWorld Score | WebShop Score |
|------|----------|----------------|---------------|
| GRPO | 72.7 | 23.5 | 63.0 |
| GRPO w/ SG | 81.3 (+8.6) | 25.3 (+1.8) | 76.7 (+13.7) |
| **GRPO w/ SG & GR** | **89.8** (+17.1) | **28.0** (+4.5) | **79.4** (+16.4) |

**Qwen2.5-7B-Instruct 结果**:

| 方法 | ALFWorld | SciWorld Score | WebShop Score |
|------|----------|----------------|---------------|
| GRPO | 83.6 | 62.7 | 77.5 |
| GRPO w/ SG | 92.2 (+8.6) | 69.4 (+6.7) | 89.3 (+11.8) |
| **GRPO w/ SG & GR** | **95.3** (+11.7) | **72.6** (+11.9) | **90.1** (+12.6) |

### 消融实验: Stage-wise Schedule（WebShop, Qwen3-1.7B）

| 变体 | Success Rate |
|------|-------------|
| Vanilla GRPO | 32.0 |
| Self-guidance Only | 49.2 |
| Immediate Full Reward | 39.7（比 SG-only 差！） |
| Early Entry (step 15) | 49.4 |
| Early Entry (step 25) | 53.8 |
| No Annealing | 53.2 |
| **Full Schedule** | **56.3** |

**关键结论**: 立即使用不成熟的 self-guidance 作为奖励比不用还差（39.7 < 49.2）

### 消融实验: Offline Distillation vs Online Co-Evolution

从 Qwen3-32B 蒸馏 self-guidance 到 Qwen3-1.7B 后：
- 作为 guidance reward → 训练不稳定
- 仅作为 rollout guidance → 无持续收益

**原因**: 分布不匹配。在固定 offline 轨迹上训练的模型无法校准到在线策略的演化 rollout 分布。

### Co-Evolution 改善引导模型本身

固定 policy 为 ckpt40，变化 self-guidance checkpoint:

| SG Checkpoint | Score | Success Rate |
|--------------|-------|-------------|
| ckpt10 | 39.8 | 18.0% |
| ckpt40 | 51.2 | 25.0% |
| ckpt80 | 60.5 | 27.0% |

→ 共演化确实同步提升了 self-guidance 质量。

## 与 SPC/URLVR 研究的关系

### 高度相关: Co-Evolving Verifier 的直接参考

1. **共演化范式的实证验证**: Self-Guide 直接证明了 policy 与内部奖励/verifier 可以在同一训练循环中共同进化，且优于外部固定的奖励模型。这为 Co-Evolving Verifier 的核心假设提供了强有力的实证支持
2. **Stage-wise Trust Schedule**: Co-Evolving Verifier 面临同样的 bootstrap 问题——早期 verifier 不可靠。Self-Guide 的梯形调度（warm-up → activation → full → annealing）提供了一个已验证的解决方案模板
3. **Online vs Offline 的关键发现**: offline distillation 失败表明 verifier/reward 信号必须在策略的在线分布上校准。这对 SPC 的设计有直接启示——probing 信号必须实时反映当前策略的 hidden state 分布

### 与 SPC 的深层连接

- **Self-Guide 的 $z_t$** 类似于 SPC 中 probing 提取的语义一致性信号，但 SPC 通过更细粒度的 hidden state probing 而非语言生成来获取
- **$r_t^{\text{sg}} = g(z_t)$** 的离散映射（+0.1/0/-0.1）过于粗糙，SPC 的连续一致性分数可以提供更精细的信用分配
- **共演化循环**: Self-Guide 的共演化是隐式的（$z_t$ 和 $a_t$ 共享同一模型），而 Co-Evolving Verifier 可以实现更显式的共演化，用 SPC 信号校准 verifier 参数

### 可直接借鉴的设计

- 梯形调度策略 → 应用到 Co-Evolving Verifier 的 trust 控制
- 推理时 + 训练时双重使用信号 → SPC 的 probing 信号也可同时服务于推理时 step selection 和训练时 credit assignment
- 必须在线共演化，不能依赖 offline 预训练的 verifier

## 面试 Q&A

### Q1: Self-Guide 的共演化循环是如何工作的？

**A**: 核心机制是双重角色统一：同一个自引导信号 $z_t$ 既在推理时作为动作条件（引导下一步决策），又在训练时转化为 step-level 内部奖励。由于 $z_t$ 和 $a_t$ 都由同一策略 $\pi_\theta$ 生成，GRPO 的单一优化目标同时改善两者。形成正反馈循环：更强的策略 → 更准确的轨迹 → 更可靠的自引导 → 更精确的内部奖励 → 进一步强化策略。

### Q2: 为什么 Stage-wise Trust Schedule 是必要的？

**A**: 面临 bootstrap 困境：可靠的内部奖励需要成熟的自引导，而成熟的自引导需要充分训练。实验证明直接使用不成熟的自引导作为奖励（Immediate Full Reward: 39.7）比不用还差（Self-guidance Only: 49.2）。梯形调度通过四阶段解决：先让模型只学会生成和使用引导（Phase I），然后渐进激活奖励（Phase II），全力共演化（Phase III），最后退火确保最终策略忠于真实环境目标（Phase IV）。关键原理是 "trust follows competence"。

### Q3: 为什么 offline distillation 的 self-guidance 无法迁移到下游 RL？

**A**: 核心问题是分布不匹配（distribution mismatch）。在固定 offline 轨迹上训练的自引导模型对特定轨迹分布做了校准，但 RL 训练中策略持续演化，rollout 分布不断变化。offline self-guidance 模型无法适应这种分布漂移，导致信号与当前策略不匹配，引起训练不稳定。这也解释了为什么 reward model 经常出现 distribution shift 问题——verifier 必须在策略的在线分布上持续校准。
