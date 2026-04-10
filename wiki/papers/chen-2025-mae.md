---
title: "Multi-Agent Evolve: LLM Self-Improve through Co-evolution"
type: paper
tags: [multi-agent, self-play, co-evolution, LLM-as-judge, self-rewarding, RLVR, proposer-solver-judge, self-improving]
created: 2026-04-10
updated: 2026-04-10
sources: [raw/papers/2510.23595]
status: active
---

# Multi-Agent Evolve (MAE): LLM Self-Improve through Co-evolution

**arXiv**: 2510.23595
**GitHub**: https://github.com/ulab-uiuc/Multi-agent-Evolve

## 基本信息

| 字段 | 内容 |
|------|------|
| 作者 | Yixing Chen†, Yiding Wang†, Siqi Zhu, Haofei Yu, Tao Feng, Muhan Zhang, Mostofa Patwary, Jiaxuan You |
| 机构 | UIUC (1), Peking University (2), NVIDIA (3) |
| 通讯 | Jiaxuan You (jiaxuan@illinois.edu) |
| 年份 | 2025 |
| Venue | arXiv preprint（未标注会议） |
| 基座模型 | Qwen2.5-3B-Instruct |

## 一句话总结

三个角色（Proposer 出题、Solver 解题、Judge 评分）从同一 LLM 实例化，通过对抗+协作的 RL co-evolution 实现无需人工标注、无需外部 verifier 的通用领域自进化。

## 核心贡献

1. **Multi-Agent Self-Evolving Framework**: 提出 Proposer-Solver-Judge 三角色架构，三者从同一 LLM backbone 实例化，形成闭环 propose→solve→judge pipeline，通过 RL 同步更新共享参数
2. **Domain-Agnostic Self-Rewarding**: 设计了无需 ground-truth 或外部 verifier 的自奖励机制——Judge-based 评估 + difficulty-aware reward + format reward，**摆脱了对 Python 解释器、游戏引擎等 grounded environment 的依赖**
3. **Quality Filtering 稳定训练**: 用 Judge 评估过滤低质量题目（quality score < 0.7 被淘汰），防止 dataset corruption 导致训练崩溃
4. **实验验证**: Qwen2.5-3B-Instruct 上平均 +4.54%，22 个 benchmark 上几乎全面提升，超越 SFT 和 AZR baseline

## 方法详述

### Problem Definition & Motivation

**核心问题**: 能否构建一个不依赖人工标注的 RL 框架，让 LLM 在通用领域实现自我进化？

**现有方法的局限**:
- **标准 RLVR**: 依赖 human-curated dataset + verifiable reward（如数学答案验证、代码执行），不可扩展
- **Self-Play RL（如 AZR）**: 依赖 grounded environment（Python 解释器等），无法泛化到 open-ended domains
- **Zero-Sum Self-Play**: 对抗设定过于限制，不适用于一般推理任务

**MAE 的核心 insight**: 用 LLM-as-a-Judge 替代外部 verifier，用三角色对抗+协作替代简单的双人零和博弈，扩展到一般领域。

### 三个 Agent 的设计

#### 1. Proposer $\pi_P$

**任务**: 生成高质量且有挑战性的题目

**输入**: 生成指令 $I_g$ + 可选的参考题目 $q_{ref}$

$$q \sim \pi_P(\cdot | I_g, (q_{ref}); \theta)$$

**Reward 设计**（三项加权，$\lambda$ 均为 1/3）:

$$R_P(q) = \lambda_{quality} R_{quality} + \lambda_{difficulty} R_{difficulty} + \lambda_{format} R_{format}$$

- **Quality Reward** $R_{quality}$: Judge 对题目质量的评分（清晰度、可解性）
- **Difficulty Reward** $R_{difficulty}$: 对当前 Solver 的难度
  $$R_{difficulty}(q) = 1 - \bar{R}_S(q), \quad \bar{R}_S(q) = \frac{1}{N_{sample}} \sum_{i=1}^{N_{sample}} V_J(a_i, q)$$
  其中 $N_{sample}=5$，采样 5 个 Solver 回答，取 Judge 评分均值的互补
- **Format Reward** $R_{format}$: 输出是否正确包含 `<question>...</question>` 标签（1.0 / 0.5 / 0.0）

**Quality Filtering**: $R_{quality} \geq 0.7$ 的题目才能进入 valid question dataset $\mathcal{D}$

#### 2. Solver $\pi_S$

**任务**: 回答 Proposer 生成的题目

$$a \sim \pi_S(\cdot | I_S, q; \theta)$$

**Reward 设计**（$\lambda_{judge} = \lambda_{format} = 0.5$）:

$$R_S(a) = \lambda_{judge} R_{judge} + \lambda_{format} R_{format}$$

- **Judge Reward** $R_{judge} = V_J(a, q)$: Judge 对 (question, answer) 对的评分
- **Format Reward**: 输出是否正确包含 `<answer>...</answer>` 标签

#### 3. Judge $\pi_J$

**任务**: 评估 Proposer 的题目质量和 Solver 的回答质量，作为 generative reward model

**评判回答**: 采用 chain-of-thought（`<think>` 标签内分析），严格 rubric:
- [1-3]: 任何事实/逻辑/计算错误
- [4-7]: 基本正确但有遗漏
- [8-10]: 完美无误

**评判题目**: 类似 rubric:
- [1-3]: 不可解/自相矛盾/违反常识
- [4-7]: 基本合理但有歧义
- [8-10]: 清晰、well-formed、逻辑自洽

**Judge 自身的 Reward**: 仅 Format Reward——输出是否正确包含 `<score>X</score>` 标签

$$r_J = R_{format}(g)$$

> **关键设计**: Judge 只训 format reward，不训评分准确性。Judge 的评判能力来自 backbone LLM 本身的能力，RL 只确保输出格式可解析。

### Training Pipeline

**RL 算法**: Task-Relative REINFORCE++（来自 AZR），为每个 role 计算独立的 baseline:

$$A^{norm}_{role} = \frac{r - \mu_{role}}{\sigma_{role}}, \quad role \in \{Proposer, Solver, Judge\}$$

**每个训练步的四个阶段**:

1. **Proposer Phase**: 生成题目 → Judge 评质量 → 质量过关则加入 dataset $\mathcal{D}$ → 计算 difficulty reward
2. **Solver Phase**: 从 $\mathcal{D}$ 采样题目 → 生成回答 → Judge 评分 → (q, a) 对加入 pair dataset $\mathcal{P}$
3. **Judge Phase**: 从 $\mathcal{P}$ 采样 → 生成评分 → 自身获得 format reward
4. **Synchronized Update**: 三个 role 的梯度**同步更新共享 backbone**

**核心超参**:
| 参数 | 值 |
|------|-----|
| Max Prompt/Response Length | 8192 |
| Batch Size | 128 |
| Learning Rate | 1e-6 |
| Training Steps | 300 |
| Optimizer | AdamW |
| N samples (difficulty estimation) | 5 |
| PPO Epochs | 1 |
| Entropy Coefficient | 0.001 |
| KL Loss/Reward | False |

### Co-Evolution 机制

Proposer 和 Solver 之间是**对抗关系**:
- Solver 被奖励"正确回答"→ incentivize 提升解题能力
- Proposer 被奖励"难倒 Solver"→ incentivize 出更难的题

但**不是纯零和**:
- Proposer 同时被 Judge 评质量 → 不能出无解/垃圾题
- Quality Filtering 进一步阻止"hack difficulty reward by generating unsolvable questions"

Judge 与两者都是**协作关系**: Judge 的评分同时驱动 Proposer 和 Solver 的改进。

## 实验结果

### 实验设置

四种配置:
| 设置 | 种子数据 | Reference 使用方式 |
|------|---------|------------------|
| MAE (zero) | 16 条模型自生成 | 50% from ref / 50% from scratch |
| MAE (no reference) | 967 条（无 GT） | 100% from scratch |
| MAE (half reference) | 967 条（无 GT） | 50% from ref / 50% from scratch |
| MAE (with reference) | 967 条（无 GT） | 100% from ref |

种子数据: 967 题来自 14 个数据集（GSM8K, MATH, HumanEval, ARC-C 等），**无 ground-truth answer**。

Baseline: Base (Qwen2.5-3B-Instruct), SFT (有 GT), AZR (官方实现, 100 steps)

### 核心数值

#### MAE (zero) vs Base vs AZR

| Benchmark | Base | AZR | MAE (zero) |
|-----------|------|-----|------------|
| MATH | 60.40 | 62.40 | **68.20** |
| GSM8K | 85.20 | 81.20 | **86.00** |
| ARC-C | 80.60 | 82.80 | **84.20** |
| BBH | 53.79 | 52.57 | **57.51** |
| AMC | 39.76 | 34.94 | **44.58** |
| SQuAD | 78.20 | 90.85 | **92.28** |
| TruthfulQA | 45.71 | 44.92 | **52.71** |
| **Overall Avg.** | **55.33** | **57.72** | **58.51** |

MAE (zero) 在几乎不用任何数据的情况下，Overall 超越 AZR +0.79%，超越 Base +3.18%。

#### MAE (half reference) — 最佳配置

| Metric | Base | SFT | MAE (half ref) | Δ vs Base |
|--------|------|-----|----------------|-----------|
| ID Avg. | 63.34 | 63.28 | **68.95** | +5.61 |
| OOD Avg. | 41.32 | 37.41 | **43.96** | +2.64 |
| Overall Avg. | 55.33 | 53.87 | **59.87** | **+4.54** |

亮点:
- MATH: 60.40 → 65.80 (+5.4)
- MMLU: 63.40 → 69.00 (+5.6)
- CQA: 66.80 → 77.20 (+10.4)
- SQuAD: 78.20 → 93.40 (+15.2)
- HellaSwag: 67.80 → 79.00 (+11.2)
- **SFT 使用了 GT answer 反而比 Base 下降**（53.87 vs 55.33），而 MAE 不用 GT 却大幅提升

#### Ablation Study（MAE half reference 设置）

| 消融 | Overall Avg. | Δ |
|------|-------------|-----|
| Full MAE (half ref) | **59.87** | — |
| No Solver training | 57.79 | -2.08 |
| No Proposer training | 57.90 | -1.97 |
| No Judge training | 57.24 | **-2.63** |
| No Quality Filtering | 56.15 | **-3.72** |
| No Format Reward | 59.44 | -0.43 |

关键发现:
- **Judge training 最重要**（-2.63），三个角色都不可或缺
- **Quality Filtering 极其关键**（-3.72），去掉后 dataset corruption 严重
- Format Reward 相对不那么关键（-0.43），因为 Quality Filtering 部分覆盖了格式问题

### Training Stability

- MAE 稳定训练 **>250 步**（batch size 128），远超 R-Zero 的 45 步
- 题目 dataset 持续增长，说明 Proposer 持续产出高质量题目
- 训练中观察到 **Desirable Difficulty Effect**: 当 Proposer 学会出适当难度的题目时，benchmark 性能同步上升

## 与 URLVR / Co-Evolving Verifier 研究的关系

### 与 [[wiki/synthesis/co-evolving-verifier-proposal|Co-Evolving Verifier 方案]] 的对比

MAE 是一个**已发表的、实际实现了 co-evolving 理念的系统**，但与我们的 Co-Evolving Verifier 提案有本质区别:

| 维度 | MAE | Co-Evolving Verifier 提案 |
|------|-----|--------------------------|
| **进化对象** | Proposer + Solver + Judge 全部进化 | Policy + PRM 进化 |
| **Reward 来源** | Judge（LLM 自身）的 generative 评分 | SPC probing → 蒸馏 PRM |
| **Granularity** | **Response-level** | **Step-level** |
| **外部锚点** | **无硬锚点**（Quality Filtering 是软锚） | TTRL outcome anchor + SPC soft anchor |
| **适用领域** | 通用（math, coding, reasoning, QA） | 主要聚焦推理（有 step 结构的任务） |
| **Mutual Sharpening 风险** | **高**（见下文分析） | 中等（有 SPC 校准缓解） |

### Mutual Sharpening 问题分析

MAE **确实面临 mutual sharpening 风险**，这是其设计中最大的理论隐患:

1. **Judge 没有外部校准**: Judge 只训 format reward，其评判能力完全依赖 backbone 的 pretrained knowledge。随着 policy（和 Judge 本身）在 RL 中变化，Judge 的评分标准可能 drift
2. **闭环正反馈**: Solver 生成看似合理的答案 → Judge 给高分 → Solver 被强化 → 答案的"合理外表"被放大，但实际 correctness 无从验证
3. **Quality Filtering 是软约束**: 它能阻止明显的垃圾，但不能阻止"看起来高质量实际错误"的问题

**MAE 的缓解手段**:
- Quality Filtering（$R_{quality} \geq 0.7$ 门槛）
- Proposer-Solver 的对抗关系（难度奖励防止 trivial questions）
- Format Reward 确保输出结构化
- **但没有任何 ground-truth / verifiable anchor**

与 [[wiki/papers/he-2026-urlvr-scale|He et al. 2026]] 的 Sharpening Theorem 对照: MAE 的 Judge 类似于一个没有外部校准的 reward model。当 policy sharpen 时，Judge 也在同一个 backbone 上 sharpen，两者的互相强化可能加速 collapse。MAE 论文的稳定训练（250+ steps）可能归功于:
1. 3B 模型规模较小，collapse 较慢
2. 多角色交互增加了系统多样性
3. Quality Filtering 的过滤效果

### 与其他相关工作的关系

| 工作 | 与 MAE 的异同 |
|------|--------------|
| **AZR (Zhao et al., 2025)** | AZR 用 verifiable environment（Python 解释器）；MAE 用 LLM Judge 替代 |
| **SPIRAL (Liu et al., 2025)** | SPIRAL 是 zero-sum self-play；MAE 引入 Judge 突破 zero-sum 限制 |
| **Self-Rewarding LM (Yuan et al., 2025)** | 类似 self-rewarding 思路；MAE 增加了 Proposer 角色和 adversarial co-evolution |
| **R-Zero (Huang et al., 2025)** | R-Zero 也是 self-evolving，但仅稳定 45 步；MAE 稳定 250+ 步 |
| **Self-Judge (Wu et al., 2026)** | Self-Judge 的 Judge 冻结不训练；MAE 的 Judge 参与 RL 训练（但只训 format） |

## 局限与开放问题

### 明确的局限

1. **仅在 3B 模型上验证**: 没有 7B+ 的实验，作者在 conclusion 中提到"future work includes scaling to larger backbones"
2. **Judge 可靠性无保证**: Judge 的评判准确性完全依赖 pretrained capability，没有与 GT 对比的校准实验
3. **评估方式存疑**: 使用 LLM judge 评估（而非 exact match），评估本身引入了不确定性
4. **300 步后发生什么未知**: 训练只跑了 300 步，长期稳定性未验证
5. **某些 benchmark 有下降**: MAE (zero) 在 MMLU 上下降（63.40→61.40），MAE (with reference) 在 GPQA 上下降
6. **无理论分析**: 没有收敛保证，没有分析 co-evolution 的均衡条件

### 开放问题

1. **Judge 能进化到多强？** Judge 只训 format reward，评判能力的上限在哪里？是否应该加入更强的 Judge 训练信号？
2. **Scaling behavior**: 更大的模型（7B, 14B）上会更好还是更容易 collapse？
3. **与 verifiable reward 的结合**: 如果在可验证领域（math, code）加入 GT reward 作为锚点，能否大幅延长稳定训练窗口？
4. **Step-level extension**: MAE 目前是 response-level reward，能否引入 step-level credit assignment 进一步提升？
5. **多轮迭代**: 当前是单轮 RL 训练，能否做多轮 self-play iteration（类似 SPIN 的迭代模式）？

## 对我们研究的启示

1. **MAE 验证了 co-evolution 的可行性**: 三角色从同一 LLM 出发、无 GT 数据、在 3B 模型上稳定训练 250+ 步并取得 +4.54% 提升，这说明 co-evolving 不是空想
2. **但 MAE 的锚点太弱**: 没有任何形式的 external anchor，长期稳定性存疑。我们的 SPC-anchored 方案在理论上更稳健
3. **Quality Filtering 是关键**: MAE ablation 显示 Quality Filtering 贡献 -3.72%，远超单个 agent 的影响。这对我们的 co-evolving verifier 设计有参考价值——需要类似的数据质量守门机制
4. **Response-level 的天花板**: MAE 在 GPQA (+0 ~ -4) 和 Olympiad (+4.7 最佳) 等高难度 benchmark 上提升有限，可能正是 response-level reward 粒度不够的体现。Step-level 方案（SPC）有进一步提升的空间

---

**Related**: [[wiki/synthesis/co-evolving-verifier-proposal]], [[wiki/synthesis/urlvr-landscape]], [[wiki/papers/wu-2026-self-judge]], [[wiki/papers/he-2026-urlvr-scale]], [[wiki/concepts/reward-hacking]]
