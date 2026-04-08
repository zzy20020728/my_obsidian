---
title: "Step Potential Advantage Estimation: Harnessing Intermediate Confidence and Correctness for Efficient Mathematical Reasoning"
type: paper
tags: [RLVR, step-level-credit-assignment, advantage-estimation, efficient-reasoning, over-checking, probing-mechanism, USTC]
created: 2026-04-08
updated: 2026-04-08
sources: [https://arxiv.org/abs/2601.03823, https://github.com/cii030/SPAE-RL]
status: active
---

# SPAE: Step Potential Advantage Estimation

## 基本信息
- **作者**: Fei Wu*, Zhenrong Zhang* (equal contribution), Qikai Chang, Jianshu Zhang, Quan Liu, Jun Du (通讯)
- **机构**: University of Science and Technology of China (USTC) & iFLYTEK Research
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2601.03823)
- **链接**: https://arxiv.org/abs/2601.03823
- **代码**: https://github.com/cii030/SPAE-RL

## 一句话总结
> 提出 training-free probing mechanism 在每个推理步骤边界提取中间 Confidence 和 Correctness 信号，合成为 Step Potential，实现 step-level credit assignment，同时提高准确率（+6.7% AIME2024）并大幅缩短生成长度（-25.1%），有效解决 Over-Checking 和 Right-to-Wrong 失败问题。

## 摘要
RLVR 通过 outcome-based reward 引导 LLM 推理，但 sparse reward 导致 credit assignment 模糊——模型无法区分哪些步骤对解题有贡献、哪些是冗余验证。现有方法（token entropy / sequence-level length control）缺乏语义层面的 step-level progress measure。SPAE 通过 training-free probing 提取每步的 Confidence（entropy-based）和 Correctness（force-fed GT probability），合成为 Step Potential 信号，构建 step-aware advantage estimation：(1) Saturation Penalty 在 potential 饱和后降权 outcome credit 以抑制 Over-Checking；(2) Difference Shaping 放大正向 potential 跃迁、惩罚负向回退。

## 核心贡献
1. **Training-Free Probing Mechanism**: 在每个推理步骤边界插入 probe prompt 诱导模型给出 tentative answer，提取 Confidence（entropy-based）和 Correctness（force-fed GT token probability），无需额外参数或训练
2. **Step Potential 度量**: 将 Confidence 和 Correctness 合成为 [-1, 1] 范围的统一标量，区分三种推理状态：Exploration (Phi ~= 0)、Correct Confidence (Phi -> +1)、False Confidence (Phi -> -1)
3. **Over-Checking 诊断与量化**: 首次形式化 Over-Checking 问题——模型在已解决问题后继续冗余验证，导致 Right-to-Wrong (R2W) 失败
4. **SPAE Advantage Estimation**: 两组件 step-aware credit assignment：Saturation Penalty + Difference Shaping，同时提升准确率和推理效率

## 方法详解

### 1. 问题形式化
给定 query q ~ D，ground-truth y*，模型生成 o = [tau; s]，其中 tau = [tau^1, ..., tau^K] 为 K 个推理步骤（以 ".\n\n" 分隔），s 为 final summary。RLVR 最大化：

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, o \sim \pi_\theta(\cdot|q)}[R(o, y^*)]$$

其中 $R(o, y^*) \in \{0, 1\}$ 为二值 reward。

### 2. Backbone: DAPO + RF-B
综合两者优势：
- **DAPO**: Clip-Higher with decoupled bounds $(\epsilon_{low}, \epsilon_{high})$、Dynamic Sampling（只更新 reward 方差非零的 query groups）、token-level global normalization
- **RF-B**: 全 batch 级别 advantage normalization（而非 GRPO 的 group 内标准化），减少小 group 高方差问题

### 3. Probing Mechanism（训练无关的探测机制）
在每个步骤 $\tau_i^k$ 边界，插入 trigger prompt $p_{\text{probe}}$ = "**Final Answer** \boxed{"，诱导 N=5 个短 continuation（每个最多 10 tokens），提取两个信号：

#### Confidence（置信度）
基于 probe continuation 的 token-level entropy：

$$\text{Conf}(\tau_i^k) = \frac{1}{N} \sum_{n=1}^{N} \exp\left(-\frac{1}{L_n} \sum_{l=1}^{L_n} H_{n,l}\right)$$

其中 $H_{n,l} = -\sum_{v \in \mathcal{V}} p_\theta(v|h_{i,k}, y_{n,<l}) \log p_\theta(v|h_{i,k}, y_{n,<l})$ 为 next-token entropy。低熵 -> 高 Conf，高熵 -> 低 Conf。值域 [0, 1]。

#### Correctness（正确性/准确度）
Force-feeding ground-truth answer tokens，计算条件概率均值：

$$\text{Acc}(\tau_i^k) = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{|y^*|} \sum_{m=1}^{|y^*|} \pi_\theta(y_m^* | h_{i,k}, y_{<m}^*)$$

值域 [0, 1]。这是 continuous score 而非 binary exact matching。

**关键设计**: probe 不引入额外参数、不反传梯度、不修改 policy optimization objective。GT 仅用于提取诊断信号。

### 4. Step Potential（步骤势能）
将 Confidence 和 Correctness 合成为统一标量，受经典 potential-based reward shaping (Ng et al., 1999) 启发：

$$\Phi(\tau_i^k) = 1.5 \cdot \text{Acc}(\tau_i^k) \cdot \text{Conf}(\tau_i^k) + 0.5 \cdot \text{Acc}(\tau_i^k) - \text{Conf}(\tau_i^k)$$

> **注意**: 论文 HTML 渲染中系数显示可能存在歧义（显示为 "5" 和 "5"）。根据论文声明 $\Phi \in [-1, 1]$ 的约束以及三种状态的语义，此处系数应理解为 1.5 和 0.5。

三种推理状态：
| 状态 | Acc | Conf | Phi | 含义 |
|------|-----|------|-----|------|
| Exploration | 低/中 | 低 | ~= 0 | 早期/中期探索，尚未收敛 |
| Correct Confidence | 高 | 高 | -> +1 | 已找到正确答案并自信 |
| False Confidence | 低 | 高 | -> -1 | 自信但错误，幻觉 |

**核心优势**: 纯 Correctness 无法区分「不确定但在探索」vs「自信但幻觉」，必须结合 Confidence。

### 5. Over-Checking 诊断
通过追踪 $\Phi(\tau_i^k)$ 的时序变化，SPAE 将 tokens 分为 **Solving**（Phi 尚未饱和时的推理）和 **Checking**（Phi 饱和后的验证）两个阶段。

**Right-to-Wrong (R2W) 失败**: 模型在 Solving 阶段已得到正确答案，但在 Checking 阶段过度验证，最终翻转为错误答案。Figure 2 量化显示：即使最新的 Qwen3-32B 仍有大量冗余 checking tokens 和 R2W 失败。

### 6. SPAE Advantage Estimation
Token-level advantage 由两部分组成：

$$\hat{A}_{i,j}^{\text{SPAE}} = \underbrace{\hat{A}_i^{\text{Group}} \cdot f(\Phi_{i,\mathcal{M}(j)})}_{\text{Saturation Penalty}} + \underbrace{\xi \cdot g(\Delta\Phi_{i,\mathcal{M}(j)})}_{\text{Difference Shaping}}$$

其中 $\mathcal{M}(j)$ 为 token j 到 step k 的映射。

#### Saturation Penalty f(Phi)
计算步骤 k 之前已饱和的步数：

$$C_{\text{sat}}^{(i,k)} = \sum_{t=1}^{k-1} \mathbb{I}[\Phi(\tau_i^t) > \varepsilon_{\text{sat}}]$$

衰减因子：

$$f(\Phi_{i,k}) = 1 - \alpha(1 - \exp(-C_{\text{sat}}^{(i,k)}))$$

从 1 衰减到 $1-\alpha$，越多饱和步骤 -> 惩罚越强。

#### Difference Shaping g(Delta Phi)

$$\Delta\Phi_{i,k} = \Phi(\tau_i^k) - \Phi(\tau_i^{k-1})$$

Min-Max 归一化后指数放大并 batch 中心化：

$$g(\Delta\Phi_{i,k}) = \exp(\Delta\tilde{\Phi}_{i,k}) - \mathbb{E}_\mathcal{B}[\exp(\Delta\tilde{\Phi})]$$

突出关键的「Aha!」跃迁（大正 Delta Phi），抑制平凡步骤，batch 中心化后赋予负贡献给相对回退。

#### 最终 Advantage
Group-relative outcome advantage（无标准差归一化）：

$$\hat{A}_i^{\text{Group}} = R_i - \text{mean}(\{R_k\}_{k=1}^G)$$

全局 batch 标准化保证稳定性：

$$\hat{A}_{i,j}^{\text{Final}} = \frac{\hat{A}_{i,j}^{\text{SPAE}} - \text{mean}(\hat{A}^{\text{SPAE}} \in \mathcal{B})}{\text{std}(\hat{A}^{\text{SPAE}} \in \mathcal{B}) + \epsilon}$$

## 实验

### 实验设置
- **模型**: DeepSeek-R1-Distill-Qwen-7B, R1-Distill-Llama-8B, Qwen3-4B-Thinking
- **训练数据**: DAPO-MATH-17K
- **框架**: VeRL, off-policy (global batch 640, mini-batch 32)
- **硬件**: 32 x H200 GPUs
- **超参数**: $\xi = \alpha = 0.5$, $N = 5$ probe samples, $\varepsilon_{\text{sat}}$ 为饱和阈值
- **评估**: AIME2024, AIME2025, AMC2023, Minerva-Math, OlympiadBench, GPQA (OOD)
- **指标**: Acc@16, Len@16 (16 runs 平均)
- **Baselines**: DAPO, RF-B, KTAE, Entropy Advantage, DAST*, LC-R1*

### 主要结果（Table 1）

#### DeepSeek-R1-Distill-Qwen-7B

| Method | AIME24 Acc | AIME25 Acc | AMC23 Acc | Minerva Acc | Olympiad Acc | GPQA Acc | Avg Acc | Avg Len |
|--------|-----------|-----------|----------|------------|-------------|---------|---------|---------|
| Base | 52.71 | 36.67 | 88.75 | 53.71 | 69.73 | 52.00 | 58.93 | 8,971 |
| DAST* | 54.37 | 38.54 | 89.69 | 55.63 | 70.89 | 52.68 | 60.30 | 9,088 |
| LC-R1* | 49.38 | 34.58 | 87.50 | 54.71 | 67.26 | 52.27 | 57.62 | 4,598 |
| Entropy | 58.54 | 41.04 | 91.09 | 58.18 | 74.16 | 55.04 | 63.01 | 8,452 |
| KTAE | 48.54 | 37.29 | 89.84 | 56.59 | 69.59 | 53.11 | 59.16 | 8,018 |
| DAPO | 56.25 | 41.46 | 91.72 | 58.30 | 73.12 | 55.52 | 62.73 | 8,213 |
| RF-B | 56.67 | 40.62 | 90.78 | 58.25 | 73.21 | 54.95 | 62.41 | 7,964 |
| **SPAE** | **59.38** | **42.71** | **92.50** | **58.64** | **73.97** | **55.94** | **63.86** | **6,825** |

**亮点**: AIME2024 +6.7% vs Base, 同时 Avg Len -24%

#### DeepSeek-R1-Distill-Llama-8B

| Method | AIME24 | AIME25 | AMC23 | Minerva | Olympiad | GPQA | Avg Acc | Avg Len |
|--------|--------|--------|-------|---------|----------|------|---------|---------|
| Base | 44.58 | 28.96 | 87.66 | 43.15 | 65.50 | 53.31 | 53.86 | 9,671 |
| DAPO | 53.75 | 37.08 | 92.50 | 50.21 | 71.29 | 55.98 | 60.14 | 9,799 |
| RF-B | 51.25 | 36.67 | 92.03 | 50.44 | 71.06 | 55.26 | 59.45 | 9,133 |
| **SPAE** | **53.96** | **36.88** | **92.97** | **49.82** | **71.39** | **56.25** | **60.21** | **8,019** |

**亮点**: Base -> SPAE +6.35%, Avg Len -17.1%

#### Qwen3-4B-Thinking

| Method | AIME24 | AIME25 | AMC23 | Minerva | Olympiad | GPQA | Avg Acc | Avg Len |
|--------|--------|--------|-------|---------|----------|------|---------|---------|
| Base | 71.04 | 63.96 | 94.84 | 61.35 | 79.55 | 51.17 | 70.32 | 10,903 |
| DAPO | 71.04 | 64.79 | 93.75 | 61.53 | 79.07 | 49.93 | 70.02 | 7,746 |
| RF-B | 68.96 | 61.67 | 94.06 | 61.65 | 79.11 | 50.93 | 69.40 | 7,849 |
| **SPAE** | **71.88** | **65.21** | **96.09** | **61.28** | **79.28** | **50.86** | **70.77** | **7,250** |

**亮点**: SPAE 是唯一在 Qwen3-4B 上超越 Base 的方法 (+0.45%)，其他 RLVR baselines 均退化

### 消融实验（Table 2, Qwen-7B）

| Setting | Avg Acc | Avg Len | 分析 |
|---------|---------|---------|------|
| SPAE (Full, alpha=0.5, xi=0.5) | **63.86** | 6,825 | 最佳 accuracy-efficiency trade-off |
| w/o Conf in Potential | 62.43 (-1.43) | 6,967 (+142) | 纯 Acc 无法区分不确定探索 vs 自信幻觉 |
| w/o Difference Shaping (xi=0) | 62.47 (-1.39) | 6,773 | step-wise progress 指导的关键作用 |
| w/o Saturation Penalty (alpha=0) | 63.98 (+0.12) | 7,517 (+692) | penalty 主要作用是抑制 Over-Checking 而非提升准确率 |
| Baseline (RF-B) | 62.41 | 7,964 | 参考基线 |

超参数敏感性：$\xi, \alpha \in \{0.1, 0.5, 1.0\}$，最佳为 $\xi = \alpha = 0.5$。过弱 shaping 指导不足，过强 shaping 过度干预。

### Reasoning Behavior 分析（Table 3, AIME2024 & 2025）

| Method | Acc | Solve Tokens | Check Tokens | Reflect Steps | R2W Rate |
|--------|-----|-------------|-------------|---------------|----------|
| Base | 46.04 | 4,354 | 1,511 | 17.72 | 8.10% |
| DAST | 46.46 | 3,970 | 1,424 | 16.56 | 10.31% |
| LC-R1 | 41.98 | 2,828 | 299 | 5.53 | 4.67% |
| DAPO | 48.86 | 4,581 | 1,787 | 19.05 | 9.50% |
| RF-B | 48.65 | 4,414 | 1,313 | 15.58 | 6.29% |
| **SPAE** | **51.05** | 3,483 | **614** | **9.08** | **2.65%** |

**核心发现**:
- SPAE Check tokens 仅 614（Base 的 41%，-59%），精准剪除冗余验证
- R2W 率从 Base 8.10% 降至 2.65%（-67%），有效防止正确答案被翻转
- LC-R1 虽然也大幅缩短（Check=299），但准确率大降（-4.06%），属于无差别截断

### 训练效率
尽管 probing 引入额外计算成本，SPAE 在相同 wall-clock 预算下达到更高准确率（Figure 5），因为生成长度的缩短节省了 rollout 时间。

## Step Potential 可靠性验证（Appendix C）

### C.1 与 Oracle 的时序对齐
使用 Qwen3-235B-A22B 作为 oracle teacher 标注 ground-truth solving step k_GT：
- **Pr(Delta k = 0) = 86.0%** — probe saturation 与 oracle 完全同步
- Pr(Delta k > 0) = 3.5%, Pr(Delta k < 0) = 10.5% — 不匹配时以提前触发为主
- E[|Delta k|] = 3.86 — 偏差小

### C.2 Oracle Truncation Test
在 probe saturation 后立即截断（插入 </think>），强制进入 final summary：
- Acc@16: 46.04 -> 48.44 (+2.40)
- Len@16: 13,765 -> 12,931
- R2W: 5.4% -> 0.0%（完全消除）
- **结论**: 饱和后生成确实大部分是冗余的，且可能有害

### C.3 方差与稳定性分析
probe 信号跨 run 的方差较小，Step Potential 是稳定的诊断信号。

## Related Work
1. **RL for LLM Reasoning**: PPO -> RLVR (ReMax, RF-B, GRPO)，sparse terminal reward 限制了 credit assignment
2. **Token-Level Advantage Estimation**: entropy-based (SEED-GRPO, Entropy Advantage)、statistical (KTAE)，但缺乏语义层面的 step-level progress 估计
3. **Efficient Reasoning**: length-aware objectives (DAST, LC-R1, AdaptThink)，但序列级操作无法区分有效推理与冗余

## GitHub 实现细节
- **仓库**: https://github.com/cii030/SPAE-RL
- **License**: Apache 2.0
- **关键文件**:
  - `inference/probe_mechanism_example.py` — probing mechanism 示例
  - `train/recipe/` — VeRL 训练修改
- **依赖**: VeRL framework, vLLM

## 与其他工作的关系
| 对比维度 | SPAE | [[wiki/papers/zhang-2026-grad2reward|Grad2Reward]] | [[wiki/papers/rahman-2025-spark|SPARK]] | Entropy Advantage |
|---------|------|-------------|-------|-------------------|
| 信号粒度 | Step-level | Token-level | Step-level (PRM) | Token-level |
| 信号来源 | Probing (Conf + Acc) | Gradient attribution | PRM (trained) | Token entropy |
| 需要训练 | 否 (training-free) | 否 | 是 (PRM 训练) | 否 |
| 需要 GT | 是 (Correctness probe) | 否 | 否 (self-consistency) | 否 |
| 解决的问题 | Over-Checking + R2W | Open-ended task reward | Dense step reward | 探索-利用平衡 |
| 效率提升 | 显著 (-25% length) | 无 | 无 | 有限 |

## Limitations
1. **训练额外计算**: probing 在每个步骤边界引入额外推理开销，未来方向是轻量/自适应 probing
2. **依赖结构化答案**: Correctness probe 假设 GT 可 tokenize 并 force-feed，不适用于 free-form 输出
3. **仅验证于数学推理**: 尚未扩展到 code generation 和其他推理领域

## 面试要点

- Q: SPAE 的核心思想是什么？为什么需要 step-level credit assignment？ 🔴
- A: RLVR 的 sparse outcome reward 导致所有 token 共享同一 advantage，无法区分关键推理步骤与冗余验证。SPAE 通过 training-free probing 在每步提取 Confidence 和 Correctness，合成 Step Potential 信号，实现 step-aware advantage estimation。

- Q: Step Potential 怎么计算？为什么同时需要 Confidence 和 Correctness？ 🔴
- A: Phi = f(Acc, Conf) 映射到 [-1, 1]。纯 Correctness 无法区分两种关键状态：(1) 低 Conf + 低 Acc = 正常探索 (Phi ~= 0)；(2) 高 Conf + 低 Acc = 自信但错误 (Phi -> -1)。加入 Confidence 后才能区分这两种状态，避免错误地惩罚正常探索。

- Q: Over-Checking 和 R2W 是什么？SPAE 怎么解决？ 🔴
- A: Over-Checking 指模型在已解决问题后继续冗余验证。R2W 是 Over-Checking 的极端后果——过度验证导致模型翻转正确答案。SPAE 的 Saturation Penalty 在 Step Potential 饱和后指数衰减 outcome advantage，鼓励模型及时终止。消融显示去掉 Saturation Penalty 后长度增加 692 tokens。

- Q: SPAE 和 Grad2Reward 的区别？ 🟡
- A: Grad2Reward 用 gradient attribution 在 token level 提取 dense reward，面向 open-ended tasks 且不需要 GT；SPAE 在 step level 操作，利用 probing 提取语义进度信号，面向数学推理，需要 GT 计算 Correctness，但额外解决了 Over-Checking 问题。

- Q: Probing mechanism 的计算开销怎么样？ 🟡
- A: 每个步骤边界需要 N=5 次短 continuation（各 10 tokens）+ 1 次 GT force-feeding。尽管增加训练时间，但因为 SPAE 显著缩短生成长度，总体 wall-clock 效率反而更优（Figure 5）。

- Q: SPAE 的局限性？ 🔴
- A: (1) 依赖 GT 答案计算 Correctness，无法用于无标注数据——与 [[wiki/papers/zhang-2025-empo|EMPO]] 等 URLVR 方法互补而非替代；(2) 假设结构化答案，不适用于 free-form 输出；(3) 仅验证于数学推理。

## 个人笔记
> SPAE 是 USTC & iFLYTEK 的工作，和我们实验室方向高度相关。核心亮点是将 probing 作为「语义传感器」，不需要额外训练就能获得 step-level 进度信号。这解决了 RLVR 中一个很实际的问题：模型做对了还在继续验证，越验证越错。
>
> 关键设计洞察：Confidence + Correctness 的组合是必要的，缺一不可。消融实验清楚地表明去掉 Confidence 后 Avg Acc 降 1.43%。
>
> 与 URLVR 方向的关系：SPAE 需要 GT 答案做 Correctness probe，所以不属于 URLVR。但 probing mechanism 本身可以启发 URLVR——如果能用 self-consistency 或其他无监督信号替代 Correctness probe，就可以将 step-level credit assignment 引入 URLVR。
>
> 值得注意：SPAE 在 Qwen3-4B 上是唯一超越 Base 的 RLVR 方法，说明对已经很强的模型，step-level credit assignment 比粗粒度 RLVR 更重要。
