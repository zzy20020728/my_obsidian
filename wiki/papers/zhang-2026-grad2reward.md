---
title: "Grad2Reward: From Sparse Judgment to Dense Rewards for Improving Open-Ended LLM Reasoning"
type: paper
tags: [dense-reward, gradient-attribution, open-ended-reasoning, LLM-as-Judge, self-judging, token-level-credit, GRPO, RLVR, process-reward]
created: 2026-04-08
updated: 2026-04-08
sources: [https://arxiv.org/abs/2602.01791]
status: active
---

# Grad2Reward: From Sparse Judgment to Dense Rewards

## 基本信息
- **作者**: Zheng Zhang, Ao Lu, Yuanhao Zeng, Ziwei Shan, Jinjin Guo, Lufei Li, Yexin Li, Kan Ren
- **年份**: 2026
- **会议/期刊**: ICML 2026 (arXiv:2602.01791)
- **提交日期**: 2026-02-02
- **链接**: https://arxiv.org/abs/2602.01791
- **分类**: cs.LG (Machine Learning)

## 一句话总结
> 通过对 Judge 模型的单次反向传播做 gradient-based attribution，将稀疏的 sequence-level reward 分解为 dense token-level reward，实现 open-ended 任务上的细粒度 credit assignment，且支持 self-judging（无需外部更强 Judge）。

## 摘要
RLVR 在数学/编程等可验证领域取得突破，但扩展到 open-ended 任务时面临两大问题：(1) LLM-as-a-Judge 只提供 sequence-level 的稀疏 reward，无法为复杂长文本提供细粒度监督；(2) 现有方法将 Judge 视为黑盒，丢弃了 Judge 内部隐含的过程评价信号。Grad2Reward 通过 gradient-based attribution 从 Judge 的单次 backward pass 中提取 dense token-level process reward，实现精确的 token-level credit assignment。同时引入 self-judging mechanism，用初始 policy 的冻结副本做 Judge，无需更强外部模型即可自我提升。实验证明在多个 open-ended 任务上性能优异，训练效率提升 1.7x-1.9x。

## 核心贡献
1. **首个 open-ended 任务的 dense reward 框架**：解决 credit assignment 问题，大幅提升训练效率
2. **Self-judging mechanism**：用 policy 自身的判别能力做 Judge，无需更强更贵的外部模型
3. **无需训练 PRM**：直接从 Judge 的梯度中提取 process reward，在可验证任务上也优于专门训练的 PRM
4. **Token-level GRPO**：扩展 [[grpo|GRPO]] 到 token-level advantage 估计

## 方法

### 问题定义

#### Open-ended LLM Reasoning as MDP
将 LLM 生成建模为有限horizon MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, r)$：
- **状态**: $s_t = (x, a_{\leq t-1})$，包含 query $x$ 和已生成的 token 序列
- **动作**: $a_t$ 为当前生成的 token
- **转移**: 确定性（拼接已有序列 + 新 token）
- **Reward**: $r(x, o) = \sum_{t=1}^{T} r_t$，其中 $r_t$ 是每个 token 的即时奖励

在 open-ended 任务中，sequence-level reward $r(x, o)$ 可通过 LLM-as-a-Judge 获得，但 individual token reward $r_t$ 未知——**这正是 Grad2Reward 要解决的核心问题**。

#### LLM-as-a-Judge 基线
给定 query $x$、policy output $o$、rubric criterion $c$ 和权重 $w$，Judge 生成 binary decision token $z \in \{\text{True}, \text{False}\}$：

$$r(x, o | c, w) = w \cdot \mathbb{I}[z \sim p_{\text{judge}}(\cdot | x, o, c)] \tag{1}$$

### 4.1 Judge 隐含过程反馈的洞察

**关键观察**：Judge 作为自回归模型，在生成最终判决前逐 token 处理整个 policy output $o$，隐式评估了 logical structure、semantic coherence 和 criterion alignment。但现有方法只提取最终 verdict 作为标量 reward，丢弃了所有中间评价信息。

**核心动机**：恢复并利用隐藏在 Judge 自回归计算中的过程反馈信号。

### 4.2 Self-Judging Mechanism

- **Judge = 初始 policy 的冻结副本**，而非更强的外部模型
- 训练过程中 Judge 固定不变，只提供评价信号；policy 通过 RL 优化
- **理论支撑**：LLM 通常具有比生成能力更强的判别能力（Song et al., 2025）
- **设计优点**：
  - 反馈信号在训练全程保持稳定
  - policy 通过自身判别能力提升，而非蒸馏外部知识
  - 可视为一种 **self-improvement** 范式

### 4.3 Fine-Grained Reward Design（核心方法）

#### Step 1: 计算梯度
设 $\mathbf{e}_t \in \mathbb{R}^d$ 为 policy output 中第 $t$ 个 token 的 embedding。计算 Judge 生成决策 token $z$ 的 log-probability 对每个 token embedding 的梯度：

$$\mathbf{g}_t = \nabla_{\mathbf{e}_t} \log p_{\text{judge}}(z | x, o, c) \tag{2}$$

#### Step 2: Gradient × Embedding 得到 attribution score
通过内积将梯度向量转化为标量 attribution score：

$$b_t = \mathbf{g}_t^\top \mathbf{e}_t \tag{3}$$

#### Step 3: Softmax 归一化
对 attribution scores 做 softmax 归一化，得到每个 token 的相对贡献权重：

$$\alpha_t = \text{Softmax}(\mathbf{b})_t = \frac{\exp(b_t / \tau)}{\sum_{k=1}^{T} \exp(b_k / \tau)} \tag{4}$$

其中 $\tau$ 是温度参数，控制分布的尖锐程度。

#### Step 4: 分解为 token-level reward
将 sequence-level reward 按 attribution weight 分解：

$$r_t = \alpha_t \cdot r(x, o) \tag{5}$$

**整个过程仅需一次 backward pass，不需要 fine-tune Judge。**

### 4.4 理论分析

设 Judge 的目标函数为：

$$F(\mathbf{e}_1, \dots, \mathbf{e}_T) = \log p_{\text{judge}}(z | x, o, c) \tag{6}$$

对每个 embedding 施加扰动 $\Delta \mathbf{e}_t$，一阶 Taylor 展开：

$$F(\mathbf{e}_1 + \Delta\mathbf{e}_1, \dots, \mathbf{e}_T + \Delta\mathbf{e}_T) \approx F(\mathbf{e}_1, \dots, \mathbf{e}_T) + \sum_{t=1}^{T} \nabla_{\mathbf{e}_t} F^\top \Delta\mathbf{e}_t \tag{7}$$

取 $\Delta \mathbf{e}_t = -\mathbf{e}_t$（即移除该 token embedding），得到：

$$F(\mathbf{e}_1, \dots, \mathbf{e}_T) - F(\mathbf{0}, \dots, \mathbf{0}) \approx \sum_{t=1}^{T} \mathbf{g}_t^\top \mathbf{e}_t \tag{8}$$

**物理意义**：每个 $\mathbf{g}_t^\top \mathbf{e}_t$ 近似度量 token $a_t$ 对 Judge 判决的一阶贡献，所有 token 贡献之和近似 sequence-level reward（相对于零 embedding 基线）。

### 4.5 Token-level GRPO

标准 GRPO/DAPO/RLOO 使用 sequence-level reward，所有 token 共享同一 advantage——导致粒度过粗。Token-level GRPO 利用 Grad2Reward 的 token-level reward 实现细粒度优化。

对 query $x$ 生成 $G$ 个 response $\{o_i\}_{i=1}^G$，每个 response 有 token-level rewards $\{r_{i,t}\}_{t=1}^T$。

#### Token-level Return & Advantage

$$R_{i,t} = \sum_{k=t}^{T} r_{i,k}$$

$$\hat{A}_{i,t} = \frac{R_{i,t} - \text{mean}(\{R_{j,s}\}_{j=1,s=1}^{G, |o_j|})}{\text{std}(\{R_{j,s}\}_{j=1,s=1}^{G, |o_j|})} \tag{9}$$

#### 优化目标

$$\mathcal{J}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\} \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left(\rho_{i,t}(\theta) \hat{A}_{i,t}, \; \text{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right) \right] \tag{10}$$

其中 $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}$ 为 token-level importance ratio。

### 完整算法（Algorithm 1）

```
Input: Policy π_θ, Judge p_judge, dataset D, rubric set R, group size G, temperature τ

1. Sample query x ~ D
2. Generate G responses {o_i} ~ π_θ_old(·|x)
3. For each response o_i:
   a. For each rubric item (c_k, w_k):
      - If Judge says True (𝕀[z ~ p_judge(·|x, o_i, c_k)] = 1):
        - Compute token-level gradients: g_{k,t} = ∇_{e_t} log p_judge(z | x, o_i, c_k)
        - Attribution: b_{k,t} = g_{k,t}^T e_t
        - Normalize: α_{k,t} = softmax_t(b_{k,t} / τ)
   b. Aggregate token rewards across rubrics:
      r_{i,t} = Σ_k w_k · α_{k,t} / Σ_k max(w_k, 0)
4. Compute token-level returns R_{i,t} and advantages Â_{i,t} (Eq. 9)
5. Update θ via token-level GRPO objective (Eq. 10)
```

**关键细节**：梯度 attribution 只在 Judge 输出 True 时执行（对失败的 rubric 不做 attribution）。

### A.5 扩展到可验证任务（数学推理）

当使用 discriminative ORM（Outcome Reward Model）$V$ 替代 generative Judge 时，梯度计算变为：

$$\mathbf{g}_t = \nabla_{\mathbf{e}_t} V(x, o) \tag{14}$$

最后一个 token 的 reward 加上全局 ORM 分数，保持 dense attribution 与全局验证的锚定：

$$r_t = \begin{cases} r_t & \text{if } t < T \\ r_t + V(x, o) & \text{if } t = T \end{cases} \tag{15}$$

## 实验

### 设置
- **Policy 模型**: Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct
- **训练**: 全参数微调，lr=1e-6, batch=32, 8 responses/prompt, clip ε=0.2
- **采样**: vLLM, temp=0.7, top-p=0.8, top-k=20, max len=4096
- **测试**: temp=0, top-p=1
- **框架**: veRL (HybridFlow), 8× NVIDIA H20 GPUs
- **Baselines**: Vanilla-GRPO, RuscaRL (Zhou et al., 2025)
- **Test Graders**: Qwen3-30B-A3B-Instruct, Mistral-Small-3.2-24B-Instruct

### 数据集
| 数据集 | 领域 | 训练集 | 测试集 |
|--------|------|--------|--------|
| HealthBench | 医疗咨询 | ~4500 | 500 |
| RaR-Medicine | 医疗问答 | 17,011 | 500 |
| ResearchQA | 学术问答（75个学科） | 16,961 | 500 |
| RaR-Science | 科学问答 | 16,365 | 500 |

### 主要结果（Table 1）

**Test Grader: Qwen3-30B-A3B-Instruct**

| Model | HealthBench | RaR-Med | ResearchQA | RaR-Sci |
|-------|-------------|---------|------------|---------|
| **Qwen2.5-1.5B-Instruct** | 32.2 | 27.7 | 41.2 | 36.2 |
| + Vanilla-GRPO | 39.5 | 32.7 | 53.1 | 41.6 |
| + RuscaRL | 40.7 | 34.3 | 53.9 | 44.1 |
| + **Grad2Reward** | **44.5** | **35.5** | **55.0** | 43.5 |
| **Llama-3.2-3B-Instruct** | 44.4 | 45.3 | 59.9 | 40.0 |
| + Vanilla-GRPO | 46.2 | 48.0 | 63.2 | 41.1 |
| + RuscaRL | 47.4 | 50.5 | 63.5 | 41.4 |
| + **Grad2Reward** | **49.4** | 49.8 | **63.6** | **43.6** |
| **Llama-3.1-8B-Instruct** | 45.5 | 54.9 | 63.3 | 53.6 |
| + Vanilla-GRPO | 47.8 | 56.7 | 65.9 | 54.5 |
| + RuscaRL | 48.6 | 61.1 | 67.0 | 56.2 |
| + **Grad2Reward** | **51.1** | **62.1** | **68.9** | **56.7** |

**关键发现**: Grad2Reward 在所有模型/任务组合上一致优于 sparse reward baselines。对小模型（1.5B）改善尤其明显（HealthBench +3.8 vs RuscaRL）。

### 训练效率分析
- **收敛加速**: Grad2Reward 达到同等性能仅需 Vanilla-GRPO 的 **1/1.7 ~ 1/1.9 训练步数**
- **更高上限**: 在 HealthBench 上相对 Vanilla-GRPO 提升 **13%**，RaR-Medicine 提升 **12%**
- 在 Mistral grader 下同样保持 **12%** 和 **8%** 的稳健提升

### Ablation Study（Table 2）

| 方法 | HealthBench | RaR-Med | ResearchQA | RaR-Sci |
|------|-------------|---------|------------|---------|
| Vanilla-GRPO | 32.2 | 27.7 | 41.2 | 36.2 |
| L₁ norm | 38.7 | 33.1 | **55.6** | 37.8 |
| L₂ norm | 38.0 | 34.5 | 54.9 | 40.0 |
| **Grad × Embed** | **44.5** | **35.5** | 55.0 | **43.5** |

**结论**: 所有梯度方法都优于 sparse reward（证明 dense reward 的有效性），但 Gradient × Embedding 最佳——因为它保留了梯度的方向信息，与 Taylor 展开理论一致。L₁/L₂ norm 只捕获梯度大小，丢弃方向。

### Extended Analysis

#### 5.5.1 Self-Judging vs 外部 Judge（Table 3, RQ1）

| Training Judge | HealthBench | RaR-Med | ResearchQA | RaR-Sci |
|---------------|-------------|---------|------------|---------|
| **Qwen2.5-1.5B (self)** | 44.5 | 35.5 | 55.0 | 43.3 |
| Qwen2.5-7B | 45.3 | 36.6 | 55.6 | 43.1 |
| Llama3.1-8B | 43.7 | 34.5 | 56.8 | 43.6 |
| Qwen3-30B | 45.9 | 36.3 | 55.4 | 43.5 |

**结论**: Self-judging (1.5B) 与 5x-20x 更大的外部 Judge 性能高度可比！证明 policy 自身的判别能力足以提供有效反馈。

#### 5.5.2 与 RLOO 结合（Table 4, RQ2）

| 方法 | HealthBench | RaR-Med | ResearchQA | RaR-Sci |
|------|-------------|---------|------------|---------|
| Vanilla-RLOO | 39.3 | 33.5 | 52.2 | 41.0 |
| RuscaRL | 40.5 | 32.5 | 49.3 | 42.3 |
| **Grad2Reward** | **42.9** | **33.9** | **57.2** | **42.8** |

**结论**: Grad2Reward 的优势不依赖特定 RL 算法，在 RLOO 下同样有效（ResearchQA +5.0 vs Vanilla-RLOO, +7.9 vs RuscaRL）。

#### 5.5.3 跨数据集泛化（Table 6, RQ3）

| 训练集 → 测试集 | Vanilla-GRPO | RuscaRL | Grad2Reward |
|----------------|--------------|---------|-------------|
| RaR-Med → HealthBench | 37.6 | 37.3 | **42.1** |
| RaR-Sci → ResearchQA | 45.1 | 46.2 | **48.6** |
| RaR-Sci → GPQA-Diamond | 24.7 | 25.6 | **26.1** |

**亮点**: 在开放任务上训练后，在可验证任务 GPQA-Diamond 上也有泛化提升，说明 dense reward 提升了输出的结构质量和事实准确性。

#### 5.5.4 vs PRMs（数学推理, Table 5, RQ4）

| 方法 | MATH500 | Minerva | OlympiadBench | AIME25 | AIME24 | AMC23 |
|------|---------|---------|---------------|--------|--------|-------|
| PURE (Cheng, 2025) | 76.0 | 30.8 | 36.7 | 13.3 | **26.6** | **70.0** |
| PRM (Wang, 2024) | 71.6 | 36.3 | 32.5 | 13.3 | 10.0 | 57.5 |
| PQM (Li & Li, 2025) | 72.0 | 34.1 | 34.3 | 13.3 | 13.3 | 52.5 |
| **Grad2Reward** | **77.6** | **36.7** | **38.3** | **16.6** | **26.6** | 65.0 |

**结论**: Grad2Reward 在 6 个数学 benchmark 中的 5 个上取得最高准确率，**无需训练专门的 PRM 即可超越 PRM 方法**。

## 计算成本分析
- **额外开销**: 每个 (response, rubric) 对需要 Judge 的 **1 次 forward pass + 1 次 backward pass**
- **不需要**: 训练任何额外的 reward model（PRM/ORM）
- **不需要**: 多次 forward pass（如 Monte Carlo 采样估计 PRM 标签）
- **Self-judging 优势**: Judge 与 policy 同尺寸（而非更大的外部模型），显著降低推理成本
- **训练加速**: 虽然单步计算略增（backward pass），但 **1.7x-1.9x 更快收敛** 使得总训练成本更低
- **硬件**: 8× NVIDIA H20 GPUs（非顶级 A100/H100，说明方法的可访问性）

## 与其他工作的关系

### 与 [[process-reward-model|PRM]] 方法的区别
- PRM 需要 **step-level 标注数据** 训练专门的 reward model
- Grad2Reward **不需要训练任何 reward model**，直接从 Judge 梯度提取
- PRM 仅适用于可验证任务（需要 ground truth 判断每步对错）
- Grad2Reward 通用于 open-ended 和 verifiable 任务

### 与 RLVR/URLVR 的关系
- 标准 RLVR（如 DeepSeek-R1）: 需要 ground truth，给 outcome reward
- Open-ended RLVR（如 RaR, RuscaRL）: 用 LLM-as-Judge 给 sparse reward
- **Grad2Reward**: 在 LLM-as-Judge 基础上，**将 sparse reward 无损分解为 dense reward**

### 与 [[grpo|GRPO]] 的关系
- 标准 GRPO: sequence-level advantage，所有 token 共享
- **Token-level GRPO**: token-level advantage，每个 token 有独立的 return 和 advantage

## 面试常问点

### 🔴 高频
- **Gradient × Embedding 的直觉是什么？** → 一阶 Taylor 展开，近似度量移除该 token 对 Judge 判决的影响
- **为什么 self-judging 能 work？** → LLM 判别能力 > 生成能力；冻结 Judge 保证稳定反馈
- **Dense reward 为什么比 sparse reward 更好？** → 解决 credit assignment 问题，为每个 token 提供独立的优化信号

### 🟡 中频
- **Gradient × Embedding vs L₁/L₂ norm？** → G×E 保留方向信息，有理论保证（Taylor 展开）
- **如何扩展到 verifiable 任务？** → 用 ORM 替代 Judge，梯度对 ORM scalar output 求，最后 token 加 ORM 分数
- **Token-level GRPO 的 advantage 如何计算？** → Return-to-go $R_{i,t} = \sum_{k=t}^T r_{i,k}$，在 group 内做 z-score 归一化

### 🟢 低频
- **温度参数 τ 的影响？** → 控制 attribution 分布的尖锐程度
- **只在 Judge 输出 True 时做 attribution？** → 是的，失败的 rubric 不做梯度计算

## 关键 Takeaway

1. **"Don't throw away the Judge's internals"**: Judge 不是黑盒，其梯度包含丰富的 token-level 评价信息
2. **Self-judging works**: 不需要更强的外部模型做 Judge，自己就够
3. **Dense > Sparse**: 在 open-ended 任务上，dense token-level reward 显著优于 sparse sequence-level reward
4. **一个框架统一两种场景**: 既适用于 open-ended（Judge-based），也适用于 verifiable（ORM-based）
5. **无需训练 PRM**: 梯度 attribution 作为 process reward 的免训练替代方案
