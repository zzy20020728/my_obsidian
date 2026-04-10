---
title: "TraPO: A Semi-Supervised Reinforcement Learning Framework for Boosting LLM Reasoning"
type: paper
tags: [semi-supervised-RLVR, URLVR, trajectory-matching, pseudo-label, data-efficiency, GRPO, model-collapse, pass-rate, Qwen, Zhejiang-University, Ant-Group]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2512.13106, https://github.com/ShenzhiYang2000/TRAPO]
status: active
---

# TraPO: A Semi-Supervised RL Framework for Boosting LLM Reasoning

## 基本信息
- **作者**: Shenzhi Yang\*, Guangcheng Zhu\*, Xing Zheng, Yingfan Ma, Zhongqi Chen, Bowen Song, Weiqiang Wang‡, Junbo Zhao‡, Gang Chen, Haobo Wang (\*Equal contribution, ‡Corresponding)
- **机构**: Zhejiang University, Ant Group
- **年份**: 2025 (arXiv: 2512.13106)
- **会议/期刊**: arXiv preprint
- **链接**: https://arxiv.org/abs/2512.13106
- **代码**: https://github.com/ShenzhiYang2000/TRAPO
- **Base Model**: Qwen2.5-Math-7B (主实验), DeepSeek-R1-Distill-Qwen-1.5B, LLaMA-3.1-8B-Instruct

## 一句话总结
> 首次提出半监督 RLVR 范式：用少量标注数据的 pass rate 学习轨迹作为"锚点"，通过轨迹余弦相似度筛选可靠无标注样本纳入训练，仅需 10% 标注数据即超越全量监督 RLVR。

## 摘要

RLVR 方法虽有效但标注成本高；无监督 RLVR（TTRL、EMPO 等）依赖模型内部一致性构造 reward，但后期常出现 model collapse（错误推理模式被不断强化）。本文提出 **半监督 RLVR (SS-RLVR)** 范式：利用小量标注集引导大量无标注样本的 RL 训练。核心技术 **TraPO** (Trajectory-based Policy Optimization) 通过匹配无标注样本与标注样本的 pass rate 学习轨迹相似度来筛选可靠的无标注样本。实验表明：仅用 1K 标注 + 3K 无标注 即超越最强无监督方法（45K 无标注）4.3%；用 4K 标注 + 12K 无标注 即超越全量 45K 标注的监督模型，仅需 **10% 标注数据**。

## 核心贡献

1. **首创半监督 RLVR 范式 (SS-RLVR)**：在监督 RLVR（高标注成本）和无监督 RLVR（model collapse 风险）之间找到最佳平衡点，提出用少量标注数据作为"锚点"引导无标注样本训练
2. **TraPO 算法**：基于 pass rate 学习轨迹的样本选择机制——将标注样本的 pass rate 变化轨迹作为"可靠模板"，通过余弦相似度匹配筛选学习动态一致的无标注样本
3. **极致的数据效率**：10% 标注数据 (4K/45K) 即达成甚至超越全量监督性能；1K 标注 + 3K 无标注 > 45K 无监督
4. **理论保证**：证明 trajectory consistency 作为正则项可以约束 generalization error bound；通过 NTK 分析证明 gradient alignment 导致 trajectory convergence
5. **即插即用**：TraPO 的轨迹选择机制可作为通用组件集成到任何半监督 baseline（TTRL/Entropy/Self-certainty），均带来一致提升

## 方法

### 问题定义：为什么纯无监督 RLVR 会 Model Collapse

无监督 RLVR 的 proxy reward（如 majority voting）基于一个 **关键假设**：higher confidence → greater probability of correctness。

**但这个假设会失效**：当 majority answer 不是正确答案时，错误回答被正向强化 → 模型对错误答案更加自信 → 下一轮投票中错误答案获得更高票数 → **自增强的退化反馈循环**。最终模型收敛到一个"高置信但错误"的状态。

这与 [[wiki/papers/he-2026-urlvr-scale|He et al. (2026)]] 的 **Sharpening Theorem** 完全吻合：所有 intrinsic reward 方法本质上都在 sharpen model's initial distribution，无法纠正初始分布中的系统性偏差。

**关键洞察**：简单混合监督和无监督 RLVR 收效甚微（仅 +0.6%），因为忽视了标注与非标注数据之间的**内在联系**。只有那些"在标注数据上被验证过的推理模式"才应该被纳入无标注数据的 RL 训练。

### 技术方案：Trajectory Similarity Matching

**核心思想**：从"模型学了什么"(what) 转向"模型怎么学的"(how)——用 **pass rate 变化轨迹** 作为连接标注与无标注数据的桥梁。

如果一个无标注样本的 pass rate 变化轨迹与标注样本群体一致（都在上升、且趋势相似），说明模型在这个无标注样本上学到的推理模式可以被标注样本间接验证。

#### 训练流程

```
Phase 1: Warm-up (仅用标注数据)
  ├── 对 D_l ∪ D_u 进行 rollout，计算 pass rate
  ├── 积累 pass rate trajectory（但只用 D_l 做 GRPO 更新）
  └── 初始化 reliable database = {标注样本的轨迹}

Phase 2: Semi-supervised Training (warm-up 后)
  ├── Step 1: 计算平均可靠轨迹 T̄_reliable
  ├── Step 2: 对每个无标注样本计算 TCS (Trajectory Cosine Similarity)
  ├── Step 3: 用 top-p ∪ threshold Γ 选择可靠无标注样本
  ├── Step 4: 将可靠样本的轨迹加入 reliable database
  └── Step 5: L(θ) = J_GRPO^labeled + M ⊙ J_GRPO^unlabeled
```

### 关键公式

#### 1. Pass Rate 定义（标注 vs 无标注）

$$P_q^{(t)} = \begin{cases} \frac{1}{G}\sum_{i=1}^{G}\mathbb{I}(a_i^{(t)} = \tilde{y}_i^{(t)}) & q \in \mathcal{D}_u \text{ (pseudo-label via majority vote)} \\ \frac{1}{G}\sum_{i=1}^{G}\mathbb{I}(a_i^{(t)} = y) & q \in \mathcal{D}_l \text{ (ground-truth)} \end{cases}$$

#### 2. Pass Rate Trajectory

$$\mathbf{T}_q^{(t)} = [P_q^{(1)}, P_q^{(2)}, \dots, P_q^{(t)}] \in [0,1]^t$$

#### 3. Trajectory Cosine Similarity (TCS) — 核心度量

$$\text{TCS}(\mathbf{T}_u^{(t)}, \bar{\mathbf{T}}_{\text{reliable}}^{(t)}) = \hat{\mathbf{T}}_u^{(t)} \cdot \hat{\bar{\mathbf{T}}}_{\text{reliable}}^{(t)} = \sum_{j=1}^{t} \hat{P}_u^{(j)} \cdot \hat{\bar{P}}_{\text{reliable}}^{(j)}$$

其中 $\hat{P}_u^{(j)} = \frac{P_u^{(j)}}{\sqrt{\sum_{i=1}^{t}(P_u^{(i)})^2}}$ 是归一化 pass rate。

#### 4. 样本选择掩码

$$\text{M}(u) = \mathbb{I}(u \in \text{top-p}(\text{TCS})) \lor \mathbb{I}(\text{TCS} \geq \Gamma)$$

结合 top-p（选最相似的 p%）和阈值 Γ（超过阈值的也选）两个准则。

#### 5. 总损失函数

$$\mathcal{L}(\theta) = \mathcal{J}_{\text{GRPO}}^{\text{labeled}}(\theta) + \text{M} \odot \mathcal{J}_{\text{GRPO}}^{\text{unlabeled}}(\theta)$$

#### 6. Generalization Error Bound (Theorem 3.1)

$$\mathcal{R} \leq \underbrace{\mathcal{R}_{\mathcal{D}_l}(\pi_\theta^{(t)})}_{\text{labeled empirical risk}} + \underbrace{\lambda'}_{\text{domain shift}} + \underbrace{\alpha \cdot \mathbb{E}_{q' \sim \mathcal{D}_u}[1 - \text{TCS}(\mathbf{T}_{q'}^{(t)}, \bar{\mathbf{T}}_{\text{reliable}}^{(t)})]}_{\text{trajectory consistency regularizer}} + \underbrace{L_y(1 - \bar{C}^{(t)} + \sqrt{\frac{\ln(2n/\delta)}{2G}})}_{\text{voting confidence term}}$$

**解读**：TCS 越高（轨迹越一致），generalization error 越小；模型自信度 $\bar{C}^{(t)}$ 越低，bound 越松，促使更谨慎的更新。

### 实现细节

| 参数 | 值 |
|------|-----|
| Base Model | Qwen2.5-Math-7B |
| 训练数据 | OpenR1-Math-220k |
| Rollout batch size | 64, 每个 prompt 8 个 rollouts |
| Learning rate | 1e-6 (constant) |
| Temperature (rollout) | T=1.0 |
| Temperature (eval) | T=0.6 |
| KL 系数 β | 0 |
| Entropy 系数 | 0.01 |
| top-p | 0.1 |
| Γ (threshold) | 0.4 |
| Warm-up epochs | 8-10 |
| 硬件 | 8× NVIDIA H200 (141GB) |
| 基础框架 | LUFFY + veRL + vLLM |

## 实验结果

### 主实验 (Table 1): Qwen2.5-Math-7B, 多范式对比

| 模型/方法 | 数据量 | AIME24/25 | AMC | MATH-500 | Minerva | Olympiad | **ID Avg** | ARC-c | GPQA♦ | MMLU-Pro | **OOD Avg** |
|-----------|--------|-----------|-----|----------|---------|----------|------------|-------|-------|----------|-------------|
| Qwen-Base | — | 11.5/4.9 | 31.3 | 43.6 | 7.4 | 15.6 | 19.0 | 18.2 | 11.1 | 16.9 | 15.4 |
| Qwen-Instruct | — | 12.5/10.2 | 48.5 | 80.4 | 32.7 | 41.0 | 37.6 | 70.3 | 24.7 | 34.1 | 43.0 |
| **TTRL (45K 无标注)** | 45K unlab | 14.1/12.7 | 51.5 | 76.6 | 33.8 | 40.3 | 38.2 | 80.5 | 35.4 | 41.3 | 52.4 |
| **Best Unsup (45K)** | 45K unlab | — | — | — | — | — | **38.3** | — | — | — | **52.4** |
| Supervised 1K | 1K lab | 14.2/13.5 | 52.6 | 80.2 | 34.9 | 40.9 | 39.4 | 76.2 | 36.4 | 43.6 | 52.1 |
| Naive SS (best) | 1K+3K | — | — | — | — | — | 40.0 | — | — | — | 52.6 |
| **TraPO** | **1K+3K** | **17.9/13.8** | **58.7** | **81.4** | **38.2** | **45.5** | **42.6** | **83.7** | **37.9** | **46.8** | **56.1** |
| Supervised 4K | 4K lab | 19.6/14.8 | 57.9 | 80.6 | 39.3 | 46.5 | 43.1 | 82.1 | 39.9 | 48.2 | 56.7 |
| **TraPO** | **4K+12K** | **24.3/17.1** | **60.0** | **84.6** | **39.3** | **48.3** | **45.6** | **84.6** | **43.9** | **50.7** | **59.7** |
| **Full Supervised** | **45K lab** | 25.1/15.3 | 62.0 | 84.4 | 39.3 | 46.8 | **45.5** | 82.3 | 40.4 | 49.3 | 57.3 |

**关键数据**:
- **TraPO (1K+3K) vs Best Unsup (45K)**: ID +4.3%, OOD +3.7%（使用不到 10% 的数据量）
- **TraPO (1K+3K) vs Naive SS (best)**: ID +2.6%, OOD +3.5%
- **TraPO (1K+3K) vs Supervised 1K**: ID +3.2%, OOD +4.0%
- **TraPO (4K+12K) vs Full Supervised (45K)**: ID +0.1%, **OOD +2.4%**（仅用 10% 标注）
- TraPO (1K+3K) ≈ Supervised 4K（25% 标注数据效率）

### OOD 无标注数据实验 (Table 2): 1K ID 标注 + 1K OOD 无标注

| 方法 | ID Avg | OOD Avg |
|------|--------|---------|
| Best Unsup (2K, all OOD) | 39.2 | 53.4 |
| Naive SS (best) | 38.6 | 51.9 |
| **TraPO** | **41.0** | **56.9** |
| Supervised 2K | 41.9 | 57.8 |

跨域设置下 TraPO 仅落后全监督 0.9%，大幅超越无监督和朴素半监督方法。

### 多模型验证

| 模型 | 方法 | ID Avg | OOD Avg |
|------|------|--------|---------|
| **LLaMA-3.1-8B** | Unsup (TTRL) | 19.5 | 17.6 |
| | TraPO (SS) | **20.7** | **18.5** |
| | Supervised 2K | 20.2 | 19.3 |
| **DeepSeek-R1-Distill-Qwen-1.5B** | Unsup (TTRL) | 42.8 | 19.2 |
| | TraPO (SS) | **45.3** | **22.6** |
| | Supervised | 47.3 | 32.1 |

LLaMA-3.1-8B 上 TraPO 甚至 ID 超越全监督 +0.5%。

### Scaling Law (Table 12)

在 25% annotation rate 下：

| 数据总量 | ID Avg | OOD Avg |
|----------|--------|---------|
| 1K (0.25K lab + 0.75K unlab) | 40.6 | 48.5 |
| 4K (1K + 3K) | **42.6** | **56.1** |
| 16K (4K + 12K) | **45.6** | **59.7** |
| Full Supervised 45K | 45.5 | 57.3 |

性能随数据量单调递增，16K 时即超越 45K 全监督。

### 超参敏感性

- **top-p**: 最优 0.1（越大→引入更多噪声样本→性能下降）
- **Γ (threshold)**: 最优 0.5（太低→低质量样本、太高→利用不足）
- **warm-up 长度**: 最优 5-8 epochs（太短→伪标签不稳定、太长→轻微过拟合标注集）
- **选择比例**: 最优 30%（top 30% 无标注样本，伪标签准确率最高）

### 训练成本 (Table 7)

| 数据量 | 无监督 | 监督 | TraPO |
|--------|--------|------|-------|
| 4K | ~7×8 GPU-hrs | ~25×8 | ~26×8 |
| 8K | ~13×8 | ~39×8 | ~38×8 |
| 45K | ~11×8 | ~57×8 | ~55×8 |

**TraPO 与监督 RLVR 训练成本几乎相同**，额外的余弦相似度计算可忽略不计。

### 稳定性 (Table 16, 3 runs)

TraPO (Qwen, 1K+3K): ID Avg = 42.8±0.4, OOD Avg = 56.4±0.5 — 非常稳定。

## 与其他工作的关系

### 与 URLVR / [[wiki/papers/he-2026-urlvr-scale|He et al. Sharpening Theorem]] 的关系

TraPO 的成功从实践角度**强烈验证了 He et al. 的核心论断**：

1. **纯 intrinsic reward 存在本质局限**：He et al. 证明所有 intrinsic 方法本质是 sharpening initial distribution，TraPO 的实验也显示简单混合监督和无监督只带来 +0.6% 提升
2. **External supervision 是突破天花板的关键**：TraPO 用极少量标注数据（10%）就超越了全量无监督方法，说明 external ground truth 的引入可以打破"自增强错误共识"的循环
3. **但 TraPO 表明不需要全量标注**：He et al. 的 external reward 方向（self-verification, generation-verification asymmetry）需要领域特定设计，而 TraPO 提供了一种更通用的方案——少量标注即可
4. **Trajectory matching 作为验证 sharpening 方向的工具**：如果一个无标注样本的 pass rate 轨迹与标注样本一致，说明它的 sharpening 方向是正确的（被标注数据间接验证了）

**关键洞察**：TraPO 的成功意味着 URLVR 的根本问题不是"不会学"，而是"没有方向感"——少量外部信号就能提供这个方向。

### 与具体无监督方法的对比
- **vs [[wiki/papers/zuo-2025-ttrl|TTRL]]**: TraPO 用 TTRL 的 majority voting 作为无标注数据的 reward baseline，但通过 trajectory matching 过滤掉了不可靠样本。TraPO (1K+3K) 大幅超越 TTRL (45K)
- **vs [[wiki/papers/zhang-2025-empo|EMPO]]**: EMPO 用 semantic entropy，TraPO 用 trajectory similarity——前者评估"当前状态"，后者评估"学习过程"
- **vs [[wiki/papers/zhang-2025-covo|CoVo]]**: CoVo 看过程中的 consistency + volatility，TraPO 看跨 epoch 的 pass rate trajectory，二者都关注 learning dynamics 但粒度不同

### 方法定位
- **基于**: [[wiki/concepts/grpo|GRPO]]（优化框架）, TTRL/Entropy（无标注 reward）
- **对比**: TTRL, Self-Certainty, Token/Sentence-level Entropy, fully supervised RLVR
- **互补**: TraPO 是通用组件，可集成到任何无监督 reward scheme

## 局限性与开放问题

1. **仅验证了 ≤7B 模型**：受计算限制未测试 13B+ 模型，更大模型可能受益更多（也可能行为不同）
2. **标注数据与模型的对齐前提**：如果标注数据的 RLVR 训练本身收效甚微（如 LLaMA 上的某些情况），则 TraPO 的引导效果也会受限。方法有效的前提是"标注数据能有效训练模型"
3. **未探索标注数据的最优选择策略**：什么类型的标注样本最适合做"锚点"？（类似 active learning 的问题）
4. **Trajectory matching 的局限性**：
   - 依赖 warm-up 积累足够的轨迹长度
   - 过早选择可能不准确（所以需要 warm-up phase）
   - 使用平均轨迹作为参考（而非每个标注样本单独匹配），虽然更稳定但可能损失细粒度信息
5. **仅限数学推理**：虽然 OOD 实验涉及 ARC-c/GPQA/MMLU-Pro，但核心训练仍在数学域，更开放的推理任务（代码、科学、法律等）未验证
6. **与 step-level reward 的结合**：TraPO 完全基于 outcome-level pass rate，未利用 step-level 信号（如 [[wiki/concepts/process-reward-model|PRM]]），可能存在进一步提升空间

## 面试相关
> 这篇论文可能被问到的面试问题

- **Q: TraPO 为什么不直接用标注数据和无标注数据一起训练，而要做 trajectory matching？**
- A: 实验表明朴素组合只带来 +0.6% 提升。原因是无监督 reward 可能给错误推理正向反馈，与监督信号冲突。必须选择性地只用那些"学习模式与标注样本一致"的无标注样本，确保引入的推理模式经过了间接验证。

- **Q: 为什么用 pass rate trajectory 而不是 embedding similarity 来匹配标注和非标注样本？**
- A: LLM 推理任务中每个问题的 solution space 是 instance-specific 的，不同问题的"正确答案"差异很大，embedding 空间的相似性不能反映推理模式的相似性。Pass rate trajectory 捕捉的是"模型怎么学的"而非"学到了什么"，在不同问题间更具可比性。

- **Q: TraPO 的 trajectory matching 和 He et al. 的 sharpening theorem 有什么关系？**
- A: Sharpening theorem 说无监督方法只是在放大初始分布偏好。TraPO 的 trajectory matching 本质上是检测"放大方向是否正确"——如果一个无标注样本的 sharpening trajectory 与有标注验证的样本一致，说明它的放大方向是对的。

## 个人笔记

### 与 SPC 研究方案的关系
TraPO 的 trajectory matching 思想与我们的 [[wiki/synthesis/step-level-se-proposal|SPC 方案]] 有共鸣之处：
- SPC 关注 step-level 的 semantic rollout consistency
- TraPO 关注 epoch-level 的 pass rate trajectory consistency
- 两者都在用"learning dynamics"作为信号，但在不同粒度上

**潜在结合方向**：TraPO 的 trajectory matching 可以作为 sample selection 策略应用到 SPC 框架中——先用 TraPO 筛选可靠样本，再用 SPC 做 step-level reward refinement。

### 关键启示
1. **少量标注 > 大量无标注**：这与 LIMO (Ye et al.) 的发现一致——深度训练少量优质数据优于浅度训练大量数据
2. **"怎么学" vs "学什么"**：TraPO 的核心创新在于 shift from what to how，这个视角对 URLVR 研究有重要启发
3. **Semi-supervised 是 URLVR 的务实出路**：完全无标注理想但不现实（sharpening 限制），完全标注理想但昂贵。10% 标注是一个很好的 sweet spot
