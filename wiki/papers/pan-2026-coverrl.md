---
title: "CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution"
type: paper
tags: [URLVR, co-evolution, generator-verifier, consensus-trap, label-free, self-verification, multi-turn-RL, GRPO, Qwen, Llama]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2603.17775]
status: active
---

# CoVerRL: Generator-Verifier 共演化突破 Consensus Trap

## 基本信息
- **作者**: Teng Pan, Yuchen Yan, Zixuan Wang, Ruiqing Zhang, Guiyang Hou, Wenqi Zhang, Weiming Lu, Jun Xiao, Yongliang Shen
- **机构**: Zhejiang University (浙江大学) & Baidu Inc.
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.17775v2)
- **链接**: https://arxiv.org/abs/2603.17775
- **代码**: https://github.com/ZJU-REAL/CoVerRL
- **Base Model**: Qwen3-1.7B-Base, Llama-3.2-3B-Instruct, Qwen2.5-7B

## 一句话总结
> 单一模型交替扮演 generator 和 verifier 角色，通过 majority voting 为 verifier 提供对比训练信号、verifier 反过来过滤 self-consistent errors，形成共演化良性循环；在 label-free 设定下超越 [[wiki/papers/zuo-2025-ttrl|TTRL]] 4.7-5.9%，同时验证准确率从 ~55% 提升到 85%+。

## 摘要
Label-free RL 通常依赖 majority voting 来获取 pseudo-label，但随着训练推进，模型输出多样性坍缩，系统性错误被高置信地强化——作者将此称为 **consensus trap（共识陷阱）**。CoVerRL 让单一模型在 generator 和 verifier 之间交替切换：generator 通过 majority voting 产生 pseudo-label，verifier 通过审查推理过程来过滤 self-consistent errors，两者互相引导改进。实验表明 CoVerRL 在 Qwen 和 Llama 模型族上超越 TTRL 4.7-5.9%，self-verification 准确率从 ~55% 提升到 85%+，证实两种能力真正共演化。

## 核心贡献
1. **识别 Consensus Trap**: 首次系统性地揭示 majority voting 下的多样性坍缩→reward accuracy 退化的恶性循环，解释了 TTRL 为何最终停滞
2. **Generator-Verifier 共演化框架**: 单一模型交替扮演双角色，在统一的 multi-turn RL 中实现相互引导（bidirectional bootstrapping），无需外部标注
3. **Answer-Anchored GRPO**: 对标准 GRPO 的改进——按目标答案分组而非按前缀分组，更好地捕捉验证路径的多样性
4. **Self-Correction 能力**: 当 verifier 判定错误时触发 generator 修正，训练端到端的自纠正能力

## 方法

### 问题定义：Consensus Trap
[[wiki/papers/zuo-2025-ttrl|TTRL]] 的 majority voting reward 本质上鼓励模型输出更一致的答案。随着训练进行：
- 输出多样性系统性下降
- 当 majority answer 错误时，形成破坏性正反馈循环
- Reward accuracy 从 ~95% 持续下降到 ~80%，label accuracy 停滞在 ~70%

关键洞察：**majority voting 捕捉一致性但无法检测正确性**。当模型反复产出相同的错误答案（self-consistent errors）时，共识机制主动强化这些错误。

### 技术方案

#### 1. Pseudo-Label 生成（Majority Voting + Verification Filtering）

**Step 1 - Majority Voting 初筛**: 给定 query $q$，generator $\pi_\theta^{\text{gen}}$ 采样 $N$ 条 response，最频繁的答案作为 draft pseudo-label $\hat{y}$，划分正集 $Y^+$ 和负集 $Y^-$。

**Step 2 - Self-consistent Verification Filtering**: Verifier $\pi_\theta^{\text{ver}}$ 审查 $Y^+$ 中最多 $M$ 条 response，对每条生成 binary judgment $v_j \in \{0, 1\}$。仅当多数验证为正时保留该 query：

$$\sum_{j=1}^{|\mathcal{V}^+|} v_j > \frac{|\mathcal{V}^+|}{2}$$

此机制专门针对 generator 产出 self-consistent errors 的失效模式。

#### 2. Online Dual-Role Co-Evolution

**Contrastive Verifier Training**: 用 pseudo-label 构造对比样本训练 verifier：
- 正集 $\mathcal{V}^+$：验证 majority answer $\hat{y}$ 对应的 response
- 负集 $\mathcal{V}^-$：选择 $Y^-$ 中最低频答案 $y_s$，生成 $|\mathcal{V}^+|$ 条验证路径
- 强制 $|\mathcal{V}^-| = |\mathcal{V}^+|$ 确保平衡训练（等价于隐式偏好优化）

**Self-Correction**: 对于 $\mathcal{V}^+ \cup \mathcal{V}^-$ 中 verifier 判定错误（$v=0$）的样本，generator 基于 $(q, y, v)$ 生成 $K$ 条修正答案。

#### 3. Reward 设计

$$r = r^f + r^a$$

- Format reward $r^f \in \{0, 1\}$：结构约束
- Accuracy reward $r^a = \mathbb{I}(o = \hat{o})$：对 generation 比较答案，对 verification 比较判断

#### 4. Answer-Anchored GRPO

标准 GRPO 按 query prefix 分组。CoVerRL 的改进：$\mathcal{V}^+$ 中的验证路径虽然前缀不同，但都指向同一 answer anchor $\hat{y}$，因此按答案分组。Advantage 计算：

$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

### 共演化循环
1. 生成 solutions → majority voting 得 pseudo-label
2. Verifier 审查正集 → 过滤 self-consistent errors
3. 构造对比验证数据（$\mathcal{V}^+$, $\mathcal{V}^-$）
4. 收集 self-correction 轨迹
5. Answer-Anchored GRPO 联合更新所有角色

### 实现细节
| 参数 | TTRL | CoVerRL |
|------|------|---------|
| First generation N | 64 | 32 |
| Training samples | 32 | 16 |
| Max Self-verification M | — | 8 |
| Self-correction K | — | 6 |
| Clip_ratio_high | 0.2 | 0.28 |
| LR | $5 \times 10^{-7}$ | $5 \times 10^{-7}$ |
| Max Response Length | 2048 | 2048 |

## 实验结果

### 主实验（Acc.@first / Acc.@final）

| Model | Method | MATH500 | AMC | AIME24 | GPQA | Average |
|-------|--------|---------|-----|--------|------|---------|
| Qwen3-1.7B-Base | Base | 53.5/53.3 | 24.6/24.5 | 3.8/3.3 | 27.5/27.3 | 27.4/27.1 |
| | TTRL | 65.1/65.0 | 31.1/30.9 | 5.2/5.2 | 30.9/30.7 | 33.1/33.0 |
| | **CoVerRL** | **69.0/71.9** | **36.0/38.6** | **9.8/10.6** | **32.9/33.6** | **36.9/38.7** |
| | $\Delta$ | +3.9/+6.9 | +4.9/+7.7 | +4.6/+5.4 | +2.0/+2.9 | **+3.8/+5.7** |
| Llama-3.2-3B | Base | 42.7/41.0 | 17.0/15.7 | 4.6/5.0 | 26.9/26.1 | 22.8/22.0 |
| | TTRL | 52.6/52.2 | 23.8/23.3 | 13.8/14.0 | 29.8/28.2 | 30.0/29.4 |
| | **CoVerRL** | **55.9/59.3** | **28.3/32.2** | **16.3/16.9** | **32.3/32.6** | **33.2/35.3** |
| | $\Delta$ | +3.3/+7.1 | +4.5/+8.9 | +2.5/+2.9 | +2.5/+4.4 | **+3.2/+5.9** |
| Qwen2.5-7B | Base | 50.1/51.4 | 25.5/26.4 | 5.2/6.5 | 29.9/29.7 | 27.7/28.5 |
| | TTRL | 73.8/74.2 | 42.2/42.2 | 12.7/12.5 | 35.8/35.6 | 41.1/41.1 |
| | **CoVerRL** | **76.8/79.6** | **47.6/49.2** | **14.6/17.1** | **36.2/37.2** | **43.8/45.8** |
| | $\Delta$ | +3.0/+5.4 | +5.4/+7.0 | +1.9/+4.6 | +0.4/+1.6 | **+2.7/+4.7** |

### Self-Verification 准确率

| Model | MATH500 (before→after) | AMC | AIME24 | GPQA |
|-------|----------------------|-----|--------|------|
| Qwen3-1.7B-Base | 55.8 → **81.2** | 28.5 → **64.4** | 11.7 → **48.9** | 38.7 → **57.9** |
| Llama-3.2-3B | 57.0 → **77.2** | 50.9 → **76.2** | 55.8 → **75.6** | 56.9 → **55.2** |
| Qwen2.5-7B | 54.0 → **86.5** | 30.9 → **70.9** | 12.9 → **58.8** | 35.7 → **58.4** |

### RewardBench 泛化
| Model | Base | + CoVerRL |
|-------|------|-----------|
| Qwen3-1.7B-Base | 46.8 | **56.4** |
| Llama-3.2-3B-Instruct | 58.6 | **70.0** (超越 GPT-4-Turbo 67.3) |
| Qwen2.5-7B | 49.7 | **60.9** |

### 消融实验（Qwen3-1.7B-Base, Acc.@first / Acc.@final）

| 消融项 | MATH500 | AMC | AIME24 | GPQA |
|--------|---------|-----|--------|------|
| CoVerRL (full) | 69.0/71.9 | 36.0/38.6 | 9.8/10.6 | 32.9/33.6 |
| w/o Ver. Filtering | 67.1/70.3 | 34.6/38.1 | 7.1/9.2 | 29.6/29.4 |
| w/o Ver. Update | 64.6/64.6 | 34.3/34.4 | 5.6/5.6 | 29.1/29.3 |
| w/o Self-Correction | 67.1/68.4 | 35.3/35.2 | 6.7/7.8 | 31.0/25.4 |
| w/o AA-GRPO | 67.7/71.1 | 35.1/38.0 | 8.5/10.2 | 31.2/31.9 |

**关键发现**:
- 冻结 verifier（w/o Ver. Update）导致 Acc.@final ≈ Acc.@first，说明不更新的 verifier 无法提供有效反馈
- 移除 Self-Correction 导致 GPQA 上 Acc.@final（25.4）反而低于 Acc.@first（31.0），说明纠正训练对 OOD 泛化至关重要
- Balanced training ($|\mathcal{V}^+| = |\mathcal{V}^-|$) 是防止 verifier 坍缩的关键——TTRL+PAG（无平衡）导致 Correct Recall 归零

## 与其他工作的关系

### 与 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的直接改进关系
- CoVerRL 保留 TTRL 的 majority voting 框架作为初始信号源，但增加 verifier 作为第二道过滤
- TTRL 的 "lucky hit" 在多样性充足时有效，但随训练进行多样性坍缩后失效——CoVerRL 的 verifier 在此时接管
- CoVerRL 维持 reward accuracy >90%，而 TTRL 从 95% 跌到 ~80%

### 与 [[wiki/papers/du-2026-dual-consensus|DCRL]] 的比较
- 两者都引入"第二视角"来解决 TTRL 的 majority voting 缺陷
- DCRL: 通过 unlearning 构造 explorer 模型（外部多样性）
- CoVerRL: 同一模型内部的 verifier 角色（内部多样性），且 verifier 在训练中持续进化
- CoVerRL 额外获得了可迁移的 verification 能力

### 与 [[wiki/papers/wu-2026-self-judge|Self-Judge]] 的比较
- 两者都试图超越 majority voting 的局限
- Self-Judge 用分布建模保留分布形状信息
- CoVerRL 通过推理级别的 verification 来识别错误，更深层次

## 局限性与开放问题
1. **Multi-turn 计算开销**: 需要额外的 verification + self-correction 回合，总体计算量高于 TTRL
2. **Verifier 初始能力弱**: 早期 verifier 准确率低（~55%），共演化需要足够多的训练步数才能启动
3. **仍依赖 majority voting 作为初始信号**: 对极端场景（所有答案都错）的鲁棒性未充分验证
4. **思考模式依赖**: Non-thinking mode 下 verifier 迅速坍缩为平凡输出，说明需要 reasoning 能力支撑

## 与 SPC/URLVR 研究的关系

### 对 Co-Evolving Verifier 方案的直接启发
CoVerRL 实现了一个与我们 Co-Evolving Verifier 方案高度相关的系统：

1. **共演化验证了可行性**: CoVerRL 证明了"verifier 和 generator 在无标注下共同进化"的核心假设是可行的。Verification accuracy 从 55% 提升到 85%+ 的结果为我们的方案提供了强有力的先验证据。

2. **Balanced Training 的重要性**: TTRL+PAG 的 verifier collapse 实验揭示了一个关键陷阱——不平衡的验证数据会导致 verifier 退化为全拒绝。SPC 方案在设计 verifier 训练时需要特别注意正负样本平衡。

3. **SPC 的差异化优势**: CoVerRL 的 verifier 仍然是 outcome-level 的（判断最终答案对错），而 SPC 通过 probing + semantic consistency 实现 step-level credit assignment，提供更精细的信号。可以考虑将 CoVerRL 的共演化框架与 SPC 的 step-level 信号结合。

4. **Answer-Anchored GRPO 可借鉴**: 按答案分组而非前缀分组的思路，对 SPC 中如何组织多 rollout 的 advantage 计算有参考价值。

## 面试相关

- **Q: 什么是 Consensus Trap？CoVerRL 如何解决？** 🔴
- A: Consensus Trap 指 majority voting 下训练鼓励输出一致性 → 多样性坍缩 → 系统性错误被高置信强化 → reward accuracy 退化的恶性循环。CoVerRL 通过让同一模型交替扮演 generator 和 verifier，利用 contrastive verification 训练 verifier 识别 self-consistent errors，将其从 pseudo-label 中过滤。Verifier 改善 → 更干净的 pseudo-label → 更好的 generator → 更好的 contrastive 信号 → 更好的 verifier，形成良性循环。

- **Q: CoVerRL 的 verifier 如何在没有 ground truth 的情况下学习？** 🔴
- A: 关键在于 contrastive training：将 majority answer 对应的 response 作为正例、最低频答案对应的 response 作为反例，强制平衡采样。虽然 majority answer 可能有噪声，但它提供了足够的对比信号——正集中更大比例是正确推理，负集中更大比例是错误推理。随着训练进行，verifier 从这些"有噪声但有信息量"的对比中逐步提升判别力。

- **Q: 为什么 Balanced Training 对 verifier 至关重要？** 🟡
- A: 论文中 TTRL+PAG 的实验表明，不平衡会导致信息不对称：正集（达到 majority answer）共享相似推理模式，多样性低；负集（各种错误答案）方差大、多样性高。Verifier 很容易过拟合到"拒绝一切"的策略（Correct Recall→0, Wrong Recall→100%），导致灾难性坍缩。平衡采样 $|\mathcal{V}^+| = |\mathcal{V}^-|$ 防止了这种退化，等价于隐式偏好优化。
