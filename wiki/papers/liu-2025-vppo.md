---
title: "Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning"
type: paper
tags: [RLVR, process-reward-model, credit-assignment, first-error-detection, GRPO, step-level-reward, reward-shaping, Tencent]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2601.18984]
status: active
---

# VPPO: Verifiable Prefix Policy Optimization

## 基本信息
- **作者**: Haolin Liu¹², Dian Yu¹, Sidi Lu¹, Yujun Zhou¹³, Rui Liu¹⁴, Zhenwen Liang¹, Haitao Mi¹, Chen-Yu Wei², Dong Yu¹
- **机构**: ¹Tencent AI Lab (Seattle), ²University of Virginia, ³University of Notre Dame, ⁴University of Maryland
- **年份**: 2025
- **会议/期刊**: arXiv preprint (arXiv:2601.18984)
- **链接**: https://arxiv.org/abs/2601.18984
- **代码**: 基于 OpenRLHF 框架实现

## 一句话总结
> 提出 VPPO，将 PRM 仅用于定位错误轨迹中的第一个错误步骤，将轨迹划分为"好前缀"和"错误后缀"，对前缀给予正向奖励、对后缀精准惩罚，解决稀疏 outcome reward 的 credit assignment 问题，在 Pass@1 和 Pass@K 上全面超越 GRPO 和现有 PRM-guided RL 方法。

## 核心贡献

1. **PRM 角色重定义**: 不再将 PRM 的逐步分数作为 reward 直接优化（noisy & biased），而是仅利用 PRM 检测第一个错误步骤的能力——这一任务有明确 ground truth、有专门 benchmark (ProcessBench) 可评测
2. **Good Prefix 发现**: 实验表明 88% 的错误响应在第一个错误前至少有 1 个正确步骤，平均 good prefix 占总步骤的 34%；从 good prefix 续写可以恢复出原本 Pass@32=0 的正确解
3. **VPPO 奖励机制**: 正确响应仅在末尾 token 给 reward=1；错误响应在 good prefix 末尾 token 给 reward=α（0<α≤1），其余 token reward=0，形成精准的 exploration + exploitation 学习信号
4. **Shorten Prefix 策略**: 发现 simple prefix 会导致 step inflation（模型拆分步骤以增加 rewarded tokens），提出截短前缀策略消除该 reward hacking 行为
5. **理论保证**: 在树结构 MDP 中证明 VPPO 相比 sparse reward 有指数级样本复杂度改进：$\mathbb{E}[N_{\text{sparse}}^*] \geq \tilde{\Omega}(2^H/H)$ vs $\mathbb{E}[N_{\text{dense}}^*] \leq \tilde{\mathcal{O}}(H)$

## 方法详解

### 1. 问题动机：为什么 sparse outcome reward 不够

标准 GRPO 仅在最后一个 token 给 binary reward：正确=1，错误=0。这导致：
- **Credit assignment 模糊**: 错误响应中的正确步骤被一同惩罚
- **梯度冲突**: 两条轨迹共享正确前缀但答案不同时，正确轨迹鼓励该前缀、错误轨迹惩罚该前缀，信号相互抵消
- **梯度消失**: 当所有采样轨迹都错误时（难题），policy 完全无学习信号

### 2. 为什么不能直接用 PRM 分数做 reward

现有方法（Zou et al. Mixed / Chen et al. RTS）将 PRM 逐步分数用于构造 response-level reward：
- **Mixed** (Zou et al., 2025): $r_{\text{mix}} = \frac{\lambda}{M}\sum_m r_{\text{prm}}(s_m) + (1-\lambda) r_{\text{corr}}$，即 PRM 平均分 + correctness 混合
- **RTS** (Chen et al., 2025): 用大模型找第一个错误步骤，计算正确步骤比例 $\text{RTS} = G/M$，通过 sigmoid 映射为 reward

两者的共同问题：**所有 token 仍共享同一个 advantage 值**，无法区分正确前缀和错误后缀。

### 3. VPPO 核心设计

#### Step 1: First Error Detection

使用 Qwen2.5-Math-PRM-7B 对每步打分，阈值 0.8 判断正确/错误（在 ProcessBench 上选择），第一个低于阈值的步骤被标记为第一个错误步骤 $M_{\text{err}}$。

PRM 检测准确率验证（与 GPT-o1 对比）：
| 类型 | Match | Less | More | Fail |
|------|-------|------|------|------|
| 比例 | 60.3% | 25.0% | 9.3% | 3.3% |

其中 "Less"（PRM 比 GT 更保守）和 "Fail" 都是安全的——仅 "More"（9.3%）会误奖错误步骤。在 ProcessBench 上，threshold=0.8 时 "Not More" 比例达 92.7%。

#### Step 2: Reward Assignment (Eq. 6)

$$r(q, o_{<t}, o_t) = \begin{cases} 1 & \text{if } o \text{ is correct and } o_t = \text{LT}(o) \\ \alpha & \text{if } o \text{ is incorrect and } o_t = \text{LT}(\mathcal{RP}(q,o)) \\ 0 & \text{otherwise} \end{cases}$$

其中 $\mathcal{RP}(q,o) \subseteq \text{Pref}(\mathcal{GP}(q,o))$ 是 reward prefix（good prefix 的一个前缀子集），$\alpha \in [0,1]$ 控制好前缀的奖励强度。

#### Step 3: Advantage Estimation

**Token-level advantage:**
$$A_{i,t}(q) = \begin{cases} 1 & \text{if } o^i \text{ is correct} \\ \alpha & \text{if } o^i \text{ is incorrect and } o^i_t \in \mathcal{RP}(q, o^i) \\ 0 & \text{otherwise} \end{cases}$$

**Response-level 平均 advantage:**
$$A_i(q) = \begin{cases} 1 & \text{if correct} \\ \frac{\alpha |\mathcal{RP}(q, o^i)|}{|o^i|} & \text{otherwise} \end{cases}$$

**归一化 (Eq. 7)** — 去除标准差（因为 token-level advantage 非均匀，std 可能极小导致数值爆炸）：
$$\hat{A}_{i,t}(q) = A_{i,t}(q) - \text{mean}(A_1(q), \ldots, A_G(q))$$

**可选 ReLU (Eq. 8)** — 对强模型（instruct-tuned），好前缀 advantage 加 ReLU 保护：
$$\hat{A}_{i,t}(q) = \max\{A_{i,t}(q) - \text{mean}(\ldots), 0\} \quad \text{(for good prefix tokens of incorrect responses)}$$

### 4. Shorten Prefix 策略（防止 Step Inflation）

**问题**: Simple prefix 直接奖励 $\mathcal{GP}$ 全部 token，模型会将一个大步骤拆成多个小步骤来增加 rewarded token 数量（reward hacking）。

**解决**: 截短 good prefix，留出一段"缓冲区"不给奖励：
$$\mathcal{RP}_{\text{short}}(q,o) = (o_1, \ldots, o_{t_{M_{\text{err}}} - c(q,o) - 1})$$

三种截短方式：
| 方式 | 截短长度 $c(q,o)$ | Average@16 |
|------|-------------------|------------|
| Fixed | 200 tokens | 34.9 |
| Partial | $0.1 \times |o|$ | 34.5 |
| **Prompt** | prompt 长度 | **35.4** |

最佳方案：**Prompt 方式**（用 prompt token 长度作为截断量），效果最好。

### 5. 两种改进场景的直觉

**Exploration 场景**: 两条轨迹走不同路径，一条对一条错。Sparse reward 惩罚错误轨迹的所有步骤；VPPO 保留错误轨迹的好前缀，惩罚仅第一个错误步骤之后，隐式鼓励模型探索好前缀之后的替代路径。

**Exploitation 场景**: 两条轨迹共享前缀但最终答案不同。Sparse reward 对共享前缀产生梯度冲突；VPPO 识别出共享前缀是好前缀，两条轨迹都对其正向更新，消除冲突。

### 6. 理论结果 (Theorem 1)

在 H 层二叉推理树（唯一正确路径）中：
- **Sparse reward**: 样本复杂度 $\mathbb{E}[N_{\text{sparse}}^*] \geq \Omega(2^H / \eta H)$（指数级）
- **VPPO reward**: 样本复杂度 $\mathbb{E}[N_{\text{dense}}^*] \leq \mathcal{O}(H / \eta\alpha \cdot \log(H/\epsilon))$（线性级）

关键 insight：VPPO 可以逐层学习——即使轨迹整体错误，只要前 h 步正确就能更新前 h 步的 policy，无需等到整条路径正确。

## 实验结果

### 实验设置
- **模型**: Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-4B (non-thinking instruct)
- **训练数据**: axon-rl MATH training data (lvl3-5, 8K)
- **PRM**: Qwen2.5-Math-PRM-7B (阈值 0.8)
- **框架**: OpenRLHF
- **超参**: α=0.5, n_samples_per_prompt=8, lr=5e-7, eps_clip=0.2, temp=1.0, 5 epochs
- **评估 benchmarks**: AIME-25, AIME-24, AMC-23, MATH-500, Minerva, Olympiadbench, Hmmt-feb-2024, Hmmt-feb-2025
- **计算成本**: PRM 推理仅增加约 1.1× GRPO 的 per-step runtime；4B 模型约 384 GPU-hours，8B 约 768 GPU-hours

### Pass@1 (Average@16) 主要结果

| 方法 | Qwen3-4B-Base Avg | Qwen3-8B-Base Avg | Qwen3-4B Avg |
|------|-------------------|-------------------|--------------|
| Base | 20.2 | 16.8 | 37.1 |
| GRPO | 33.7 | 36.6 | 42.9 |
| Mixed | 31.8 | 35.2 | 39.8 |
| RTS | 32.4 | 35.0 | 43.1 |
| **VPPO (ours)** | **35.4** | **38.0** | **44.8** |

**关键发现**:
- VPPO 在所有三个模型上全面超越 GRPO 和 PRM 增强方法
- Mixed 方法在所有模型上反而低于 GRPO
- RTS 在 base 模型上低于 GRPO，仅在 instruct 模型上有优势
- VPPO 是唯一在 base 和 instruct 模型上都稳定优于 GRPO 的方法

**竞赛级难题表现** (AIME-25):
- Qwen3-4B-Base: GRPO 11.9 → VPPO **16.6** (+4.7)
- Qwen3-8B-Base: GRPO 17.2 → VPPO **18.3** (+1.1)
- Qwen3-4B: GRPO 26.0 → VPPO **29.2** (+3.2)

### Pass@K 结果 (AIME-24 + AIME-25 average, n=256)

| 方法 | Pass@1 | Pass@8 | Pass@32 | Pass@128 |
|------|--------|--------|---------|----------|
| **Qwen3-4B-Base** | | | | |
| Base | 6.5 | 21.2 | 33.0 | 47.0 |
| GRPO | 15.4 | 28.8 | 38.3 | 49.9 |
| PassK (专门优化) | 12.2 | 24.9 | 37.3 | **54.6** |
| **VPPO** | **17.4** | **32.3** | **42.1** | 53.4 |
| Δ(vs GRPO) | +2.0 | +3.5 | +3.8 | +3.5 |
| **Qwen3-8B-Base** | | | | |
| GRPO | 19.1 | 32.2 | 41.4 | 53.3 |
| PassK | 18.8 | 35.3 | 46.5 | 56.7 |
| **VPPO** | **20.5** | 33.1 | 45.9 | **59.9** |
| Δ(vs GRPO) | +1.4 | +0.9 | +4.5 | +6.6 |

**关键发现**: VPPO 在 Pass@K 上可媲美专门的 Pass@K 优化算法 (Chen et al., 2025b)，同时保持显著更好的 Pass@1。

### 消融实验

**α 敏感性** (Qwen3-4B-Base):
| α | Avg@16 |
|---|--------|
| 0.3 | 35.3 |
| 0.5 | **35.4** |
| 0.7 | 35.4 |
| 0.9 | 33.0 (退化) |

α=0.3~0.7 均稳定有效，α=0.9 过大导致性能退化（好前缀权重过高影响正确答案的优化）。

**标准差 & ReLU 消融** (Qwen3-4B-Base best: w.o. std / w.o. ReLU = 35.4; Qwen3-4B best: w.o. std / w. ReLU = 44.8):
- 去除标准差对所有模型都有益（因为 token-level advantage 非均匀，std 极小时归一化爆炸）
- ReLU：base 模型不用更好（允许负梯度修正低质量 prefix），instruct 模型用更好（保护高质量 prefix）

## 与 URLVR / SPC 研究的关系

### VPPO 是否需要 Ground Truth？

**是的，VPPO 需要两种外部监督**:
1. **Outcome-level GT**: 依赖 verifiable reward（最终答案对/错）来决定对正确响应给 reward=1
2. **PRM**: 依赖预训练的 Qwen2.5-Math-PRM-7B 来检测第一个错误步骤——PRM 本身是在含 GT 标注的 ProcessBench 类数据上训练的

因此 VPPO **不属于 URLVR 范畴**，它是有监督 RLVR 中的方法。

### "First Error Localization" 能否用于无监督 SPC？

**有若干值得借鉴的 idea**:

1. **Good prefix 保留思想**: SPC 可以借鉴 VPPO 的核心 insight——即使轨迹整体错误，前面的步骤可能有价值。在 TTRL pseudo-label 场景下，如果能识别"从哪一步开始偏离一致性"，就能构造类似的 prefix reward。

2. **First divergence 替代 first error**: 在 URLVR 中没有 PRM 或 GT，但 SPC 的 step-level consistency 信号可以起到类似作用——找到第一个"consistency 急剧下降"的步骤，将其之前视为"good prefix"。这是 SPC 版本的 first error localization。

3. **Reward 设计可直接复用**: VPPO 的 Eq. 6 reward scheme（prefix 给 α，suffix 给 0）非常简洁，如果 SPC 能可靠地检测 first divergence point，可以直接套用这个 reward 结构。

4. **Shorten prefix 策略**: 防止 step inflation 的 shorten prefix trick 在任何 step-level reward 方法中都适用，SPC 实现时应注意类似的 reward hacking。

5. **理论结果的迁移**: Theorem 1 的指数 vs 线性加速不依赖于 PRM 的具体形式，只要 good prefix 的检测大致正确即可。这为 SPC 的理论分析提供参考。

**核心区别**: VPPO 的 PRM first-error detection 在 ProcessBench 上有 92.7% 的 "Not More" 安全率。SPC 的 first divergence detection 需要证明类似的可靠性——如果误判导致 "More"（将错误步骤误判为正确 prefix），会引入 harmful reward。

6. **降低 SPC 计算开销**: 采用 VPPO 的 first-error 范式后，SPC 不需要对每步都做完整 probing（计算昂贵），只需从第一步开始扫描 consistency，扫到第一个 SPC 急剧下降的步骤即可停止——后续步骤全部归入 erroneous suffix。这可以显著降低 SPC 的推理开销。

7. **与 TTRL pseudo-label 双层兼容**: Outcome reward 用 TTRL majority voting pseudo-label，step-level 用 SPC first-divergence detection，形成 "outcome anchor + step signal" 两层信号。这与 SPC 方案的三层自举架构（TTRL → SPC → Co-Evolving Verifier）天然契合。

8. **SPC 可靠性的空间结构**: SPC 在推理链前几步的 consistency 信号通常比后几步更可靠（后几步的 continuation 空间更大、不确定性更高）。这与 VPPO 的 prefix-reward 思想天然匹配——SPC 最自信的区域恰好是前缀区域。

### 与已有 Wiki 论文的关联
- **[[wu-2026-spae|SPAE]]**: 同样做 step-level credit assignment，但 SPAE 是 training-free probing + GT correctness，VPPO 是 PRM-based first error detection + reward shaping
- **[[rahman-2025-spark|SPARK]]**: 同样用 PRM 做 step-level reward，但 SPARK 训练 PRM → 逐步分数，VPPO 只用 PRM 做 binary first-error detection
- **[[zuo-2025-ttrl|TTRL]]**: VPPO 的 prefix reward 思想可迁移到 TTRL pseudo-label 场景
- **[[zhang-2025-covo|CoVo]]**: CoVo 的 consistency/volatility 信号在概念上类似"检测推理轨迹何时偏离"，但在 token-level likelihood 层面操作，VPPO 在 step-level PRM 层面操作

## 局限性与开放问题

1. **依赖步骤格式**: 要求模型以 "Step 1... Step 2..." 格式输出，步骤定义依赖字面匹配 marker。Free-form 推理（如 DeepSeek-R1 的长 CoT）无法直接适用
2. **Step Inflation 仍是 fundamental risk**: Shorten prefix 是 empirical fix，缺乏对 step inflation 行为的 principled 理论理解
3. **Shorten prefix 截断长度 $c(q)$ 需要调参**: 最优选择（prompt 长度）有效但非普适最优
4. **PRM 质量依赖**: 虽然只用 PRM 做 first error detection（比 step-wise scoring 更鲁棒），但仍需一个高质量 PRM；PRM 训练本身需要标注数据
5. **仅验证数学推理**: 实验仅涵盖数学任务，未验证代码、逻辑推理等领域的泛化性
6. **模型多样性有限**: 仅在 Qwen3 系列上实验

## 面试关联

- 🔴 **Process reward vs Outcome reward**：PRM 在 RL 中的使用方式
- 🔴 **Credit assignment**：如何精确归因推理链中的错误
- 🟡 **Reward hacking**：为什么 dense PRM scores 不可靠
- 🟡 **First-error detection**：将 PRM 用途简化为二值定位

## 面试 Q&A

### Q1: VPPO 和直接用 PRM 分数做 reward 有什么区别？
**A**: 传统方法（如 Mixed）将 PRM 逐步分数作为 reward 直接优化，问题是 PRM 分数有系统性偏差、难以解释好坏界限。VPPO 只用 PRM 做一个 binary 判断——第一个错误步骤在哪——这个任务有明确 GT、有 ProcessBench 可评测，然后用 first error 位置做 trajectory partition，给 prefix α reward、suffix 0 reward。核心区别是从"优化 noisy continuous score"变为"利用 reliable binary detection"。

### Q2: 为什么需要 shorten prefix？简单给好前缀全部 reward 不行吗？
**A**: 因为 reward hacking。步骤是用 "Step k" marker 切分的，模型发现拆分一个大步骤成多个小步骤可以增加 good prefix token 数量（从而增加总 reward），导致 step inflation。Shorten prefix 截掉末尾一段 buffer，使得 boundary 附近新增的 micro-steps 落入无 reward 区域，打破 inflation 激励。

### Q3: α=0.5 是怎么选的？太大或太小会怎样？
**A**: 消融显示 α=0.3~0.7 性能稳定，α=0.9 退化。直觉：α 太大等于把错误响应的好前缀权重提到接近正确响应，这会削弱模型对"最终答案正确"的追求，导致 exploitation 不足。

### Q4: Theorem 1 的指数加速是什么意思？
**A**: 在 H 步推理树中，sparse reward 只有采到完整正确路径才能学习，概率 ~2^{-H}，需要 ~2^H 样本。VPPO 可以逐层学习：采到前 h 步正确就能更新前 h 步 policy，每层只需 O(1/α) 样本，共 H 层，总 O(H) 样本。从指数到线性。
