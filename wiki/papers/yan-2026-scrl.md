---
title: "SCRL: Selective-Complementary Reinforcement Learning at Test Time"
type: paper
tags: [URLVR, TTRL, negative-pseudo-labeling, entropy-gating, selective-labeling, robust-training, GRPO, weak-consensus]
created: 2026-04-10
updated: 2026-04-10
sources: [https://arxiv.org/abs/2603.19880]
status: active
---

# SCRL: 选择性正标签 + 熵门控负标签改进 TTRL

## 基本信息
- **作者**: Dong Yan, Jian Liang, Yanbo Wang, Shuo Lu, Ran He, Tieniu Tan
- **机构**: University of Chinese Academy of Sciences (中国科学院大学) & Institute of Automation, CAS (中科院自动化所) & Nanjing University
- **年份**: 2026
- **会议/期刊**: arXiv preprint (arXiv:2603.19880v1)
- **链接**: https://arxiv.org/abs/2603.19880
- **代码**: https://github.com/Jasper-Yan/SCRL
- **Base Model**: Qwen2.5-3B, Qwen2.5-Math-7B, Qwen3-4B, Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct

## 一句话总结
> 首次在 TTRL 中引入 **负 pseudo-labeling** 机制：当 majority voting 共识弱时，通过严格阈值筛选正标签（Selective Positive），同时用 entropy-gated 方式识别"低频+高不确定性"的负标签来修剪搜索空间，在受限 rollout 预算下显著提升鲁棒性，AIME25 从 2.6% 提升到 8.4%（Qwen2.5-3B）。

## 摘要
[[wiki/papers/zuo-2025-ttrl|TTRL]] 依赖 majority voting 共识来派生 pseudo-reward，但现有方法仅使用正 pseudo-labeling 策略。在答案分布高度分散的困难场景下，弱共识会将错误轨迹作为监督信号强化。SCRL 提出三重机制：(1) **Selective Positive Pseudo-Labeling** 强制严格共识标准过滤不可靠的多数投票；(2) **Entropy-Gated Negative Pseudo-Labeling** 首次在 TTRL 中引入负监督，基于生成不确定性可靠地修剪错误轨迹；(3) **Dynamic Reward Shaping** 根据共识强度校准强化幅度。在多个推理 benchmark 上实现显著提升，尤其在受限 rollout 预算下表现稳健。

## 核心贡献
1. **首次 TTRL 负监督**: 提出 Entropy-Gated Negative Pseudo-Labeling，是 test-time RL 中第一个负监督机制，利用"低频+高不确定性"可靠识别错误答案
2. **Selective Positive Labeling**: 严格的共识阈值 + margin 条件，在弱共识时主动放弃正监督（abstention），防止噪声放大
3. **Dynamic Reward Shaping**: 基于分布感知的动态 reward 设计，平滑正信号和负信号的强化幅度
4. **受限预算下的鲁棒性**: 在 rollout budget 减半（32 candidates）时仍保持显著优势，解决实际部署中计算资源受限的问题

## 方法

### 问题分析：弱共识下的噪声放大

GRPO 中 group normalization 的数学分析揭示：当正标签比例 $f$ 很小时（弱共识），正样本的 advantage 变为：

$$\hat{A}^+ = \sqrt{\frac{1-f}{f}}$$

$f$ 越小，$\hat{A}^+$ 越大，少量（可能错误的）正 pseudo-labeled 轨迹会**不成比例地影响策略更新**。如果投票答案错误，GRPO 会快速强化这个虚假信号。

### 技术方案

#### 1. Selective Positive Pseudo-Labeling

将 majority voting 转化为**带弃权**的选择性标注规则。给定 $N$ 条 response 的答案分布 $\mathcal{A} = \{a_j\}_{j=1}^K$，比例 $p_j = n_j / N$，仅当满足以下两个条件时才给出正 pseudo-label：

$$y^+ = a_{j^*} \quad \text{if} \quad p_{j^*} \geq \tau_{\text{pos}} \;\wedge\; (p_{j^*} - p_{(2)}) > \tau_{\text{marg}}$$

否则 $y^+ = \varnothing$（弃权）。

- $\tau_{\text{pos}} = 0.375$：最高频答案的最低支持比例
- $\tau_{\text{marg}} = 0.125$：与第二高频答案的最低间距
- 弃权时不进行正强化，转向负标签

#### 2. Entropy-Gated Negative Pseudo-Labeling

**核心洞察**: 在高不确定性下，识别错误答案比识别正确答案更可靠。

**Step 1 - Token-level 熵计算**: 对每条 response 的每个 token 计算 next-token 分布的 Shannon 熵：

$$h_{i,t} = -\sum_{v \in \mathcal{V}} \pi_{\text{old}}(v \mid o_{i<t}) \log \pi_{\text{old}}(v \mid o_{i<t})$$

**Step 2 - Trajectory-level 不确定性聚合**:

$$\bar{h}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} h_{i,t}$$

**Step 3 - Answer-level 不确定性**:

$$\bar{H}_j = \frac{1}{n_j} \sum_{i: a_i = a_j} \bar{h}_i, \quad \bar{H} = \frac{1}{N} \sum_{i=1}^{N} \bar{h}_i$$

**Step 4 - 负标签规则**: 同时满足低频 + 高不确定性：

$$\mathcal{N}^- = \{a_j \in \mathcal{A} \mid p_j < \tau_{\text{neg}} \;\wedge\; \bar{H}_j \geq \bar{H}\}$$

- $\tau_{\text{neg}} = 0.125$：低支持阈值
- $\bar{H}_j \geq \bar{H}$：不确定性高于平均，防止误杀低频但正确的答案

#### 3. Dynamic Reward Shaping

$$R_i = p(a_i) \cdot \mathbb{I}[a_i = y^+] + (p(a_i) - \tau_{\text{neg}}) \cdot \mathbb{I}[a_i \in \mathcal{N}^-] - \lambda_H (\bar{H}(a_i) - \bar{H})$$

- **正项**: 以答案比例 $p(a_i)$ 加权，共识越强 reward 越大
- **负项**: $(p(a_i) - \tau_{\text{neg}})$ 为负值（因为 $p(a_i) < \tau_{\text{neg}}$），且越低频惩罚越大
- **熵惩罚项**: $\lambda_H = 0.1$，轻微偏向低不确定性的 response

### 实现细节
| 参数 | 值 |
|------|-----|
| Rollout candidates | 64 或 32 |
| Training samples | 32 或 16 |
| LR | $5 \times 10^{-7}$ (cosine schedule) |
| $\tau_{\text{pos}}$ | 0.375 |
| $\tau_{\text{marg}}$ | 0.125 |
| $\tau_{\text{neg}}$ | 0.125 |
| $\lambda_H$ | 0.1 |
| Max gen length | 3072 |
| Training episodes | 10 (MATH/Minerva), 30 (AMC), 80 (AIME) |
| GPUs | 8×A100 80GB |

## 实验结果

### 主实验（pass@1, 32 candidates / 16 training）

| Method | AIME25 | AMC | MATH-500 | Minerva | GPQA | Avg |
|--------|--------|-----|----------|---------|------|-----|
| Qwen2.5-3B (base) | 1.9 | 23.2 | 50.0 | 21.4 | 21.6 | 23.6 |
| + TTRL | 2.6 | 39.4 | 66.9 | 31.6 | 25.0 | 33.1 |
| **+ SCRL** | **8.4** | **41.5** | **68.2** | 29.5 | **25.7** | **34.7** |
| $\Delta$ | **+5.8** | +2.1 | +1.3 | -2.1 | +0.7 | **+1.6** |

| Method | AIME25 | AMC | MATH-500 | Minerva | GPQA | Avg |
|--------|--------|-----|----------|---------|------|-----|
| Qwen2.5-Math-7B (base) | 4.6 | 34.0 | 46.5 | 10.1 | 23.0 | 23.6 |
| + TTRL | 16.8 | 65.7 | 85.7 | 14.5* | 25.5 | 41.6 |
| **+ SCRL** | **26.9** | **66.9** | 85.6 | **41.6** | 25.4 | **49.3** |
| $\Delta$ | **+10.1** | +1.2 | -0.1 | **+27.1** | -0.1 | **+7.7** |

*: TTRL 峰值性能，后续发生严重退化

### Instruct 模型结果

| Method | AIME24 | MATH/AMC | Average |
|--------|--------|----------|---------|
| Llama-3.2-1B + TTRL | 6.7 | 27.8 | 17.3 |
| Llama-3.2-1B + **SCRL** | **6.7** | **39.7** | **23.2** (+7.1) |
| Llama-3.1-8B + RESTRAIN | 16.7 | 40.0 | 28.4 |
| Llama-3.1-8B + **SCRL** | **21.9** | 36.1 | **29.0** (+0.6) |

### Long-CoT 模型（Qwen3-4B thinking mode）

| Max Gen Length | Qwen3-4B | + TTRL | + SCRL | $\Delta$ |
|----------------|----------|--------|--------|----------|
| 10,240 | 38.9 | 50.9 | **53.6** | +2.7 |
| 15,360 | 51.9 | 59.3 | **60.2** | +0.9 |

### 消融实验（Qwen2.5-3B, pass@1）

| Method | AIME25 | AMC |
|--------|--------|-----|
| SCRL (full) | **8.4** | **41.5** |
| w/o Selective Labeling | 3.9 (↓4.5) | 41.8 (↑0.3) |
| w/o Negative Labeling | 1.2 (↓7.2) | 37.7 (↓3.8) |
| w/o Entropy Gate | 3.6 (↓4.8) | 39.3 (↓2.2) |
| w/o Dynamic Reward | 5.1 (↓3.3) | 39.8 (↓1.7) |

**关键发现**:
- **Negative Labeling 是最关键组件**: 移除导致 AIME25 从 8.4 跌到 1.2（↓7.2），比移除任何其他组件影响都大
- **Entropy Gate 不可或缺**: 仅用频率判断负标签（无熵门控）导致 AIME25 从 8.4 跌到 3.6，说明会误杀低频正确答案
- **Selective Labeling 在困难 benchmark 关键**: AIME25 影响大（↓4.5），但在较简单的 AMC 上反而略微降低（说明过于保守）
- **Minerva 的稳定性**: TTRL 在 Minerva 上峰值 14.5% 后严重退化，SCRL 稳定达到 41.6%——差距达 27.1%

### 泛化性（在 AIME25 上训练 → 其他 benchmark 评估）

| Method | AMC pass@1 | AMC pass@16 | MATH pass@1 | MATH pass@16 |
|--------|-----------|-------------|-------------|--------------|
| Qwen2.5-3B (base) | 23.2 | 67.5 | 50.0 | 86.4 |
| + TTRL | 27.3 | 54.2 | 58.5 | 84.8 |
| **+ SCRL** | **27.5** | **60.2** | **65.1** | **87.0** |

TTRL 的 pass@16 大幅下降（67.5→54.2），说明 majority voting 缩窄了解空间；SCRL 的 conservative labeling 保持了探索能力。

## 与其他工作的关系

### 与 [[wiki/papers/zuo-2025-ttrl|TTRL]] 的直接改进关系
- SCRL 是 TTRL 的直接改进，保留 GRPO + majority voting 框架
- 核心创新在于**何时使用正标签**（selective）和**何时使用负标签**（entropy-gated），而非改变投票机制本身
- TTRL 的 "lucky hit" 在弱共识时失效——SCRL 通过 abstention 机制避免强化虚假共识

### 与 [[wiki/papers/pan-2026-coverrl|CoVerRL]] 的比较
- CoVerRL: 训练 verifier 过滤 self-consistent errors（重模型）
- SCRL: 用统计信号（频率+熵）过滤不可靠标签（轻量级，无需额外模型/角色）
- 两者解决同一问题（弱共识下的噪声放大），但 SCRL 计算开销更低
- SCRL 的负标签思路可能与 CoVerRL 的 verifier 互补

### 与 [[wiki/papers/du-2026-dual-consensus|DCRL]] 的比较
- DCRL 通过引入 explorer 的"第二视角"
- SCRL 不需要第二个模型，而是利用 entropy 信号从单一模型的输出中提取更多信息
- SCRL 的负标签机制在 DCRL 中无对应物——DCRL 仅改进正标签的选取

### 与 [[wiki/papers/liu-2025-ettrl|ETTRL]] 的比较
- ETTRL 也使用 entropy 信号，但用于控制 rollout 的探索-利用平衡
- SCRL 的 entropy 用于 **判断 negative label 可靠性**，目标不同
- SCRL 在主实验中超越 ETTRL（ETMR 变体）

## 局限性与开放问题
1. **阈值超参**: $\tau_{\text{pos}}, \tau_{\text{marg}}, \tau_{\text{neg}}$ 需要人工设定，虽然不同模型使用相同值效果尚可，但最优值可能因数据/模型而异
2. **仍然是 outcome-level**: 正/负标签都基于最终答案，无 step-level credit assignment
3. **简单 benchmark 上改进有限**: AMC/MATH-500 上 SCRL 改进较小（1-2%），说明弱共识问题主要出现在困难题目
4. **Negative labeling 的覆盖率**: 当所有答案都高频或都低频时，负标签可能失效

## 与 SPC/URLVR 研究的关系

### 对 SPC 方案的启发
1. **负监督的可行性验证**: SCRL 首次证明在 TTRL 框架中，负标签（"这肯定不对"）比正标签（"这可能对"）在困难问题上更可靠。SPC 的 step-level credit assignment 可以借鉴类似思路——对 step-level 也区分"高可信正步骤"和"高可信负步骤"。

2. **Entropy 作为 step-level 信号**: SCRL 用 token-level entropy 聚合到 trajectory-level 来判断不确定性。SPC 可以类似地利用 token-level entropy 作为 step-level 的辅助信号——高熵 token 可能对应推理中的关键决策点。

3. **SPC 可解决 SCRL 的局限**: SCRL 仍然是 outcome-level 的负标签（判断整条轨迹），而 SPC 的 probing + semantic consistency 可以定位到具体出错的步骤，提供更精细的负信号。

4. **Co-Evolving Verifier 可整合负监督**: 在 verifier 的训练中，除了正例/负例的对比学习，还可以引入 SCRL 的 entropy-gated negative 信号来校准 verifier 的不确定性感知。

## 面试相关

- **Q: SCRL 为什么在困难题目上改进巨大，但在简单题目上改进有限？** 🔴
- A: 因为 SCRL 主要解决"弱共识下的噪声放大"问题。在简单题目上，majority voting 的共识通常足够强（$f$ 大），正标签准确度高，GRPO 的 advantage 放大效应不严重。在困难题目上（如 AIME25），答案分布高度分散，$f$ 小导致 $\hat{A}^+ = \sqrt{(1-f)/f}$ 很大，少量错误正标签会被 GRPO 急剧放大。SCRL 的 selective labeling（弃权机制）和 negative labeling（修剪搜索空间）在此场景下至关重要。

- **Q: Entropy-Gated Negative Pseudo-Labeling 为什么需要同时满足低频和高不确定性两个条件？** 🔴
- A: 仅用低频会误杀"少数但正确"的答案——某些难题上正确答案可能恰好是少数。加上高不确定性条件（$\bar{H}_j \geq \bar{H}$）可以区分两种低频答案：(1) 模型高确定性地给出了一个少数答案→可能是正确的少数→不应惩罚；(2) 模型在高不确定性下偶然产生了少数答案→大概率是错误的→应惩罚。消融实验显示移除 entropy gate（仅用频率）导致 AIME25 从 8.4 跌到 3.6，验证了双条件的必要性。

- **Q: Dynamic Reward Shaping 相比 hard reward (+1/0/-1) 好在哪里？** 🟡
- A: Hard reward 对所有正/负标签等同对待，而 Dynamic Reward Shaping 以 $p(a_i)$ 加权——共识越强的正标签给更大 reward，频率越低的负标签给更大惩罚。这使得 GRPO 的策略更新与信号可靠性成比例，避免了在弱共识下过度强化和在强负信号下过度惩罚。消融实验显示替换为 hard reward 导致 AIME25 从 8.4 跌到 5.1（↓3.3）。
