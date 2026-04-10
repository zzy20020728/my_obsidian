---
title: "SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning"
type: paper
tags: [RLVR, process-supervision, credit-assignment, hierarchical-advantage, potential-estimation, token-efficiency]
created: 2026-04-10
updated: 2026-04-10
sources: ["https://arxiv.org/abs/2604.06636"]
status: active
---

# SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation

## 一句话总结

将推理过程建模为可解性状态空间中的轨迹，通过**段级阶段感知优势函数**和**token 级熵驱动重分配**实现分层信用分配，在提升 3% 准确率的同时减少 30% 的 token 消耗。

## 基本信息

- **作者**: Zhengyang Ai, Zikang Shan, Xiaodong Ai, Jingxian Tang, Hangkai Hu, Pinyan Lu†
- **机构**: 华为 Taylor Lab, 北京大学数据科学中心, 上海财经大学
- **发表日期**: 2026-04-08
- **arXiv**: 2604.06636

## 摘要

过程监督已成为增强 LLM 推理的有前途方法，但现有方法无法区分有意义的进展和冗余表述，导致推理能力有限且 token 效率低。SHAPE 将推理形式化为经验可解性状态空间中的轨迹传播，引入分层信用分配：在**段级**使用阶段感知优势函数优先奖励低潜力状态的突破；在**token 级**利用熵驱动重分配锐化执行信号。在三个基础模型和五个基准上，SHAPE 平均准确率提升 3%，同时减少 30% token 消耗。

## 核心贡献

1. **形式化最优推理路径的三原则**: Potential Gain（潜力增益）、Stage Awareness（阶段感知）、Token Efficiency（token 效率），并指出现有方法只能部分满足
2. **提出 SHAPE 框架**: 利用动态折扣机制同时满足三个准则，辅以分层 token 级精细化
3. **实证验证**: 在多个基准上实现准确率提升 + token 消耗降低的帕累托前沿

## 方法

### 关键概念: 推理潜力 (Reasoning Potential Φ)

推理潜力 $\Phi(s_k)$ 量化状态 $s_k$ 的即时可解性。通过在每个段边界执行 $m$ 次 rollout，计算平均正确率：

$$\Phi(s_k) = \frac{1}{m} \sum_{i=1}^{m} r_i$$

### 1. 基于熵的轨迹分割

使用 token 级熵识别推理边界（高熵点 = 关键逻辑转折点）：

$$H(x_t) = -\sum_{v \in \mathcal{V}} \pi_\theta(v | x_{<t}) \log \pi_\theta(v | x_{<t})$$

超过阈值 $\tau$ 的位置作为候选切分点，下采样为 $K$ 个段。

### 2. 段级阶段感知优势（核心公式）

基于 PBRS (Potential-Based Reward Shaping) 理论，引入长度依赖的动态折扣因子：

$$\gamma_k(L_k) = \max\left(\gamma_{\min},\ 1 - \frac{L_k}{L_{\text{ref}}}(1 - \gamma_{\min})\right)$$

段级优势定义为：

$$A_k = R_{\text{outcome}} + \alpha \cdot \underbrace{\left(\gamma_k(L_k) \cdot \Phi(s_{k+1}) - \Phi(s_k)\right)}_{\text{Potential-Based Shaping}}$$

**机制分析**: 将 shaping 项分解可得：

$$F_k \approx \Delta_k - \underbrace{(1 - \gamma_k(L_k)) \cdot \Phi(s_k)}_{\text{Reasoning Tax}}$$

- **阶段感知**: Tax 与 $\Phi(s_k)$ 成正比。低潜力阶段（模型困惑）Tax 小 → 鼓励突破；高潜力阶段 Tax 大 → 抑制潜力膨胀
- **Token 效率**: Tax 与 $(1-\gamma_k)$ 成正比，后者随段长 $L_k$ 线性增长 → 更长的段承受更重的税

### 3. Token 级信用重分配

使用 Z-score 标准化的熵作为 token 重要性权重：

$$\tilde{H}(x_t) = \frac{H(x_t) - \mu_H}{\sigma_H + \epsilon}, \quad w_t = \text{clip}(1 + \beta \cdot \tilde{H}(x_t), \delta_{\min}, \delta_{\max})$$

最终 token 优势: $A_t = A_k \cdot w_t$

### 与 MRT 的关键区别

| 维度 | MRT | SHAPE |
|------|-----|-------|
| 优势计算 | $R - \Phi(s_k)$（仅当前状态） | $\gamma_k \Phi(s_{k+1}) - \Phi(s_k)$（相邻差分） |
| 长度控制 | 无 | 动态折扣因子 $\gamma_k(L_k)$ |
| token 信用 | 段内均匀 | 基于熵的重分配 |
| Sandbagging | 存在风险 | PBRS 消除 |

## 实验结果

### 主实验（5 基准平均）

| 模型 | 方法 | 准确率 | Token 数 |
|------|------|--------|----------|
| DS-R1-Distill-Qwen-1.5B | GRPO | 52.1 | 6111 |
| | MRT | 51.9 | 4632 |
| | **SHAPE** | **54.7** (+2.6) | **4165** (-31.8%) |
| DeepScaleR-1.5B | GRPO | 55.6 | 5416 |
| | MRT | 57.1 | 4238 |
| | **SHAPE** | **59.4** (+3.8) | **3765** (-30.5%) |
| Qwen3-4B | GRPO | 74.4 | 9650 |
| | MRT | 74.2 | 8295 |
| | **SHAPE** | **77.5** (+3.1) | **7404** (-23.3%) |

### 消融实验（DS-R1-Distill-Qwen-1.5B, AIME 24）

| 变体 | AIME 24 Acc | Token 数 |
|------|-------------|----------|
| SHAPE | 37.1 | 6164 |
| w/o EBS（无熵分割） | 36.8 (-0.3) | 6380 (+3.5%) |
| w/o TCR（无 token 重分配） | 36.2 (-0.9) | 6080 (-1.4%) |
| Fixed $\gamma_k=0.9$ | 36.5 (-0.6) | 6955 (+12.8%) |
| $\gamma_{\min}=0.7$（过度折扣） | 30.8 (-6.3) | 4580 (-25.7%) |

### 关键发现

- **分割粒度**: $K=8$ 为最佳甜点，训练开销 1.17x 换取 ~30% 推理节省
- **$\gamma_{\min}$ 理论下界**: 0.875。$\gamma_{\min}=0.9$ 保持安全边际；$\gamma_{\min}=0.7$ 导致 Reward Sign Consistency 违反，训练崩溃
- **Sandbagging 验证**: MRT 的 potential drop rate 呈高波动 + 晚期飙升；SHAPE 稳定下降
- **自适应计算**: SHAPE 对难度的响应更陡峭且方差更低，精准匹配推理深度与问题难度

## 与 SPC/URLVR 研究的关系

### 直接关联

1. **信用分配粒度对比**: SHAPE 使用段级 + token 级分层分配，而 SPC 提出通过 probing + 语义一致性实现步级分配。SHAPE 的 "基于熵的分割" 可类比 SPC 中基于 probing 的语义边界检测
2. **Potential 估计 vs Probing**: SHAPE 通过 rollout 估计 $\Phi$ 需要额外采样开销，而 SPC 通过 hidden state probing 可以实现更轻量的潜力估计
3. **对 Co-Evolving Verifier 的启示**: SHAPE 的 "Task Consistency" 定理（$\alpha < 0.5$ 保证正确解奖励严格高于错误解）为轻量 verifier 的噪声容忍度提供理论参考

### 可借鉴的思路

- **阶段感知思想**: 低潜力状态的突破比高潜力状态的微调更有价值 → SPC 的信用分配也应考虑阶段感知加权
- **动态折扣机制**: 长度惩罚防止冗余推理 → 可整合到 SPC 的 step-level reward 设计中
- **Reasoning Tax 分解**: 优雅地统一了效率与质量优化

## 面试 Q&A

### Q1: SHAPE 如何解决 MRT 的 "sandbagging" 问题？

**A**: MRT 的优势函数 $A_k^{MRT} = R + \alpha(R - \Phi(s_k))$ 仅依赖当前状态潜力与终点的差距，不考虑相邻状态的潜力变化。模型可以故意走入低潜力状态再"恢复"，获得更大的 bonus。SHAPE 通过 PBRS 差分建模 $\gamma_k \Phi(s_{k+1}) - \Phi(s_k)$ 直接惩罚潜力下降的步骤，从结构上消除了这一漏洞。实验也验证了 MRT 的 potential drop rate 存在晚期飙升现象。

### Q2: 为什么 $\gamma_{\min}=0.7$ 会导致训练崩溃？

**A**: SHAPE 推导了 Reward Sign Consistency 条件：对于任何正向推理步（$\Phi(s_{k+1}) > \Phi(s_k)$），shaping reward 必须为正。最严格情况下需要 $\gamma_k > \Phi(s_k)/\Phi(s_{k+1})$，当从 $7/8$ 提升到 $1$ 时，临界值为 $0.875$。$\gamma_{\min}=0.7 \ll 0.875$ 会导致正确的推理步反而获得负奖励，使训练信号与优化目标矛盾。

### Q3: SHAPE 的 token 级重分配为什么选择在段内而非全局进行？

**A**: 全局 outcome reward 本身是稀疏且高方差的信号，在其上做 token 级调制会放大噪声。SHAPE 将重分配锚定在段级优势 $A_k$——一个基于潜力估计的稠密、低方差信号上。这种 "稳定锚点" 策略确保重分配是在精细化一个有效信号，而非放大混沌。
