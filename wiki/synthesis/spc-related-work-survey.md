---
title: SPC 相关工作文献调研报告
type: synthesis
tags: [SPC, step-level-credit-assignment, rollout-tree, prefix-sharing, process-reward, divergence-detection, RLVR, literature-survey]
created: 2026-04-15
updated: 2026-04-15
sources: [arXiv API search, 18+ rounds of keyword queries]
status: active
---

# SPC 相关工作文献调研报告

> **目标**：系统性搜索与 SPC（Semantic Process Consistency）研究方案相关的 arXiv 论文。  
> **方案核心**：在 RLVR 数学推理训练中，对同一问题采样 64 条完整推理轨迹，通过 majority voting 分正确/错误组，纯统计地分析正确组"锁定点"（收敛）和正确/错误组"分叉点"（决策点），用这些信号做 step-level credit assignment。  
> **关键概念**：trajectory divergence detection, token overlap/n-gram similarity between rollouts, step-level credit assignment from rollout statistics, process reward from rollout agreement/disagreement, prefix sharing analysis, voting consistency at intermediate steps.

---

## 搜索方法

共执行 **20+ 轮 arXiv 搜索**，覆盖以下关键词组合：

| 轮次 | 查询关键词 |
|------|-----------|
| 1-10 | 原始 10 组预定义查询（step-level credit assignment, trajectory divergence, token overlap, prefix sharing, rollout agreement, voting consistency, n-gram similarity, process reward from rollout statistics, etc.） |
| 11 | rollout consistency / trajectory clustering |
| 12 | MC process reward / shared prefix + branching |
| 13 | fork point + rollout / unsupervised process reward |
| 14 | completion agreement / OmegaPRM / Math-Shepherd / MCTS |
| 15 | weak-strong consistency / token overlap + step |
| 16 | SPAE / CoVo / SLATE（已知论文名称搜索） |
| 17 | gradient entanglement + token overlap + shared prefix |
| 18 | step-level majority voting / intermediate step voting |
| 19 | convergence point / lock-in + reasoning + rollout |
| 20 | rollout tree / branching rollout + GRPO |
| 21 | n-gram / token overlap + rollout + reward + reasoning |

---

## 高度相关（11 篇）

> 方法论/思路与 SPC 方案**直接类似或高度重叠**，构成最核心的竞品或基础工作。

### 1. RTMC — Rollout-Tree Monte Carlo (2604.11037) ⭐⭐⭐ 最相关

| 字段 | 内容 |
|------|------|
| **标题** | RTMC: Step-Level Credit Assignment via Rollout Trees |
| **作者** | Tao Wang, Suhang Zheng, Xiaoxiao Xu |
| **日期** | 2026-04-13 |
| **链接** | https://arxiv.org/abs/2604.11037 |

**核心思想**：观察到 group rollouts 对同一问题经常遍历**重叠的中间状态**，隐式形成一棵树，树的分支在连续决策点发散。利用跨 rollout 的**状态匹配**（state-action signature system）产生 per-step Q-values 和 advantages，无需学习 critic。

**与 SPC 的关系**：
- **几乎完全一致的核心观察**：rollouts 共享前缀、在关键步骤分叉
- **关键区别**：RTMC 用于 agentic RL (SWE-bench)，状态匹配基于交互历史的 hash 签名；SPC 面向数学推理，用语义等价判断
- **威胁等级**：极高。RTMC 是 SPC 在 agentic 领域的近似实现

---

### 2. GRSD / UI-Voyager (2603.24533) ⭐⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | UI-Voyager: Group Relative Self-Distillation |
| **作者** | (UI-Voyager team) |
| **日期** | 2026-03 |
| **链接** | https://arxiv.org/abs/2603.24533 |

**核心思想**：**识别 group rollouts 中的 critical fork points**，从成功轨迹构建 dense step-level supervision 来纠正失败轨迹。

**与 SPC 的关系**：
- 直接使用"分叉检测"概念
- 关键区别：面向 UI agent，fork point 检测方法可能不同

---

### 3. GiGPO — Group-in-Group Policy Optimization (2505.10978) ⭐⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | GiGPO: Group-in-Group Policy Optimization |
| **作者** | (GiGPO team) |
| **日期** | 2025-05 |
| **链接** | https://arxiv.org/abs/2505.10978 |

**核心思想**：引入 **anchor state grouping mechanism**，通过识别跨轨迹的重复环境状态来回溯构建 step-level groups。从**同一状态**出发的 actions 分组在一起进行 micro relative advantage estimation。

**与 SPC 的关系**：
- 直接实现了"跨轨迹状态匹配→step-level advantage"这一核心路径
- 关键区别：GiGPO 用状态精确匹配（适合 code/tool），SPC 用语义等价判断（适合数学推理的自然语言步骤）

---

### 4. PRAISE — Prefix-Based Rollout Reuse (2604.03675) ⭐⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | PRAISE: Prefix-Based Rollout Reuse for Step-Level Reward |
| **作者** | (PRAISE team) |
| **日期** | 2026-04 |
| **链接** | https://arxiv.org/abs/2604.03675 |

**核心思想**：从完整轨迹中提取不同 **prefix states**，利用 prefix 之间的性能差异推导 step-level rewards（adjacent prefix gains）。

**与 SPC 的关系**：
- 直接使用 prefix sharing 概念做 step reward
- 关键区别：PRAISE 需要 GT scoring，面向 agentic search，用 PPO；SPC 面向 URLVR，无需 GT
- 已创建 wiki 页面：[[wiki/papers/zhang-2026-praise|PRAISE]]

---

### 5. ELPO — Error-Localized Policy Optimization (2602.09598) ⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | Learning from the Irrecoverable: Error-Localized Policy Optimization |
| **作者** | Qiao Liang, Yuke Zhu, Chao Ge et al. |
| **日期** | 2026-02-10 |
| **链接** | https://arxiv.org/abs/2602.09598 |

**核心思想**：通过 **binary-search rollout trees** 定位第一个不可恢复的错误步骤（irrecoverable step），转化为 hierarchical advantage attribution + error-localized adaptive clipping。

**与 SPC 的关系**：
- 直接实现了 SPC 中"分叉点 = 第一个关键错误步骤"的想法
- 关键区别：ELPO 用 binary search rollout（计算密集），面向 tool-integrated reasoning

---

### 6. VPPO — Save the Good Prefix (2601.18984) ⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | Save the Good Prefix: Precise Error Penalization via Process-Supervised RL |
| **作者** | Haolin Liu et al. (Tencent AI Lab) |
| **日期** | 2026-01-26 |
| **链接** | https://arxiv.org/abs/2601.18984 |

**核心思想**：利用 PRM 定位第一个错误，将轨迹分为 **verified correct prefix** 和 **erroneous suffix**，前者正奖励、后者惩罚。

**与 SPC 的关系**：
- "first divergence localization" 思想可迁移到 SPC
- 关键区别：需要 GT + PRM（非 URLVR）
- 已创建 wiki 页面：[[wiki/papers/liu-2025-vppo|VPPO]]

---

### 7. BranPO — Branching Relative Policy Optimization (2602.03719) ⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | BranPO: Branching Relative Policy Optimization |
| **作者** | (BranPO team) |
| **日期** | 2026-02 |
| **链接** | https://arxiv.org/abs/2602.03719 |

**核心思想**：发现性能分叉主要发生在**尾部附近**，在 shared prefixes 上构建 contrastive suffixes 进行 step-level 对比监督。

**与 SPC 的关系**：
- 直接验证了"shared prefix + divergent suffix"的结构假设
- 重要发现：分叉主要在尾部（与 SPC 假设的"中间关键步骤分叉"不同，需实验验证）

---

### 8. BranchGRPO (2509.06040) ⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | BranchGRPO: Branching Group Relative Policy Optimization |
| **作者** | (BranchGRPO team) |
| **日期** | 2025-09 |
| **链接** | https://arxiv.org/abs/2509.06040 |

**核心思想**：将 rollout 过程重组为 **branching tree**，shared prefixes 分摊计算，引入 **depth-wise advantage estimator**。

**与 SPC 的关系**：
- 直接使用 prefix sharing 减少计算 + depth-wise advantage
- 与 Tree-GRPO、RTMC 形成 rollout-tree 方法族

---

### 9. GPO — Guided Pivotal Optimization (2509.16456) ⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | GPO: Guided Pivotal Optimization |
| **作者** | (GPO team) |
| **日期** | 2025-09 |
| **链接** | https://arxiv.org/abs/2509.16456 |

**核心思想**：通过 advantage function 估计识别推理轨迹中的 **"critical step"**，然后在 critical step 重置策略并重新采样。

**与 SPC 的关系**：
- "关键步骤定位 + 重采样"与 SPC 的分叉点概念高度一致

---

### 10. Tree-GRPO (2509.21240) ⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | Tree-GRPO: Tree-based Group Relative Policy Optimization |
| **作者** | (Tree-GRPO team) |
| **日期** | 2025-09 |
| **链接** | https://arxiv.org/abs/2509.21240 |

**核心思想**：将推理建模为 **rollout tree**，通过 shared common prefixes 和 MC estimation 构建 step-wise process supervision。

**与 SPC 的关系**：
- 直接实现了 rollout tree → step reward 的完整管线
- 已创建 wiki 页面参考

---

### 11. CSO — Verified Critical Step Optimization (2602.03412) ⭐⭐

| 字段 | 内容 |
|------|------|
| **标题** | CSO: Verified Critical Step Optimization |
| **作者** | (CSO team) |
| **日期** | 2026-02 |
| **链接** | https://arxiv.org/abs/2602.03412 |

**核心思想**：聚焦于 **verified critical steps**（alternate actions 可以翻转任务结果的决策点），使用 PRM 识别 candidate critical steps。

**与 SPC 的关系**：
- "critical step = 翻转结果的决策点"与 SPC 分叉点概念一致
- 关键区别：依赖 PRM

---

## 中度相关（15 篇）

> 部分思路与 SPC 重合，但核心方法或目标不同。可作为 SPC 的 baseline、灵感来源或互补方法。

### Process Reward 自动构建

| # | ID | 标题 | 日期 | 与 SPC 的关系 |
|---|-----|------|------|-------------|
| 12 | 2312.08935 | **Math-Shepherd** | 2023-12 | 开创性工作：用 rollout outcomes 自动构建 process-wise supervision data。SPC 的 MC rollout 概念可追溯至此 |
| 13 | 2406.06592 | **OmegaPRM** | 2024-06 | Divide-and-conquer MCTS + binary search 识别 CoT 第一个错误。ELPO/VPPO 的方法论源头 |
| 14 | 2412.01981 | **Free Process Rewards** | 2024-12 | 证明 implicit PRM 可通过训练 ORM 获得，无需步骤标注。理论上 SPC 可用类似方法降低 PRM 依赖 |
| 15 | 2506.03570 | **FreePRM** | 2025-06 | Pseudo step-level labels + Buffer Probability 去噪的弱监督 PRM。与 SPC 的伪标签方法有思路交集 |
| 16 | 2503.02382 | **EpicPRM** | 2025-03 | 基于每个中间步骤的量化贡献 + adaptive binary search 增强标注精度。与 SPC 的"步骤贡献量化"目标一致 |

### Process Reward 建模与融合

| # | ID | 标题 | 日期 | 与 SPC 的关系 |
|---|-----|------|------|-------------|
| 17 | 2509.26578 | **CRM (Conditional Reward Modeling)** | 2025-09 | 将推理建模为时间过程，每步 reward 条件化于前序步骤并链接到最终结果。SPC 也强调步骤间的因果关系 |
| 18 | 2509.03403 | **PROF (Process Consistency Filter)** | 2025-09 | PRM + ORM 互补，consistency-driven sample selection。SPC 的 consistency 概念与之呼应 |
| 19 | 2604.02341 | **PROGRS** | 2026-02 | Outcome-conditioned centering：在 outcome groups 内将 PRM scores 作为相对偏好。与 SPC 的"组内对比"思想一致 |
| 20 | 2509.16548 | **SCAN (Self-Denoising MC)** | 2025-09 | 分析 MC estimation 中的噪声分布并自去噪。SPC 的 probing 噪声问题可参考 |

### 特定技术与 SPC 部分重合

| # | ID | 标题 | 日期 | 与 SPC 的关系 |
|---|-----|------|------|-------------|
| 21 | 2604.10660 | **CPMI (Contrastive PMI)** | 2026-04 | 利用模型内部概率量化推理步骤对正确答案的贡献。与 SPC 用统计信号量化步骤贡献的目标一致 |
| 22 | 2509.19893 | **FPA (Future Policy Approximation)** | 2025-09 | 揭示 **gradient entanglement** 问题：正确和错误解共享大量 token overlap，导致梯度冲突。这是 SPC 方案需要解决的核心问题的理论阐释 |
| 23 | 2506.09532 | **Athena-PRM** | 2025-06 | 利用 **weak/strong completers 之间的 prediction consistency** 识别 reliable process labels。与 SPC "用 rollout 一致性/不一致性作为信号"的思路重合 |
| 24 | 2510.14942 | **GroundedPRM** | 2025-10 | 通过 MCTS 构建结构化推理路径进行 fine-grained credit assignment |
| 25 | 2509.19199 | **iStar** | 2025-09 | Implicit Step Rewards for Agentic RL：交替优化 implicit PRM 和 policy model |
| 26 | 2508.01773 | **UnPRM** | 2025-08 | Uncertainty-driven PRM data construction + Hybrid Majority Reward Vote 聚合。结合 majority voting 与 PRM 的思路与 SPC 有交集 |

---

## 低度相关（11 篇）

> 可作为背景知识、baseline 参考或边缘相关。

| # | ID | 标题 | 日期 | 简要说明 |
|---|-----|------|------|---------|
| 27 | 2601.06922 | **TreePS-RAG** | 2026-01 | Tree-based process supervision for agentic RAG，MC estimation over descendant outcomes |
| 28 | 2502.10581 | **Jia et al.** | 2025-02 | 理论证明 outcome supervision 在统计上不比 process supervision 更难 |
| 29 | 2503.22233 | **EDU-PRM** | 2025-03 | 熵驱动的不确定性 PRM，token entropy 自动锚定步骤边界 |
| 30 | 2501.04519 | **rStar-Math** | 2025-01 | MCTS + process preference model 的自我进化方法 |
| 31 | 2507.08297 | **KAT/Step-SRPO** | 2025-07 | 将 intermediate supervision 引入 GRPO 框架 |
| 32 | 2603.17815 | **MC Net Info Gain** | 2026-03 | 信息论方法自动生成 step-level labels |
| 33 | 2501.07301 | **Lessons of PRMs** | 2025-01 | MC estimation vs LLM-as-judge vs human annotation 全面比较 |
| 34 | 2506.15498 | **SPARE** | 2025-06 | 对齐 solution steps 到 reference solutions 进行 per-step annotation |
| 35 | 2410.13828 | **Gradient Entanglement** | 2024-10 | Margin-based alignment 的 gradient entanglement 理论分析（FPA 的理论基础） |
| 36 | 2602.02150 | **ECHO** | 2026-02 | Entropy-Confidence Hybrid 的 tree rollout + adaptive branching |
| 37 | 2601.08274 | **DART** | 2026-01 | Dynamic rollout trees + tree-based process advantage for tool-use |

---

## 方法族谱系图

```
┌─────────────────────────────────────────────────────────────┐
│                   Rollout-Tree 方法族                        │
│                                                              │
│   Math-Shepherd (2023) ─→ OmegaPRM (2024)                   │
│        │                       │                              │
│        ▼                       ▼                              │
│   Tree-GRPO (2025/09)    BranchGRPO (2025/09)               │
│        │                       │                              │
│        ▼                       ▼                              │
│   ELPO (2026/02)         BranPO (2026/02)                    │
│        │                       │                              │
│        ▼                       ▼                              │
│   RTMC (2026/04)  ←── GiGPO (2025/05)                       │
│        ↑                                                      │
│        │                                                      │
│    PRAISE (2026/04) ── SLATE (2025) ── CSO (2026/02)        │
│                                                              │
│                    ▲ SPC 方案的位置 ▲                        │
│   独特性：无监督 + 语义等价 + 短续写 probing + URLVR        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               Step-Level 信号方法对比                        │
│                                                              │
│   方法        信号来源       需要GT?  计算开销   粒度        │
│   ──────────  ─────────────  ───────  ────────   ────        │
│   RTMC        状态 hash      否       低         step        │
│   GiGPO       状态匹配       否       低         step        │
│   PRAISE      prefix rollout 是(GT)   高         step        │
│   SLATE       shared prefix  否       中         step        │
│   ELPO        binary search  否       高         step        │
│   VPPO        PRM first err  是(PRM)  低         binary      │
│   GPO         advantage      否       中         step        │
│   CSO         PRM + alt act  是(PRM)  高         step        │
│   BranPO      suffix contrast 否      低         suffix      │
│   SPC (ours)  semantic cons  否       中         step        │
└─────────────────────────────────────────────────────────────┘
```

---

## 对 SPC 方案的关键启示

### 1. SPC 的新颖性确认

尽管 rollout-tree / prefix-sharing / divergence-detection 的大方向已有大量工作，SPC 仍保持以下独特组合：
1. **完全无监督**：RTMC/GiGPO/BranchGRPO 不需要 GT 但面向 agentic/code；PRAISE/VPPO/CSO 需要 GT/PRM
2. **语义等价判断**：所有竞品用 exact match / hash / likelihood，无人用 semantic equivalence
3. **短续写 probing**：比完整 MC rollout 低开销，比 hidden state probing 更可解释
4. **面向 URLVR 数学推理**：唯一专注无标签数学推理场景的 step-level credit assignment

### 2. 最大威胁

- **RTMC (2604.11037)**：核心观察几乎完全一致，但面向不同领域。SPC 必须强调在**自然语言数学推理**中状态匹配需要语义理解（非 hash 可解决），这是差异化的关键
- **GiGPO (2505.10978)**：anchor state grouping 与 SPC 的跨轨迹步骤分组高度类似

### 3. BranPO 的重要发现

BranPO 发现分叉主要在**尾部**而非中间步骤。这对 SPC 假设（中间关键步骤的分叉很重要）构成挑战，需在实验中验证数学推理任务中分叉分布的实际模式。

### 4. Gradient Entanglement (FPA/Yuan et al.) 的理论支撑

FPA 和 Yuan et al. 从理论上证明了"正确和错误解共享大量 token overlap 导致梯度冲突"这一问题确实存在且严重。这为 SPC 方案提供了重要的 motivation：精确定位分叉点、对分叉前后的步骤赋予不同 credit，正是解决 gradient entanglement 的直接手段。

### 5. 建议 Related Work 写作框架

1. **MC-based PRM construction**: Math-Shepherd → OmegaPRM → EpicPRM → SCAN
2. **Rollout tree methods**: Tree-GRPO → BranchGRPO → BranPO → RTMC → GiGPO
3. **First error localization**: VPPO → ELPO → CSO
4. **Prefix-based step reward**: PRAISE → SLATE
5. **Gradient entanglement motivation**: Yuan et al. → FPA
6. **SPC positioning**: 无监督 + 语义等价 + 短续写，填补 URLVR 场景空白

---

## 搜索完整性说明

### 已搜索并确认无相关结果的方向
- **SPAE / CoVo / SLATE 名称搜索**：SPAE/SLATE 已在 wiki 中，CoVo 已在 wiki 中。arXiv 搜索这些缩写未返回额外相关论文
- **"step-level majority voting"**：无专门工作（最接近的是 UnPRM 的 Hybrid Majority Reward Vote，但这是 output aggregation 而非 step-level）
- **"intermediate step voting"**：无直接匹配结果
- **"convergence point" / "lock-in" + reasoning**：无相关结果（lock-in 在 arXiv 中主要指其他含义）
- **"n-gram" / "token overlap" + rollout + reward**：无直接结果

### 搜索局限性
- arXiv API 搜索对自然语言查询的匹配质量有限，部分论文可能因关键词差异被遗漏
- 2026 年 4 月之后的新论文未覆盖
- 非 arXiv 平台（如 OpenReview、会议论文集）未搜索

---

## 参见

- [[wiki/synthesis/step-level-se-proposal|SPC 研究方案]]
- [[wiki/synthesis/co-evolving-verifier-proposal|Co-Evolving Verifier 方案]]
- [[wiki/synthesis/urlvr-landscape|URLVR 领域综述]]
- [[wiki/synthesis/spc-experiment-plan|SPC 实验设计方案]]
