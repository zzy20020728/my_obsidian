---
title: 操作日志
type: log
---

# 操作日志

## [2026-04-07] init | Wiki 知识库初始化
- 创建目录结构：raw/, wiki/, plans/, templates/
- 创建 Schema (AGENTS.md)、索引 (index.md)、模板
- 创建秋招备战总计划
- 初始化 Git 仓库并推送到 GitHub

## [2026-04-07] update | 学习计划重构
- 修正个人背景：实习内容就是 URLVR 研究（10:00-19:00），算法岗积累在白天自然完成
- 重构时间安排：晚上 3h 全部给 Java，周末做论文 Wiki 整理 + 八股
- 移除 LeetCode 安排（暂不需要）
- 更新 master-plan.md 和 weekly-tracker.md

## [2026-04-07] content | 教程与示例页面
- 创建 Obsidian & Wiki 使用教程 (plans/obsidian-tutorial.md)
- 创建示例概念页：PPO (wiki/concepts/ppo.md)
- 创建示例概念页：RLHF (wiki/concepts/rlhf.md)
- 创建示例论文页：Schulman 2017 PPO (wiki/papers/schulman-2017-ppo.md)
- 更新 index.md 索引
- 优化 Obsidian 配置（shortest link format）

## [2026-04-07] ingest | URLVR 四篇论文批量摄入
- 摄入论文：EMPO (Zhang et al., 2025) — 语义熵最小化，完全无监督 RLVR
- 摄入论文：SPARK (Rahman et al., 2025) — 三阶段 PRM 训练，step-level reward
- 摄入论文：PRISM (Ghimire et al., 2026) — PRM + self-certainty 混合，纯内部信号失败分析
- 摄入论文：Self-Judge (Wu et al., 2026) — Actor-Judge 多模态无监督自进化
- 创建论文页：wiki/papers/zhang-2025-empo.md
- 创建论文页：wiki/papers/rahman-2025-spark.md
- 创建论文页：wiki/papers/ghimire-2026-prism.md
- 创建论文页：wiki/papers/wu-2026-self-judge.md
- 创建概念页：wiki/concepts/semantic-entropy.md（语义熵）
- 创建概念页：wiki/concepts/self-consistency.md（自一致性）
- 创建概念页：wiki/concepts/process-reward-model.md（过程奖励模型）
- 创建概念页：wiki/concepts/grpo.md（组相对策略优化）
- 创建概念页：wiki/concepts/reward-hacking.md（奖励攻击）
- 创建综合分析：wiki/synthesis/urlvr-landscape.md（URLVR 领域综述，含多维分类对比表格）
- 更新 index.md 索引（按类别组织所有新条目）

## [2026-04-08] ingest | Grad2Reward 论文摄入
- 摄入论文：Grad2Reward (Zhang et al., 2026, ICML) — 梯度 attribution 提取 dense token-level reward for open-ended tasks
- 来源：arXiv HTML 全文 (https://arxiv.org/html/2602.01791v1)
- 创建论文页：wiki/papers/zhang-2026-grad2reward.md（完整版：方法/公式/实验/消融/扩展分析/计算成本/面试要点）
- 更新 index.md 索引（添加 "Open-ended RL / Dense Reward" 分类）

## [2026-04-08] ingest | URLVR 第二批四篇论文摄入
- 摄入论文：MCNIG (Royer et al., 2026) — 信息论自动生成 PRM 训练数据，O(N) 复杂度
- 摄入论文：ProRAG (Wang et al., 2026) — 四阶段 RAG RL，MCTS-based PRM，dual-granularity advantage
- 摄入论文：CTRL-RAG (Tan et al., 2026) — Contrastive Likelihood Reward，轻量 RAG faithfulness 信号
- 摄入论文：How Far Can URLVR Scale? (He et al., 2026, ICLR) — 统一理论框架 [DRAFT: 仅 abstract]
- 创建论文页：wiki/papers/royer-2026-mcnig.md
- 创建论文页：wiki/papers/wang-2026-prorag.md
- 创建论文页：wiki/papers/tan-2026-ctrl-rag.md
- 创建论文页：wiki/papers/he-2026-urlvr-scale.md [draft]
- 创建概念页：wiki/concepts/information-gain.md（信息增益）
- 创建概念页：wiki/concepts/mcts.md（蒙特卡洛树搜索）
- 创建概念页：wiki/concepts/contrastive-likelihood.md（对比似然）
- 更新概念页：wiki/concepts/process-reward-model.md（添加 MCNIG/ProRAG 的 PRM 训练方法 + 对比表格）
- 大幅更新综合分析：wiki/synthesis/urlvr-landscape.md（从4篇扩展到8篇，添加 RAG 分类维度、Sharpening 理论、PRM 训练数据生成方法对比）
- 更新 index.md 索引（添加新论文、新概念页、RAG+RL 分类）

## [2026-04-08] ingest | SPAE 论文摄入
- 摄入论文：SPAE (Wu et al., 2026, USTC & iFLYTEK) — Training-free probing 提取 Step Potential，step-level credit assignment
- 来源：arXiv HTML 全文 (https://arxiv.org/html/2601.03823v1) + GitHub (https://github.com/cii030/SPAE-RL)
- 创建论文页：wiki/papers/wu-2026-spae.md（完整版：方法/公式/实验/消融/Behavior分析/可靠性验证/面试要点）
- 更新 index.md 索引（添加 "Step-Level Credit Assignment / Efficient Reasoning" 分类）

## [2026-04-08] research | Step-Level Semantic Entropy 可行性调研
- 搜索目标：是否已有工作将 semantic entropy 应用于 step-level 做 process reward
- **结论：无人做过「SE 作为 step-level process reward」这一具体组合，属于开放空白**
- 最相关工作：
  - Wang et al. 2025 (2511.06168) "Chain-of-Thought as a Lens"：step-level SE matrix 但仅用于评估，非 reward
  - ROSE (Zhao et al. 2026, 2601.05053)：SE 用于 MCTS branching 点选择，非 reward
  - CoFiCot (Zhang et al. 2026, 2603.08251)：SE 做 query-level 分流 + PRM 做 step-level，两者分离
  - EDU-PRM (Cao et al. 2025, 2503.22233)：token-level predictive entropy 做步骤分割，非语义熵
  - SEED-GRPO (Chen et al. 2025, 2505.12346)：SE 在 question-level 调节 GRPO 更新幅度
- 计算成本分析：
  - 标准 SE 需 K=5-10 次采样；step-level 需 T×K 次前向传播（T=步骤数），RL 训练中成本爆炸
  - 关键加速方案：SEPs (Semantic Entropy Probes, Kossen et al. 2024, 2406.15927) — 线性探针近似 SE，开销趋近零
  - 其他加速：自适应贝叶斯采样（Sun et al. 2026, ~50% 样本减少）、嵌入聚类替代 NLI

## [2026-04-08] synthesis | Step-Level SE 研究方案撰写
- 创建综合分析页：wiki/synthesis/step-level-se-proposal.md
- 核心方案：SPAE-SE（复用 SPAE probe 续写做语义聚类 → SC 替代 Correctness，实现无监督 step-level credit assignment）
- 三级递进技术路线：A. SPAE-SE（零额外成本）→ B. SPAE-SE-Long（更长续写）→ C. SPAE-SEP（线性探针零采样）
- 实验设计：Phase 1 相关性验证 → Phase 2 RL 训练对比 → Phase 3 SEP 优化
- 更新 index.md 索引

## [2026-04-08] tooling | 安装 arxiv-mcp-server
- 安装：`uv tool install arxiv-mcp-server` (v0.4.11)
- 可执行文件：`C:\Users\lenovo\.local\bin\arxiv-mcp-server.exe`
- 配置：创建 `opencode.json`（项目级 MCP 配置）
- 论文存储路径：`C:\Users\lenovo\Desktop\my_obsidian\raw\papers`
- **重启 OpenCode 后生效**，届时可用 search_papers / download_paper / read_paper 等工具

## [2026-04-08] git | 推送上一 session 所有更改
- commit `2e5598f`: SPAE + Grad2Reward 论文页、step-level-se-proposal、opencode.json、Obsidian 插件配置
- 更新 .gitignore 排除 PDF 文件（raw/papers/*.pdf）
- push 到 origin/main

## [2026-04-08] ingest | TTRL 论文深度摄入
- 论文：TTRL: Test-Time Reinforcement Learning (Zuo & Zhang et al., 2025, arXiv:2504.16084)
- 创建 wiki/papers/zuo-2025-ttrl.md（完整方法/Lucky Hit 理论/实验表格/失败分析/面试 Q&A）
- TTRL 是双层无监督架构（TTRL + SPAE-SE）的 Layer 1 (Outcome) 核心组件
- 更新 index.md 索引

## [2026-04-08] update | He et al. 2026 论文从 DRAFT 升级为完整版
- 论文：How Far Can Unsupervised RLVR Scale LLM Training? (He et al., 2026, ICLR)
- wiki/papers/he-2026-urlvr-scale.md 从 ~30 行 draft 升级为 ~275 行完整版
- 新增内容：
  - 完整统一分类体系（Certainty-based / Ensemble-based / External rewards）
  - **Sharpening Theorem (Theorem 1)**: 数学证明 majority voting reward 下策略几何收敛到确定性策略
  - **Unified Reward Framework**: 推广到所有 intrinsic rewards
  - **三种 Failure Modes**: gradual degradation / length collapse / repetition collapse
  - **Per-Problem 分析**: 22/25 问题只是 amplification 非 correction
  - **Small Dataset Safety**: 32-128 samples 不 collapse，TTT 安全
  - **MCS 定义与实验**: Reward Accuracy < 1% 步数，5.6x faster，无需 GT
  - **External Reward Self-Verification 实验**: Countdown 任务，远优于 intrinsic
  - 与所有已整理论文（TTRL/EMPO/SPARK/PRISM/Self-Judge/MCNIG/SPAE/Grad2Reward）的关系分析
  - 对 Step-Level SE 研究方案的理论影响分析
  - 面试 Q&A（6 题）

## [2026-04-08] update | URLVR 综述页更新 He et al. 2026 新内容
- 更新 wiki/synthesis/urlvr-landscape.md
- 新增 Theorem 1 数学细节、三种 Failure Modes、MCS 实用指标、External Reward 路径
- 更新核心发现章节：纳入 He et al. 的定量证据
