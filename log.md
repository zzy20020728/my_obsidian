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

## [2026-04-08] ingest | CoVo 论文深度摄入
- 论文：CoVo (Zhang et al., 2025, arXiv:2506.08745)
- 核心内容：Consistency + Volatility，用中间状态是否持续支持最终答案来构造无监督 reward
- 创建论文页：wiki/papers/zhang-2025-covo.md
- 更新 URLVR 综述页：纳入 CoVo，URLVR 核心论文由 9 篇扩展到 10 篇
- 更新 index.md 索引

## [2026-04-08] synthesis | Step-Level SE 方案重构为 SPC
- 重写 wiki/synthesis/step-level-se-proposal.md
- 放弃旧的 SPAE-SE / Semantic Certainty 主叙事，改为 **Semantic Process Consistency (SPC)**
- 新方案核心：在每个 step boundary 复用 SPAE probing，检查短续写导向的答案是否与轨迹最终答案语义一致
- 吸收 CoVo 的 consistency / volatility insight，但从 token likelihood judgment 升级为 semantic rollout behavior
- 明确回答旧方案的核心缺口：SPC 的目标是缓解 TTRL 的"一致地错"问题，而不只是替换一个 certainty 指标

## [2026-04-08] analysis | SPC 方案系统性评估
- 对 SPC 方案做全面优劣分析，识别六大问题（按严重程度排序）：
  1. Probing 计算开销（中）
  2. 短续写可能不收敛到明确答案（高）
  3. 语义等价判断可靠性（中）
  4. **仍属 intrinsic signal → sharpening（高，fundamental）**
  5. 没有跨轨迹信息（中）
  6. Φ_SPC 公式可能 over-parameterized（低）
- 结论：问题 #4 是唯一 fundamental 限制，其他都是工程问题
- Story 定位确认："CoVo 看模型怎么想，SPC 看模型怎么做"
- 该分析直接催生了 Co-Evolving Verifier 分支方向

## [2026-04-08] research | Co-Evolving PRM 文献调研
- 搜索目标：是否已有工作让 PRM 和 policy 在 RL 训练中同时进化
- 找到三篇核心相关工作：
  - **SPARK (Liu et al., 2025/09, arXiv:2509.22624)** — Policy 和 Reward Model 同时训练，有 verifiable reward anchor（非纯 URLVR）
  - **rePIRL (Wu et al., 2026/02, arXiv:2602.07832)** — Inverse RL 框架交替更新 policy 和 PRM，需要 expert demonstrations
  - **Sci-CoE (He et al., 2026/02, arXiv:2602.12164)** — 两阶段 co-evolving：sparse supervision → geometric consensus 自迭代，最接近 URLVR 设定
- **关键理论风险**：纯 URLVR 下 co-evolving 会陷入 mutual sharpening（He et al. Sharpening Theorem 的双重版本）
- SPARK (Liu et al.) 能成功是因为有 verifiable reward 作为硬锚点

## [2026-04-08] synthesis | Co-Evolving Verifier 研究方案
- 创建 wiki/synthesis/co-evolving-verifier-proposal.md（~230行）
- 定位为 SPC 方案的**第二分支方向**
- 核心设计：三层自举架构
  - Layer 1: TTRL outcome anchor（majority voting）
  - Layer 2: SPC step-level signal（expensive but accurate，周期性运行）
  - Layer 3: Lightweight Co-Evolving Verifier（cheap，用 SPC labels 训练，日常替代 probing，被 SPC 周期性校准）
- 策略建议：首发论文做纯 SPC，Co-Evolving 作为第二篇或 Phase 4 扩展
- 更新 index.md 索引、step-level-se-proposal.md 交叉引用

## [2026-04-08] synthesis | SPC 实验设计方案
- 创建 wiki/synthesis/spc-experiment-plan.md
- 明确整体路线：`TTRL baseline -> SPAE-GT -> SPAE-Pseudo-FF -> Confidence / Step Semantic Entropy -> SPC`
- 设计原则：先复现 SPAE 工程骨架，再逐步替换 GT correctness，避免一开始直接做最终版 SPC
- 细化 6 个阶段：环境接入、SPAE-GT 上界、pseudo-label 替换、简单 baseline、SPC 主实验、长期稳定性分析
- 明确给下一个 agent 的实现优先级、最小实验矩阵、评估指标与常见踩坑点
- 更新 index.md 索引

## [2026-04-09] project | 简历项目资料与 Agentic RAG 工程重构
- 创建简历项目文件夹：`wiki/projects/resume/`
- 创建总览页：`wiki/projects/resume/index.md`
- 创建项目手册：`wiki/projects/resume/scholarlens-v2-interview-kit.md`
- 创建项目手册：`wiki/projects/resume/agentic-rag-system-interview-kit.md`
- 更新 `index.md` 项目索引
- 对 `D:/VScodeProject/agentic_rag_system` 完成二次工程化重构：
- 新增 chunk-based ingestion：`agentic_rag/retrieval/chunking.py`、`ingestion.py`
- 重写 hybrid retrieval：dense + sparse + RRF + Cross-Encoder rerank
- 将向量模型从英文 MiniLM 升级为 `BAAI/bge-m3`
- 重写 LangGraph 节点逻辑，改为结构化 `step_type` 工作流
- 重写 Streamlit UI，统一上传入库链路
- 补充 README、Dockerfile、config.yaml、main.py、tests
- 修复懒加载与测试导入问题，`pytest` 通过（4 passed）

## [2026-04-09] interview | 广告搜索算法速成补充
- 创建面试手册：`wiki/interview/ad-search-algorithms.md`
- 覆盖广告/搜索算法核心框架：召回、粗排、精排、CTR/CVR/eCPM、GSP/OCPX、多目标优化、偏差与校准
- 补充如何将现有两个项目映射到广告搜索算法面试表达
- 更新 `index.md` 面试索引

## [2026-04-09] interview | 广告搜索算法题库与速背材料
- 创建题库：`wiki/interview/ad-search-algorithms-qa.md`
- 创建速背版：`wiki/interview/ad-search-algorithms-one-day-cram.md`
- 补充 50 道广告搜索算法高频问答
- 补充 1 天突击版标准答法与项目迁移话术
- 更新 `index.md` 面试索引

## [2026-04-10] interview | Linux 常用命令速查手册
- 创建面试文档：`wiki/interview/linux-commands.md`
- 覆盖 13 大类：GPU/显卡监控（nvidia-smi、gpustat、nvtop、CUDA）、内存/CPU 监控（free、htop、vmstat）、进程管理（ps、kill、nohup、tmux/screen）、文件操作、磁盘管理、网络/SSH/SCP、压缩解压、系统信息、Python/Conda 环境管理、文本处理、深度学习实用组合命令、面试 Q&A、快捷键技巧
- 重点场景：查看谁在用 GPU、远程训练标准流程、OOM 排查、批量杀残留进程
- 更新 `index.md` 面试索引

## [2026-04-10] ingest | MAE 论文摄入
- 论文：Multi-Agent Evolve (Chen & Wang et al., 2025, arXiv:2510.23595) — UIUC / PKU / NVIDIA
- 核心：Proposer-Solver-Judge 三角色从同一 LLM co-evolution，无 GT/外部 verifier，通用领域自进化
- 创建论文页：wiki/papers/chen-2025-mae.md（完整版：方法/公式/实验表格/Ablation/与 Co-Evolving Verifier 提案对比/Mutual Sharpening 分析）
- 更新 index.md 索引（新增 "Self-Play / Multi-Agent Co-Evolution" 分类）
- 与研究的关系：MAE 验证了 co-evolution 可行性，但缺乏外部锚点，Response-level 粒度有天花板，为 SPC step-level 方案提供了差异化定位

## [2026-04-10] ingest | VPPO 论文摄入
- 论文：VPPO — Save the Good Prefix (Liu et al., 2025, arXiv:2601.18984) — Tencent AI Lab / UVA
- 核心：PRM 仅做 first error detection，将轨迹划分为 good prefix + erroneous suffix，精准 credit assignment
- 创建论文页：wiki/papers/liu-2025-vppo.md（完整版：方法/Eq. 6-8 公式/Shorten Prefix/Theorem 1/实验表格/消融/与 SPC 关系分析/面试 Q&A）
- 更新 index.md 索引（添加到 Step-Level Credit Assignment 分类）
- 与研究的关系：VPPO 需要 GT + PRM（非 URLVR），但 "first divergence localization" 思想可迁移到 SPC——用 consistency 急剧下降点替代 PRM first error detection

## [2026-04-10] ingest | TraPO 论文摄入
- 论文：TraPO — A Semi-Supervised RL Framework for Boosting LLM Reasoning (Yang et al., 2025, arXiv:2512.13106) — Zhejiang University / Ant Group
- 核心：首创半监督 RLVR 范式，用 pass rate trajectory cosine similarity 从标注样本引导无标注样本选择
- 创建论文页：wiki/papers/yang-2025-trapo.md（完整版：方法/6 个关键公式/训练 pipeline/Table 1-8 实验数据/Scaling Law/超参敏感性/训练成本/稳定性/与 He et al. sharpening 理论的关系/面试 Q&A/与 SPC 研究的潜在结合）
- 更新 index.md 索引（新建 Semi-Supervised RLVR 分类）
- 关键数据：1K 标注 + 3K 无标注 → ID Avg 42.6%（超越 45K 无监督的 38.3% +4.3%）；4K + 12K → ID 45.6%/OOD 59.7%（超越 45K 全监督的 45.5%/57.3%）
- 与研究的关系：TraPO 从实践验证了 sharpening theorem 的局限性，证明少量外部监督即可突破 URLVR 天花板；trajectory matching 思想可与 SPC step-level 方案结合

## [2026-04-10] ingest | ETTRL 论文摄入
- 论文：ETTRL — Balancing Exploration and Exploitation in LLM Test-Time RL Via Entropy Mechanism (Liu et al., 2025, arXiv:2508.11356) — Kuaishou / Beihang / Northwestern Polytechnical
- 核心：TTRL 直接改进版，两个组件：(1) ETMR: 高熵 token 位置分叉的 tree rollout，仅需 60% token budget 获得同等 rollout 数且多样性更高；(2) EAR: 相对熵重塑 advantage，低熵高置信响应 advantage 放大，高熵低置信响应 advantage 抑制
- 创建论文页：wiki/papers/liu-2025-ettrl.md（完整版：方法/公式/Table 1-2/对 SPC 实验的启示）
- 关键数据：Llama3.1-8B AIME24 +69%（10.0→16.9），Qwen2.5-Math-1.5B +32.9%（15.8→21.0），全部仅需 60% token budget
- 与研究的关系：ETTRL 可作为 SPC 实验的改进版 TTRL baseline；ETMR 的"高熵 token 决定分叉"洞察可用于 SPC probing；EAR 可缓解 SPC step-level advantage 的 sharpening

## [2026-04-10] ingest | CAPO 论文摄入
- 论文：CAPO — Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment (Xie et al., 2025, arXiv:2508.02298) — Renmin University / Tencent WeChat
- 核心：用 off-the-shelf 70B LLM 作为生成式 PRM（LLM-as-GenPRM），单次推理生成所有步骤正误判定 + 多次 critique 投票降噪 + 非对称奖励塑形（W_whole > W_process）
- 创建论文页：wiki/papers/xie-2025-capo.md（完整版：方法/公式/实验表格/Voting 策略/非对称权重 ablation）
- 关键数据：Qwen2.5-7B AIME24 从 3.6 到 10.8（+200%），Llama-3-3B AMC23 +59%
- 与研究的关系：CAPO 的非对称奖励塑形（outcome > process）可迁移到 SPC；投票降噪策略对 SPC probing 有参考价值；但依赖外部 70B 模型 + GT answer，非 URLVR

## [2026-04-10] cleanup | VPPO 重复文件清理
- 合并 wiki/papers/xie-2026-vppo.md 中独有的 SPC 分析（计算开销降低策略、TTRL 双层兼容、SPC 可靠性空间结构、面试频率标签）到 wiki/papers/liu-2025-vppo.md
- 删除重复文件 wiki/papers/xie-2026-vppo.md（作者前缀错误 xie→应为 zhou，且 liu-2025 版本内容更全面准确）
- 更新 index.md VPPO 链接：xie-2026-vppo → liu-2025-vppo，修正作者为 Liu et al., 2025
