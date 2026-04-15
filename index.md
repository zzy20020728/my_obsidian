---
title: Wiki 索引
type: index
updated: 2026-04-15
---

# Wiki 索引

## 论文 (wiki/papers/)
> 论文摘要与深度分析

### RL 基础
- [[wiki/papers/schulman-2017-ppo|PPO 原论文 (Schulman et al., 2017)]] — 策略梯度经典，RLHF 核心算法

### URLVR（无监督/无参考 RL 推理）
- [[wiki/papers/zuo-2025-ttrl|TTRL (Zuo & Zhang et al., 2025)]] — ⭐ Majority voting 做 pseudo-reward，test-time self-evolution，Qwen2.5-Math-7B AIME +211%
- [[wiki/papers/liu-2025-ettrl|ETTRL (Liu et al., 2025)]] — ⭐ TTRL 改进版：entropy-fork tree rollout（60% token budget）+ entropy advantage reshaping（抑制早期 overconfidence），Llama3.1-8B AIME +68%
- [[wiki/papers/zhang-2025-empo|EMPO (Zhang et al., 2025)]] — 语义熵最小化，完全无监督，仅需 {q} 不需要 {a}
- [[wiki/papers/zhang-2025-covo|CoVo (Zhang et al., 2025)]] — Consistency + Volatility，显式建模中间状态是否持续支持最终答案
- [[wiki/papers/rahman-2025-spark|SPARK (Rahman et al., 2025)]] — 三阶段 PRM 训练，step-level reward 超越 GT RLVR
- [[wiki/papers/ghimire-2026-prism|PRISM (Ghimire et al., 2026)]] — 系统性证明纯内部信号不可靠，PRM + self-certainty 混合方案
- [[wiki/papers/wu-2026-self-judge|Self-Judge (Wu et al., 2026)]] — Actor-Judge 多模态无监督自进化，distributional reward modeling
- [[wiki/papers/royer-2026-mcnig|MCNIG (Royer et al., 2026)]] — 信息论自动生成 PRM 训练数据，O(N) 复杂度，跨任务泛化
- [[wiki/papers/he-2026-urlvr-scale|How Far Can URLVR Scale? (He et al., 2026)]] — ⭐ICLR 2026，URLVR 统一理论框架，sharpening mechanism，MCS 概念

### URLVR Reward Estimation（改进 TTRL Reward 质量）
- [[wiki/papers/du-2026-dual-consensus|DCRL (Du et al., 2026)]] — ⭐ 双阶段投票逃离 TTRL 虚假多数：clone anchor + unlearn explorer → harmonic mean 共识 → 三级 reward + adaptive sampling，Qwen3-8B Avg 50.9
- [[wiki/papers/du-2026-dare|DARE (Du et al., 2026, ICML)]] — ⭐ 分布感知 reward 替代 point-level MV：uncertainty-normalized distribution + exploration bonus + pruning，AIME24 +25.3% over TTRL
- [[wiki/papers/liao-2026-t3rl|T³RL (Liao et al., 2026)]] — 工具验证锚定 TTRL reward：Verifier LLM → Python 代码 → code interpreter 执行 → 加权 MV（ω=5），AIME24 +31.6% over TTRL
- [[wiki/papers/pan-2026-coverrl|CoVerRL (Pan et al., 2026)]] — ⭐ 单一模型 generator-verifier 共演化突破 consensus trap：MV 提供对比信号训练 verifier → verifier 过滤 self-consistent errors，verification accuracy 55%→85%，超越 TTRL 4.7-5.9%
- [[wiki/papers/yan-2026-scrl|SCRL (Yan et al., 2026)]] — ⭐ 首次 TTRL 负 pseudo-labeling：selective positive + entropy-gated negative supervision，受限 rollout 预算下 AIME25 +5.8%，Minerva +27.1%
- [[wiki/papers/yang-2026-distribttrl|DistriTTRL (Yang et al., 2026)]] — GMM 建模置信度分布 + shift correction 聚合历史 rollout + diversity penalty 防 reward hacking，AIME24 +7.50
- [[wiki/papers/bai-2026-ttvs|TTVS (Bai et al., 2026)]] — ⭐ 动态增广测试数据为语义等价变体 + hybrid IGE/CGE 探索，无标签超越有监督 RLVR（1.5B > DeepSeek-R1-Distill-7B）
- [[wiki/papers/kim-2026-dbb|DBB (Kim et al., 2026)]] — Beta-Bernoulli 贝叶斯 reward estimation，利用历史统计降低方差，OOD +12.49%（1.7B）/ +6.92%（8B），零额外计算

### URLVR 分布匹配（Distribution Matching 范式）
- [[wiki/papers/chen-2026-powerflow|PowerFlow (Chen et al., 2026)]] — ⭐ GFlowNet + α-power 分布匹配的无监督微调：LA-TB 目标消除长度偏差，α>1 锐化推理匹配/超越 GRPO，α<1 平坦化释放创造力，证明 MV-RLIF 等价极端锐化

### RLVR Training Stability（训练稳定性与优化改进）
- [[wiki/papers/ma-2026-dcpo|DCPO (Ma et al., 2026)]] — 证明 accuracy-calibration gradient conflict 并解耦：masked gradient 分别优化 reasoning/confidence tokens，ECE -71.6% 且 accuracy 不降
- [[wiki/papers/wang-2026-pipo|PIPO (Wang et al., 2026)]] — ⭐ 暴露 GRPO 梯度爆炸缺陷 η(p)→∞，提出 PIRL 框架 + dual-stage explore-verify，跨 GRPO/GSPO/DAPO 一致提升
- [[wiki/papers/cui-2026-clipo|CLIPO (Cui et al., 2026, Alibaba Qwen)]] — InfoNCE 对比学习抑制 spurious reasoning：successful rollouts 表示聚拢 + reward augmentation，跨 4 种 RL 算法一致提升
- [[wiki/papers/gu-2026-asymgrpo|AsymGRPO (Gu et al., 2026)]] — 将 entropy 分解为 informative vs spurious，非对称调制正负 rollouts，Qwen3-4B avg 60.32%（+3.82% over GRPO）
- [[wiki/papers/ai-2026-shape|SHAPE (Ai et al., 2026)]] — 层次化 credit assignment：段级 solvability potential + token 级 entropy 重分配，+3% accuracy with 30% less tokens

### Label-Free / Open-Domain RL（无标注/开放域 RL）
- [[wiki/papers/wang-2026-sarl|SARL (Wang et al., 2026)]] — ⭐ 推理拓扑结构做 reward：小世界网络 SR(G)=0.5·C+1/(1+L)，完全 label-free，Qwen3-4B 数学超越 GT RL，WildBench +9.10
- [[wiki/papers/huang-2026-darl|DARL (Huang et al., 2026)]] — 动态多样性 reward 鼓励多样正确路径，General +9.5（需要 GT，非纯 URLVR）

### Label Noise Robustness（标签噪声鲁棒性）
- [[wiki/papers/yang-2026-olr|OLR (Yang et al., 2026)]] — ⭐ 系统分析 RLVR 中 noisy labels（inactive vs active 两类），Early Correctness Coherence 现象，Online Label Refinement 渐进自纠正，robust across noise ratio 0.1-0.9
- [[wiki/papers/plesner-2026-imperfect-verifier|Imperfect Verifier (Plesner et al., 2026)]] — 证明 RLVR 对 noisy verification 鲁棒：15% noise rate 仍在 clean baseline 2pp 内，"moderate accuracy + high precision" 原则

### Multimodal Self-Improvement（多模态自进化）
- [[wiki/papers/tan-2026-meta-ttrl|Meta-TTRL (Tan et al., 2026)]] — TTRL 扩展到 T2I：元认知架构（rubric-based 二值验证 + 几何平均 reward），自我内省(7B) > 外部强模型(235B)
- [[wiki/papers/wang-2026-v-zero|V-Zero (Wang et al., 2026)]] — ⭐ 零标注 VLM 自进化：Questioner-Solver co-evolution + Dual-Track Reasoning Reward（intuition vs reasoning），无监督超越有监督 GRPO
- [[wiki/papers/yu-2026-csrs|CSRS (Yu et al., 2026)]] — 多模态无监督自进化：Retracing Re-inference + Softened Frequency Reward + Visual Semantic Perturbation，几何任务 SOTA

### Semi-Supervised RLVR
- [[wiki/papers/yang-2025-trapo|TraPO (Yang et al., 2025)]] — ⭐ 首创半监督 RLVR：trajectory matching 选可靠无标注样本，1K 标注 + 3K 无标注超越 45K 无监督，10% 标注超全量监督

### Self-Play / Multi-Agent Co-Evolution
- [[wiki/papers/chen-2025-mae|MAE (Chen & Wang et al., 2025)]] — Proposer-Solver-Judge 三角色 co-evolution，无需 GT/外部 verifier，Qwen2.5-3B +4.54%
- [[wiki/papers/wang-2026-self-guide|Self-Guide (Wang et al., 2026)]] — Policy-reward co-evolution：同一模型生成 internal reward 用于 inference-time guidance 和 training-time step-level reward，+8% over environment-reward-only baselines

### Step-Level Credit Assignment / Efficient Reasoning
- [[wiki/papers/wu-2026-spae|SPAE (Wu et al., 2026)]] — Training-free probing 提取 Step Potential，step-level advantage estimation，解决 Over-Checking 和 R2W 失败
- [[wiki/papers/liu-2025-vppo|VPPO (Liu et al., 2025)]] — PRM 仅做 first error detection，partition 轨迹为 good prefix + erroneous suffix，精准 credit assignment，理论证明指数级样本效率提升
- [[wiki/papers/xie-2025-capo|CAPO (Xie et al., 2025)]] — LLM-as-GenPRM 单次推理生成所有步骤判定，非对称奖励塑形平衡 outcome 与 process 信号，token-level credit assignment
- [[wiki/papers/zhang-2026-praise|PRAISE (Zhang et al., 2026)]] — ⭐ Prefix rollout reuse + adjacent prefix gains 做 step reward，agentic search 5 个 benchmark 全面提升，与 SPC 思想最接近的已有工作
- [[wiki/papers/chen-2026-slate|SLATE (Chen et al., 2026)]] — Shared prefix 做 k 个 continuation 估计 step-level advantage，理论证明方差降低 T 倍，LLM judge 做 dense reward
- [[wiki/papers/liang-2026-step|STEP (Liang et al., 2026)]] — Hidden states 训练 lightweight MLP step scorer，前 25% 步骤已能区分正确/错误 traces，45%–70% 推理延迟降低

### Open-ended RL / Dense Reward
- [[wiki/papers/zhang-2026-grad2reward|Grad2Reward (Zhang et al., 2026)]] — 梯度 attribution 提取 dense token-level reward，self-judging，ICML 2026

### RAG + RL
- [[wiki/papers/wang-2026-prorag|ProRAG (Wang et al., 2026)]] — 四阶段 RAG RL，MCTS-based PRM，dual-granularity advantage
- [[wiki/papers/tan-2026-ctrl-rag|CTRL-RAG (Tan et al., 2026)]] — Contrastive Likelihood Reward，轻量 RAG faithfulness 信号

## 概念 (wiki/concepts/)
> 核心概念与知识点

### RL 算法
- [[wiki/concepts/ppo|PPO]] — 近端策略优化，RLHF 中最常用的 RL 算法
- [[wiki/concepts/grpo|GRPO]] — 组相对策略优化，URLVR 四篇论文的共同优化框架

### 对齐技术
- [[wiki/concepts/rlhf|RLHF]] — 基于人类反馈的强化学习，LLM 对齐核心技术

### URLVR 核心概念
- [[wiki/concepts/semantic-entropy|Semantic Entropy (语义熵)]] — EMPO 的核心 reward 信号
- [[wiki/concepts/self-consistency|Self-Consistency (自一致性)]] — 多路径采样一致性，URLVR 基础信号
- [[wiki/concepts/process-reward-model|Process Reward Model (PRM)]] — 步骤级奖励模型，SPARK/PRISM/MCNIG/ProRAG 核心
- [[wiki/concepts/reward-hacking|Reward Hacking (奖励攻击)]] — URLVR 最大挑战，所有论文都涉及
- [[wiki/concepts/information-gain|Information Gain (信息增益)]] — MCNIG 的核心信息论度量
- [[wiki/concepts/mcts|MCTS (蒙特卡洛树搜索)]] — ProRAG 的 PRM 训练数据探索方法
- [[wiki/concepts/contrastive-likelihood|Contrastive Likelihood (对比似然)]] — CTRL-RAG 的 faithfulness reward 信号

## 实体 (wiki/entities/)
> 模型、团队、公司、人物

*尚无条目*

## 面试 (wiki/interview/)
> 面试知识点整理

- [[wiki/interview/llm-and-rl|大模型 & 强化学习面试题]] — 算法岗核心战场
- [[wiki/interview/ml-basics|机器学习基础面试题]] — ML 八股
- [[wiki/interview/ad-search-algorithms|广告搜索算法面试速成手册]] — 召回/粗排/精排/拍卖/指标/项目迁移表达
- [[wiki/interview/ad-search-algorithms-qa|广告搜索算法高频 Q&A]] — 50 道高频问答，适合面试前快速刷题
- [[wiki/interview/ad-search-algorithms-one-day-cram|广告搜索算法 1 天速背版]] — 最小可用答案集，适合临场突击
- [[wiki/interview/linux-commands|Linux 常用命令速查手册]] — GPU 监控、内存/进程管理、远程训练、环境管理，面向 DL 研究场景
- [[wiki/interview/java-and-programming|Java & 编程面试题]] — 应用开发岗需要

## 项目 (wiki/projects/)
> 项目经验与总结

- [[wiki/projects/resume/index|简历项目面试资料总览]] — 面向秋招/实习面试的项目手册入口
- [[wiki/projects/resume/scholarlens-v2-interview-kit|ScholarLens v2 面试手册]] — 多智能体学术检索系统，含实现过程与高频追问
- [[wiki/projects/resume/agentic-rag-system-interview-kit|Agentic RAG 实习项目面试手册]] — Agentic RAG 二次工程化重构，含升级点与面试口径

## 综合分析 (wiki/synthesis/)
> 跨论文对比与领域综述

- [[wiki/synthesis/urlvr-landscape|URLVR 领域综述]] — 二十篇核心论文多维分类对比，信号来源/打分粒度/Reward 模型/任务类型/Sharpening 理论
- [[wiki/synthesis/step-level-se-proposal|Semantic Process Consistency 研究方案]] — 从 SPAE-SE 重构为 SPC：用 step-level semantic rollout consistency 纠正 TTRL 的 outcome-only 信号
- [[wiki/synthesis/co-evolving-verifier-proposal|Co-Evolving Verifier 研究方案]] — SPC 的第二分支：让 PRM 跟着 RL 训练一起进化，三层自举架构（TTRL anchor → SPC signal → lightweight PRM）
- [[wiki/synthesis/spc-experiment-plan|SPC 实验设计方案]] — 从 TTRL 环境复现 SPAE 骨架开始，逐步替换 GT correctness，最终落到 SPC 的分阶段实现路线
- [[wiki/synthesis/spc-related-work-survey|SPC 相关工作文献调研报告]] — 20+ 轮 arXiv 搜索，37 篇论文按相关度分级（11 高/15 中/11 低），方法族谱系图，新颖性确认

## 学习计划 (plans/)
> 秋招备战计划与进度

- [[plans/master-plan|秋招备战总计划]]
- [[plans/weekly-tracker|每周进度追踪]]
- [[plans/obsidian-tutorial|Obsidian & Wiki 使用教程]]

---
*本索引由 LLM 自动维护，每次 ingest/query 操作后更新*
