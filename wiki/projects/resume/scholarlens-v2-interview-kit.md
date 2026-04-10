---
title: ScholarLens v2 面试手册
type: project
tags: [project, llm, retrieval, multi-agent, fastapi, interview]
created: 2026-04-09
updated: 2026-04-09
sources:
  - "D:/VScodeProject/thesis_retrieval_MCP/ScholarLens_v2_面试准备.md"
  - "D:/VScodeProject/thesis_retrieval_MCP/src_v2/"
  - "D:/VScodeProject/thesis_retrieval_MCP/configs/settings.yaml"
status: active
---

# ScholarLens v2 面试手册

## 项目定位

这是一个面向科研检索场景的多智能体学术搜索系统。

最稳的项目定义是：

> 我把一个偏 ReAct/MCP 工具调用风格的学术检索原型，重构成了可部署的多智能体检索后端。系统围绕复杂科研问题，完成了 Query Planning、多源并行召回、RRF 融合、Cross-Encoder 精排、Judge 驱动评估、Refine 自适应改写和结构化综述生成，并通过 FastAPI + Docker 交付为服务化能力。

这句话的好处是既能体现项目升级，又不会和真实代码冲突。

## 简历写法

### 一段话版本

【ScholarLens v2：多智能体学术检索系统】

基于 FastAPI 重构科研检索后端，围绕复杂学术问题设计 QueryAgent 查询规划、ArXiv/Semantic Scholar/OpenAlex 多源并行检索、RRF 融合、Cross-Encoder 重排、JudgeAgent 质量评估与 RefineAgent 自适应改写闭环，支持结构化文献综述生成、差异化模型配置与 Docker 部署，完成从 ReAct 原型到服务化后端的升级。

### 30 秒版本

这个项目本质上是一个科研检索后端。v1 更像 ReAct 风格的工具调用原型，重点是验证 LLM 能不能驱动检索；v2 我重点把它重构成多智能体服务化架构，把查询规划、并行检索、结果融合、质量评估、改写重试和综述生成拆成独立模块，通过 FastAPI 对外暴露统一接口，更接近真实上线系统。

### 2 分钟版本

这个项目主要解决两个问题。第一，科研用户输入的是自然语言问题，不是标准关键词，直接搜很容易漏召回。第二，单一数据源结果不稳定，复杂问题很难一次搜全。

我做的升级有三层。第一是算法链路，把原来单 Agent 的 ReAct 流程拆成 QueryAgent、JudgeAgent、RefineAgent、SynthesisAgent 四类角色，由 Orchestrator 统一编排。第二是检索链路，接入 ArXiv、Semantic Scholar、OpenAlex 三个真实学术源，做多源并行召回、RRF 融合和 Cross-Encoder 精排。第三是工程化，把系统做成 FastAPI 服务，补上配置中心、Docker、缓存层、评测脚手架和模型工厂，让它从 demo 变成一个可交付后端。

## 系统架构

系统主链路是：

前端或调用方
-> FastAPI API
-> Orchestrator 编排层
-> QueryAgent 生成结构化检索计划
-> RetrievalEngine 多源并行检索
-> RRF 融合
-> Cross-Encoder 重排
-> JudgeAgent 评估质量
-> RefineAgent 生成下一轮查询
-> SynthesisAgent 输出结构化综述

代码里最关键的对象是：

- `Paper`：统一论文对象，屏蔽不同学术源响应格式差异。
- `SearchState`：共享状态，保存子查询、原始结果、排序结果、CoT 推理链、历史 top10、收敛指标等。
- `Evaluation`：JudgeAgent 的评估输出，包括相关性、主题覆盖、缺失主题和建议。
- `SearchResult`：最终对外结果。

## 完整实现过程

### 1. 先明确 v1 的问题

我先把 v1 的问题拆成三类：

- 查询理解、结果评估、查询改写都塞在一个 Agent 里，prompt 容易相互干扰。
- 单轮工具调用在复杂科研问题下不稳，缺少“搜得够不够好”的判断机制。
- 代码形态偏脚本，不适合前端接入、部署和持续迭代。

所以 v2 的目标不是“多接几个 API”，而是把检索过程拆成可解释、可评估、可迭代的服务化流水线。

### 2. 建统一数据模型

我先在 `src_v2/models.py` 建了统一的数据抽象。

- `Paper` 统一承接 title、authors、year、abstract、source、doi、arxiv_id、citation_count 等字段。
- 去重主键 `canonical_id` 的优先级是 DOI > arXiv ID > URL，这比只按标题去重更稳。
- `SearchState` 作为共享黑板，贯穿多轮检索全过程。

这一步的意义是把后续所有 Agent 和检索源都收敛到统一对象上，否则多源融合、去重、重排都很难稳定。

### 3. 设计 QueryAgent 的两阶段 CoT

`QueryAgent` 不是直接吐关键词，而是拆成两段。

- Stage 1 Reasoning：先做意图分析、概念分解、搜索难度分析和策略制定，输出 `QueryReasoning`。
- Stage 2 Planning：基于 Stage 1 的推理摘要，生成 3-5 条带类型的子查询，输出 `QueryPlan`。

这里我专门用了 Pydantic + `JsonOutputParser`，强制模型输出结构化 JSON，而不是自由文本。这样推理链后面可以被 RefineAgent 复用，不需要靠正则做脆弱解析。

### 4. 做多源并行检索与统一适配

`RetrievalEngine` 接了三个真实学术源：

- ArXiv：偏预印本和前沿方向。
- Semantic Scholar：元数据更丰富，还有 citation graph 接口。
- OpenAlex：补充论文元数据、领域标签和开放学术图谱信息。

关键点不是“能搜三个源”，而是每个源都做 adapter，最后全部映射到统一 `Paper` 模型。上层编排逻辑永远只面向 `Paper`，不依赖底层 API 的字段格式。

### 5. 做去重、融合和精排

结果返回后，我做了三层处理。

第一层是去重。

- 优先用 DOI。
- DOI 没有时用 arXiv ID。
- 再退化到 URL。
- 如果同一篇论文在多个源里出现，保留元数据更丰富的一份。

第二层是 RRF 融合。

我没有强行比较不同源的原始分数，而是用 rank-based 的 Reciprocal Rank Fusion，因为异构源分数分布不可比，RRF 在工程上更稳。

第三层是 Cross-Encoder 精排。

当前代码里真实落地的是 `query + title + abstract` 配对后进入 Cross-Encoder 打分。这里面试时一定要讲真实口径：当前主链路真正落地的是 RRF + Cross-Encoder，而不是把配置里预留的多阶段排序都说成已上线。

### 6. 做 Judge/Refine 闭环

JudgeAgent 的任务不是只看“有没有几篇相关论文”，而是判断结果集是否已经足够回答用户问题。

它主要看三类信息：

- 平均相关性是否够高。
- topic coverage 是否达标。
- missing topics 是否已经足够少。

如果 Judge 判定不够好，就由 RefineAgent 基于三样东西生成下一轮查询：

- 初始 CoT 推理链
- 已尝试的查询计划
- Judge 给出的缺失主题和建议

这也是项目最值得讲的技术亮点：它不是静态 query rewrite，而是带诊断信息的闭环优化。

### 7. 设计 Orchestrator 的终止策略

`Orchestrator` 不是固定跑三轮，而是做自适应终止。

停止条件有四层：

- 最大迭代次数兜底。
- marginal gain 过低，说明新结果增益不大。
- top10 overlap 过高，说明结果已经收敛。
- Judge 质量达到阈值，直接结束。

代码层面还有一个容易被忽略但很好讲的点：这里其实是两层并行。

- 第一层是 QueryAgent 生成多条子查询后，`Orchestrator` 对子查询并发。
- 第二层是 `RetrievalEngine` 内部对三个学术源再并发。

所以它不是简单串行工具调用，而是 query-level 和 source-level 双层并发检索。

### 8. 做服务化与配置化

最后我把整条链路封成 FastAPI 服务。

- `/api/v1/search`：完整多智能体链路。
- `/api/v1/search/quick`：单轮快速路径。
- `/health`：健康检查。

同时通过 `settings.yaml` 和 `LLMFactory` 支持按 Agent 配不同 provider、model 和 temperature。这样 QueryAgent、JudgeAgent、RefineAgent、SynthesisAgent 可以走不同模型组合，更符合真实工程里的成本/效果平衡。

部署层我做成了 Docker + uvicorn，缓存层预留 Redis，评测层补了 NDCG、MAP、MRR、Recall、Precision 等基础 IR 指标实现。

## 真实代码对齐的亮点

### 1. Blackboard 模式

多 Agent 不直接互相强耦合，而是统一读写 `SearchState`。这样链路更容易调试、扩展和复盘。

### 2. Per-Agent 模型配置

`LLMFactory` 支持按 Agent 独立配置模型和温度。这个点很适合讲“强推理模型用于 Query/Judge，稍高温度用于 Refine，语言组织用于 Synthesis”。

### 3. TerminationPolicy

停止不是拍脑袋设固定轮数，而是结合 marginal gain、结果收敛和 Judge 质量一起决定。

### 4. 统一 Paper Schema

这是多源检索系统最基础但最容易被忽略的部分。没有统一 schema，后面去重、融合、重排都不稳。

## 高频 Q&A

### Q1：这个项目到底解决了什么问题？

答：它解决的是复杂科研问题一次检索不准、不全的问题。用户输入通常是自然语言研究问题，而不是标准检索式，所以系统先做查询理解和查询规划，再做多源并行召回、结果评估和自动改写，目标不是“搜到几篇论文”，而是“用稳定的检索策略把问题搜全并组织成可读结果”。

### Q2：v1 和 v2 最大差别是什么？

答：v1 重点是证明 LLM 驱动检索这件事可行；v2 重点是把它做成稳定、可调用、可部署的后端系统。架构上从单 Agent 走向多 Agent 编排；检索上从单轮工具调用走向多源并行 + RRF + Judge/Refine 闭环；工程上从脚本走向 FastAPI + Docker + 配置中心。

### Q3：为什么不继续用单个 ReAct Agent？

答：因为查询理解、结果评估和查询改写是不同认知任务。全部塞在一个 prompt 里，模型很容易不稳定。拆成多个 Agent 后，每个模块的输入输出更明确，prompt 更稳定，也便于独立调参和替换模型。

### Q4：完整链路怎么走？

答：用户请求进入 FastAPI；Orchestrator 初始化 `SearchState`；QueryAgent 生成 3-5 条子查询；RetrievalEngine 并发访问三个学术源；结果去重并用 RRF 融合；Cross-Encoder 精排；JudgeAgent 判断质量；如果不满意，就让 RefineAgent 生成下一轮查询；满意后交给 SynthesisAgent 生成综述。

### Q5：为什么 RRF 适合这个场景？

答：因为不同学术源的原始分数不可比。有的接口给相似度分，有的给内部相关度，有的几乎只有排序。RRF 只依赖 rank，不强依赖原始分值，对多源异构结果融合更稳。

### Q6：怎么去重？

答：用 `Paper.canonical_id`。优先 DOI，其次 arXiv ID，再退化到 URL。对于重复论文，保留元数据更丰富的一份，比如有 citation_count、venue、fields_of_study 的版本。

### Q7：JudgeAgent 具体怎么决定要不要停？

答：一方面看平均相关性和 topic coverage，另一方面结合 missing topics 数量。工程上再叠加 marginal gain、结果收敛和最大迭代次数三个维度，形成自适应终止，而不是固定轮数。

### Q8：多源检索为什么一定要并发？

答：一是为了压时延，二是为了保证复杂问题下的召回上限。代码里实际上是两层并发：多个 sub-query 并发，每个 sub-query 内部又对多个 source 并发。

### Q9：当前重排到底用什么模型？

答：真实落地的是 Cross-Encoder。配置里有 bi-encoder 和 llm_judge 的预留，但当前主链路和最稳口径是“RRF 后接 Cross-Encoder 精排”。

### Q10：为什么按 Agent 配不同模型？

答：因为任务性质不同。QueryAgent 和 JudgeAgent 更强调稳定推理，temperature 更低；RefineAgent 允许适度发散；SynthesisAgent 更重视语言组织。按 Agent 差异化配置能更好地做成本/效果平衡。

### Q11：为什么更强的 reranker 不能解决所有问题？

答：因为 reranker 只能给已有候选排序，不能把没召回进来的论文“变出来”。系统上限首先取决于查询质量和候选池质量，所以 Query Planning 和多源召回比单独堆 reranker 更关键。

### Q12：项目里最有技术含量的一点是什么？

答：我会讲 QueryAgent + JudgeAgent + RefineAgent 的闭环。难点不是接 API，而是系统如何判断“现在搜得够不够好、缺在哪、下一轮怎么改”。这部分体现的是检索系统设计，而不是简单工具调用。

### Q13：当前有哪些短板？

答：我会诚实说三点。第一，`QueryCache` 已设计但还没有完整接进主链路；第二，配置里预留了 sparse/dense 本地索引和多阶段 reranking，但当前主实现重点仍是外部学术源 + Cross-Encoder；第三，API 还没有做流式返回和更完整的可观测性。

### Q14：如果继续升级，下一步会怎么做？

答：我会往三条线走。第一，做真正的 Hybrid Retrieval，把本地稀疏/稠密召回也纳入主链路；第二，引入 citation graph expansion，把检索从纯文本相似度扩展到引文图谱；第三，做 section/chunk 级证据下钻，让最终生成回答可以定位到论文内部证据，而不只是 paper-level 摘要。

## 不能主动说死的内容

- 不要主动报没有压测支撑的 QPS。
- 不要主动报没有系统 benchmark 的具体数字。
- 不要把配置里“预留但未完全打通”的能力说成已上线。
- 不要把“支持校内场景接入”夸大成“大规模全校使用”。

## 最稳收尾

> 这个项目我最想表达的不是“我做了一个搜论文工具”，而是“我把一个针对复杂科研问题的检索原型，做成了有查询规划、结果评估、自动改写和服务化交付能力的后端系统”。如果需要，我可以继续展开讲 Query Planning、RRF 融合、Judge 驱动迭代，或者它如何从 ReAct 原型演进到服务化架构。
