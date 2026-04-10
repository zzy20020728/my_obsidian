---
title: Agentic RAG 实习项目面试手册
type: project
tags: [project, rag, langgraph, hybrid-retrieval, interview, internship]
created: 2026-04-09
updated: 2026-04-09
sources:
  - "D:/VScodeProject/agentic_rag_system/"
status: active
---

# Agentic RAG 实习项目面试手册

## 项目定位

这是一个面向企业知识库问答场景的 Agentic RAG 系统，技术栈是 `LangGraph + Streamlit + ChromaDB + 文档解析 + Human-in-the-loop`。

这次我重点做的不是“补几行功能”，而是把一个比较粗糙的原型做了二次工程化重构：补齐切块入库、混合检索、结构化工作流、统一 ingestion pipeline、基础测试、README 和 Docker，把它从“演示型项目”升级成“可以讲清楚工程逻辑的简历项目”。

## 推荐的简历写法

【Agentic RAG：企业知识库智能问答系统】

基于 LangGraph 设计企业知识库问答工作流，围绕文档解析、混合检索和多步推理完成二次工程化重构：将原始整篇文档入库升级为 chunk-based ingestion，引入 `BAAI/bge-m3` 向量检索、稀疏关键词召回、RRF 融合和 Cross-Encoder 精排，重构 Planner/Executor/Critic/Generator 节点为结构化执行链路，并补充 Streamlit 上传入库、MCP 工具链一致性、Docker 与基础测试，使系统更接近真实企业知识助手。

## 升级前的问题

原始版本最明显的几个问题是：

- 文档整篇直接入库，长文档检索粒度太粗。
- 向量模型是 `all-MiniLM-L6-v2`，偏英文，不适合中文知识库。
- `HybridSearch` 实际上只有向量召回 + rerank，不是真正 hybrid。
- Planner 输出是自由文本，Executor 靠字符串匹配决定步骤类型，稳定性一般。
- UI、工具层和检索层各自有一套逻辑，入库和检索链路不一致。
- README、Docker、测试基本空白，工程可信度不够。

所以我这次改造的目标很明确：围绕“检索质量、工作流稳定性、工程完整性”三条线重做。

## 升级后的主线

### 1. 切块入库

我新增了：

- `agentic_rag/retrieval/chunking.py`
- `agentic_rag/retrieval/ingestion.py`

核心变化是把“整篇文档直接入库”改成“按 chunk 入库”。切块逻辑优先按段落边界分块，超长段落再做滑窗切分，并保留 overlap。这样做的价值是：

- 提高召回粒度，避免整篇文档相似但局部证据不相关。
- 给 reranker 更短、更聚焦的候选文本。
- 为后续做 chunk-level attribution 和 context compression 打基础。

### 2. 向量层升级到 `BAAI/bge-m3`

我把 `VectorStore` 从原来的英文 MiniLM 嵌入切到 `BAAI/bge-m3`，并统一读取 `AppConfig` 中的可配置模型。

这样做的原因是：

- `bge-m3` 更适合中英文混合语料。
- 作为检索模型更贴近中文企业知识库场景。
- 后续如果需要换模型，只改配置，不用改主链路。

### 3. 真正做出 dense + sparse hybrid retrieval

我重写了 `agentic_rag/retrieval/hybrid_search.py`。

新的实现不是“单向量搜一遍再重排”，而是：

- dense 检索：`VectorStore.search()`
- sparse 检索：`VectorStore.keyword_search()`
- 融合：RRF
- 精排：Cross-Encoder rerank

同时我保留了 fallback：如果重排模型不可用，就先退化到融合结果返回，而不是整条链路直接失败。

### 4. 把 Planner/Executor 改成结构化工作流

我重写了 `agentic_rag/graph/nodes.py`，核心是把 Planner 输出从自由文本改成结构化 `step_type + description`。

支持的步骤类型是：

- `RETRIEVE`
- `WEB_SEARCH`
- `SQL_QUERY`
- `CALCULATE`
- `SYNTHESIZE`

这样 Executor 不再依赖字符串模糊匹配，而是稳定执行结构化步骤。这个改动会明显提升工作流的可预测性，也更符合现在工业里常见的 agent orchestration 写法。

### 5. 统一 ingestion pipeline

原始项目里 UI 上传、MCP 入库、文件工具入库是三套分散逻辑。我这次把它们统一到 `ingestion.py`。

这带来两个价值：

- 所有入口都走同一套“解析 -> 切块 -> 写入向量库”流程，行为一致。
- 后续如果要增加 metadata enrichment、embedding cache、异步批处理，只改一处。

### 6. 修正 UI 和工程入口

我重写了 Streamlit 主 UI，修掉了几个直接影响工程质量的问题：

- 去掉原先的 `sys.path` hack。
- 上传文档后不再整篇入库，而是走统一切块入库。
- 抽出通用 `resume_agent()`，不再复制粘贴两份逻辑。
- 知识库统计改成显示 chunk 数，更符合现在的真实实现。

同时我补了：

- `README.md`
- `Dockerfile`
- `main.py` CLI 入口
- `config.yaml`
- 基础测试

### 7. 做基础测试与懒加载

这个项目原来几乎没有测试。我补了两类轻量测试：

- `test_tools.py`：验证切块逻辑。
- `test_graph.py`：验证工作流可编译。

同时把一部分重依赖改成懒加载，比如 reranker 和 provider SDK。这样项目在没有完整下载所有模型依赖时，至少能完成基础模块导入和单元测试，不会在 import 阶段直接挂掉。

## 这次改造后可以怎么讲

一句话总结：

> 我把一个偏 demo 的 Agentic RAG 原型，重构成了更接近企业知识助手的工作流系统，重点补强了 chunk-based ingestion、中文 hybrid retrieval、结构化 planner/executor 和工程完整性。

## 技术亮点

### 1. Chunk-based ingestion

这是最重要的升级点之一。RAG 系统里整篇入库通常效果很差，因为召回粒度过粗。切块后，系统真正检索的是“证据单元”，不是“整份文档”。

### 2. Dense + Sparse + RRF + Rerank

这个组合是非常适合面试表达的，因为它体现你对检索系统不是只停留在“向量搜一下”。

- dense 负责语义召回
- sparse 负责关键词精确匹配
- RRF 负责融合异构候选
- reranker 负责细粒度排序

### 3. Planner 结构化输出

这体现的是 Agent Workflow 的稳定性意识。相比让模型随便吐自然语言计划，再靠字符串判断，结构化 step_type 更可控，也便于后续做可视化和观测。

### 4. 统一入口，避免多套逻辑漂移

UI、MCP、工具层统一走 ingestion pipeline，这个改造虽然不花哨，但非常有工程味道，也很适合面试里讲“可维护性”。

## 高频 Q&A

### Q1：你在这个项目里最核心的改造是什么？

答：最核心的是三件事。第一，把整篇文档入库改成切块入库；第二，把单一路线检索升级成 dense + sparse + RRF + rerank 的混合检索；第三，把 Planner/Executor 改成结构化 workflow，让工作流从 demo 风格变成更稳定的工程实现。

### Q2：为什么切块比整篇入库更重要？

答：因为 RAG 系统最终需要给生成模型提供的是“相关证据”，不是“整个文档”。整篇入库会导致召回粒度太粗，相关片段被大段无关内容稀释。切块后，召回单元更精细，reranker 和生成阶段都更容易聚焦真正有用的信息。

### Q3：为什么要用 `bge-m3`？

答：原项目用的是偏英文的 MiniLM embedding，不太适合中文知识库。`bge-m3` 更适合中英文混合语料，而且对检索场景更友好。这个升级直接影响召回质量，也是一个很容易被面试官认可的工业化改造点。

### Q4：你说 hybrid retrieval，具体 hybrid 在哪？

答：这里的 hybrid 不是“向量搜一下再重排”，而是两种召回信号并行：dense 语义召回和 sparse 关键词召回。两路候选先通过 RRF 融合，再交给 Cross-Encoder 精排。这样既保留了语义泛化能力，也保留了关键词命中的精确性。

### Q5：为什么用 RRF？

答：因为 dense 和 sparse 的分数分布本来就不一样，直接线性加权容易不稳定。RRF 只依赖排名，不强依赖原始分值，对异构候选融合更稳，也更容易落地。

### Q6：为什么要把 Planner 输出结构化？

答：因为原始版本 Planner 输出是自然语言，Executor 靠字符串包含关系判断要走哪个工具，这种方式太脆弱。结构化成 `step_type + description` 后，执行器逻辑更稳定，后续也更容易扩展到 tracing、可视化和策略分析。

### Q7：这个项目里 LangGraph 的作用是什么？

答：LangGraph 的价值在于把整个工作流显式化。Router、Planner、Executor、Critic、Generator 这些节点以及它们之间的条件分支，都可以被明确建模。相比单个 Agent 黑盒循环，这种图式 workflow 更适合复杂任务和后续维护。

### Q8：Human-in-the-loop 放在哪一层？

答：放在 Critic 之后。如果 Critic 判断答案质量还不够，就先在 `human_approval` 节点暂停，让用户决定是否继续反思迭代。这种设计适合企业知识助手场景，因为有些请求追求速度，有些请求更重视准确性。

### Q9：这个项目还有哪些没完全做完的地方？

答：我会诚实讲三点。第一，多模态检索模块虽然有独立实现，但当前主链路还是以文本/表格为主；第二，图片理解里的 VLM 还是占位，后续可接 Qwen2.5-VL 或 InternVL；第三，SQL 工具仍是 demo 数据库，不适合当作生产能力去宣传。

### Q10：如果再给你一轮迭代，你下一步会做什么？

答：我会做三件事。第一，补真正的 BM25 或 Elasticsearch 稀疏检索，而不是当前的轻量关键词重打分；第二，把多模态检索真正接进主工作流；第三，引入 retrieval evaluation，比如 RAGAS/DeepEval 或自定义命中率评测，让“检索质量提升”从主观判断变成可测指标。

### Q11：为什么你说这次改造更接近工业技术？

答：因为这次改造用了现在工业里比较常见的几类思路：chunk-based ingestion、hybrid retrieval、rank fusion、Cross-Encoder rerank、结构化 planner/executor、可配置模型与懒加载依赖。这些都不是“花哨 AI demo 功能”，而是更接近实际知识系统的基础设施改造。

### Q12：你在这个项目里最想体现的能力是什么？

答：不是“我会接 LangGraph”，而是“我能看出一个原型系统哪里会影响真实效果和工程质量，然后做有优先级的重构”。这次项目里，我优先改的是检索粒度、检索路线、工作流稳定性和工程完整性，而不是先堆 UI 花样。

## 面试里诚实口径

- 可以说：做了二次工程化重构，显著补强了检索链路和工作流稳定性。
- 可以说：更接近企业知识助手，但仍是内部项目/原型级系统，不要夸成大规模生产平台。
- 可以说：当前主链路以文本知识检索为主，多模态模块仍在扩展中。
- 不要说：已经做完完整企业级多模态生产系统。

## 最稳收尾

> 这个项目我最想表达的，不是我做了一个会聊天的 RAG 页面，而是我把一个原型系统里真正影响效果和工程可信度的部分重新梳理了一遍：把整篇入库改成切块，把单一路线检索改成 hybrid retrieval，把字符串驱动的工作流改成结构化执行，并补上工程入口和测试。这个过程更能体现我做系统重构和技术取舍的能力。
