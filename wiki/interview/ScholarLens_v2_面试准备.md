# ScholarLens v2 面试准备

## 使用原则

- 简历上写短一点，只写你能在面试里展开并自证的点。
- 面试里重点讲“为什么这么设计”“和 v1 比升级了什么”“上线后怎么保证稳定性”。
- 如果你没有真实的 DAU、调用量、覆盖人数数据，不建议直接写“全校都在用”。更稳的写法是“已完成校内部署/面向校内科研场景上线”。

---

## 一、简历项目描述

### 版本 A：适合简历一段话

【ScholarLens v2：企业级多智能体学术检索系统】

基于 FastAPI + 多智能体编排重构科研检索系统，围绕复杂学术问题实现 QueryAgent 查询规划、ArXiv/Semantic Scholar/OpenAlex 多源并行检索、RRF 融合、Cross-Encoder 重排、JudgeAgent 质量评估与 RefineAgent 自适应改写，支持结构化文献综述生成与容器化部署，完成从原型式 ReAct 检索到服务化后端的升级。

### 版本 B：适合简历两段式

【项目简介】面向科研检索场景，独立设计并实现企业级学术检索后端。系统从早期的 ReAct + 工具调用原型升级为多智能体服务化架构，通过查询规划、多源并行检索、结果融合、质量评估和自动改写闭环，提高复杂问题下的检索覆盖率与结果稳定性。

【核心工作】负责 QueryAgent / JudgeAgent / RefineAgent / SynthesisAgent 编排逻辑设计，接入 ArXiv、Semantic Scholar、OpenAlex 三个学术数据源，使用 RRF 做多源融合，使用 Cross-Encoder 做精排，并通过 FastAPI 暴露 search 与 quick search 接口；同时补充配置中心、缓存层、容器化部署与检索评测脚手架，支撑面向校内场景的上线交付。

### 版本 C：更偏“上线表达”

【ScholarLens v2：校内科研检索服务】

负责校内科研检索服务后端架构升级，将原有单轮检索原型重构为可部署的多智能体学术检索系统。后端基于 FastAPI 提供统一检索 API，内部采用 Query Planning + 多源并行召回 + RRF 融合 + Judge 驱动迭代改写的流水线，支持结构化综述生成、模型按 Agent 差异化配置、Docker 部署与缓存加速。

### 建议你在简历里保留的 4 个点

1. 从 ReAct 原型升级到多智能体服务化后端。
2. 三个真实数据源并行检索：ArXiv、Semantic Scholar、OpenAlex。
3. QueryAgent + JudgeAgent + RefineAgent 的闭环改写机制。
4. FastAPI + Docker + 缓存层的工程化交付。

---

## 二、面试开场怎么讲

### 30 秒版本

这个项目本质上是一个面向科研场景的智能检索后端。v1 更像 MCP/ReAct 风格的工具调用原型，能做多源查询和初步重排；v2 我重点把它重构成了服务化的多智能体检索系统，把查询规划、并行检索、结果评估、改写重试和综述生成拆成独立 Agent，通过 FastAPI 暴露统一接口，支持 Docker 部署，更接近真实上线系统。

### 2 分钟版本

这个项目解决的是传统学术检索里两个很典型的问题：第一，用户问题通常是自然语言，不是标准关键词，直接搜很容易漏；第二，单一数据源的结果不稳定，复杂问题很难一次搜全。

我在 v2 做的核心升级有三件事。第一，把原来偏单轮的 ReAct 检索流程拆成多智能体流水线：QueryAgent 先做两阶段推理和查询规划，再把子查询分发到 ArXiv、Semantic Scholar、OpenAlex 并行检索，返回结果之后用 RRF 做融合，再交给 JudgeAgent 判断结果质量，不够好就由 RefineAgent 改写查询进入下一轮。第二，把系统改造成标准 API 服务，基于 FastAPI 提供 full search 和 quick search 两种接口，便于前端或其他系统接入。第三，补齐工程化能力，包括统一配置、容器化部署、缓存、评测指标和模型工厂，让它不仅能跑 demo，也能作为校内科研检索服务去交付。

如果面试官继续追问亮点，我会重点讲 QueryAgent 的两阶段 CoT 查询规划、RRF 为什么适合多源异构结果融合，以及 JudgeAgent 驱动的自适应终止和查询改写。

---

## 三、系统架构主线

### 1. 总体架构

前端/调用方
→ FastAPI API 层
→ Orchestrator 编排层
→ QueryAgent 查询规划
→ RetrievalEngine 多源并行检索
→ RRF 融合
→ Cross-Encoder 重排
→ JudgeAgent 质量评估
→ RefineAgent 迭代改写
→ SynthesisAgent 生成结构化综述

### 2. 当前代码里的核心组件

- API 层：FastAPI，对外提供 /api/v1/search、/api/v1/search/quick、/health。
- 编排层：Orchestrator，负责多轮搜索状态推进和终止策略。
- 检索层：并行调用 ArXiv、Semantic Scholar、OpenAlex。
- 融合层：RRF，解决不同来源分数不可直接比较的问题。
- 重排层：Cross-Encoder 精排候选结果。
- 评估层：JudgeAgent 对 Top 结果做相关性与覆盖度判断。
- 改写层：RefineAgent 结合原始推理链和评估结果生成下一轮查询。
- 生成层：SynthesisAgent 输出中文结构化文献综述。
- 工程层：配置中心、模型工厂、缓存层、Docker 部署。

### 3. 真实部署口径

如果面试官问“部署在哪、怎么上线的”，你可以这样答：

当前交付形态是 Docker 化部署，核心服务是 FastAPI 检索 API，旁路是 Redis 缓存。应用容器通过 uvicorn 启动，前端可以是 React 页面，也可以是 Streamlit 形式的校内检索入口。对外统一暴露 HTTP API，对内由 Orchestrator 调度多 Agent 和检索链路。

这句话能讲得稳，因为和现有代码是对得上的。

---

## 四、高频 Q&A

### Q1：这个项目到底解决了什么问题？

答：

它解决的是“复杂科研问题很难一次检索到高质量论文”的问题。用户输入的往往是自然语言问题，比如“RAG 在知识密集型任务里的应用”，这类问题直接做关键词匹配会有两个痛点：一是关键词不标准，二是单一数据源结果不稳定。我的做法是把查询理解、并行召回、结果评估和自动改写串成闭环，让系统不是只搜一次，而是能根据当前结果质量决定要不要继续搜、怎么改写再搜。

你可以把它概括成一句话：它不是一个“帮你搜论文的工具”，而是一个“把复杂科研问题翻译成检索策略并自动迭代优化”的后端系统。

---

### Q2：v2 和你之前的 v1 最大的区别是什么？

答：

v1 更偏原型验证，核心是 ReAct 风格的单轮或少量循环工具调用，重点在证明“LLM 可以驱动检索”；v2 则是面向上线交付的服务化重构，重点变成三个方面：

1. 架构上，从单 Agent 串行流程升级为多 Agent 编排。
2. 检索上，从简单多源调用升级为多源并行 + RRF 融合 + 评估驱动迭代。
3. 工程上，从脚本式原型升级为 FastAPI + 配置中心 + 容器化部署的后端服务。

如果要再压缩成一句话，就是：v1 证明思路可行，v2 把它做成了可对外提供能力的系统。

---

### Q3：为什么你不用传统 ReAct 继续做，而要拆成多个 Agent？

答：

ReAct 的问题是“所有事情都让一个 Agent 决定”，这在 demo 阶段可以，但到复杂检索场景就会暴露几个问题：

1. 查询理解、结果评估、查询改写是三种不同认知任务，混在一个 prompt 里很容易互相干扰。
2. 单 Agent 更像串行思考，难以把多源检索的并行能力发挥出来。
3. 如果要做终止判断、结果收敛判断和质量打分，单个 ReAct 循环会越来越重，维护成本也高。

所以我在 v2 做了职责拆分：

- QueryAgent 负责“先想清楚怎么搜”。
- JudgeAgent 负责“判断搜得够不够好”。
- RefineAgent 负责“如果不够好，下一轮怎么改”。
- SynthesisAgent 负责“把最终结果组织成用户可读的综述”。

这样拆的好处是每个 Agent 的 prompt 更稳定，模型参数可以独立配置，整条链路也更容易调试和扩展。

---

### Q4：完整检索链路是怎样的？

答：

完整链路可以分成 7 步：

1. 用户请求进入 FastAPI 的 /api/v1/search。
2. Orchestrator 创建 SearchState，并调用 QueryAgent 生成 3 到 5 条子查询。
3. RetrievalEngine 对每个子查询并行访问 ArXiv、Semantic Scholar、OpenAlex。
4. 不同来源的结果先去重，再用 RRF 融合成统一候选集。
5. 候选结果进入 Cross-Encoder 做精排，得到当前轮排名。
6. JudgeAgent 对当前结果做相关性、覆盖度和缺失主题判断。
7. 如果结果不满足阈值，就调用 RefineAgent 改写查询进入下一轮；如果已经满足，就交给 SynthesisAgent 生成结构化综述。

这里最关键的一点是：系统不是“搜完就结束”，而是“评估后决定下一步动作”。这就是它和普通搜索接口最大的区别。

---

### Q5：QueryAgent 的两阶段 CoT 是怎么设计的？

答：

我没有让模型直接从原始问题生成检索词，而是拆成了两个阶段。
 
第一阶段是 Reasoning，也就是先做查询理解。模型会先分析用户真正想找什么，是综述、方法比较还是具体技术；再把问题拆成概念图谱，识别核心概念、同义词、缩写和可能的歧义；最后给出检索策略，比如应该偏精准术语、语义扩展还是种子论文锚定。

第二阶段是 Planning，也就是基于前面的推理结果真正生成检索子查询。这样生成出来的查询不是随意扩词，而是带有明确分工的，例如有的负责精准命中，有的负责扩大召回，有的负责锚定经典论文。

我觉得这个设计的价值在于：系统保留了原始的推理链，后面的 RefineAgent 不需要“重新猜用户想要什么”，而是直接站在第一次推理的基础上做针对性修正。

---
### Q5 深挖：CoT 的 prompt 到底怎么写的？为什么这么设计？

答：

这是整个项目最核心的设计环节，也是我花时间最多的地方。我按"为什么要用 CoT → prompt 结构设计 → 输出约束 → 推理链传递 → 迭代改写如何复用 → 踩过的坑"这条线来讲。

#### 1. 为什么不直接让 LLM 生成搜索关键词？

最朴素的做法是一次 LLM 调用，输入用户问题，输出搜索词列表。我一开始也是这么做的，但观察到三个问题：

1. 生成的查询高度同质化。比如用户问"RAG 在知识密集型任务中的应用"，模型会生成三条几乎一样的关键词变体，召回结果高度重叠。
2. 模型没有分析搜索难度的意识。有些问题需要精确术语匹配，有些需要语义扩展，有些需要锚定经典论文；但一次调用不会区分。
3. 后续改写没有上下文可复用。如果第一轮结果不满足，RefineAgent 不知道"之前是怎么理解这个问题的"，只能重新猜。

所以我把它拆成两次 LLM 调用，第一次专门做推理分析，第二次基于分析结果生成查询。

#### 2. Stage 1 Reasoning Prompt 的设计思路

Stage 1 的 prompt 核心设计原则是**结构化分步推理**，不是让模型自由发挥，而是用四步强制引导模型从不同角度分析问题：

```
Step 1 - Intent Analysis（意图分析）
Step 2 - Concept Decomposition（概念分解）
Step 3 - Search Difficulty Analysis（搜索难度分析）
Step 4 - Strategy Formulation（策略制定）
```

为什么是这四步？因为它们分别对应检索规划中四个不同层面的认知任务：

- Step 1 解决"用户到底想找什么"。我让模型判断研究类型是 survey / method_comparison / specific_technique / application / theoretical_foundation 五类之一。这一步的价值是确定搜索的宏观方向。

- Step 2 解决"问题里有哪些概念、它们是什么关系"。这是整个推理链最关键的一步。我设计了一个 ConceptNode 结构，每个概念必须标注三个属性：category（核心方法 / 应用领域 / 评估指标 / 基线 / 数据集 / 理论）、importance（essential / important / peripheral）、synonyms（同义词和缩写）。比如"RAG"会被标注为 core_method、essential，同义词是 "Retrieval-Augmented Generation"。这样后面生成查询时能精准知道哪些词是必须出现的，哪些可以替换。

- Step 3 解决"这个问题为什么难搜"。我让模型从五个角度分析：术语歧义、跨学科性、时效性要求、关键词过载、长尾还是主流。这一步决定了后面应该用什么类型的查询组合。比如如果模型判断出"关键词过载"，后面就需要 negation_boundary 类型的查询来排除噪声。

- Step 4 解决"应该用什么搜索策略"。这是推理的汇总层，模型基于前三步给出具体策略建议和理想结果画像。

整段 prompt 最后通过 JsonOutputParser 要求模型输出一个结构化的 QueryReasoning 对象，而不是自由文本。这保证了输出可程序化消费。

#### 3. Stage 1 的输出结构（代码里的 Pydantic 模型）

```
QueryReasoning:
  user_intent: str          ← 一段话描述用户真正想找什么
  research_type: str         ← survey | method_comparison | specific_technique | application | theoretical_foundation
  concepts: [ConceptNode]    ← 概念图谱
    - name: str              ← 概念名称
    - category: str          ← core_method | application_domain | evaluation_metric | baseline | dataset | theory
    - importance: str        ← essential | important | peripheral
    - synonyms: [str]        ← 同义词列表
  concept_relationships: [str] ← 概念间关系描述
  difficulty_factors: [str]  ← 搜索难度因素
  search_strategy: str       ← 总体搜索策略
  expected_result_profile: str ← 理想结果画像
```

这个结构的设计意图是：不让模型输出自由文本推理，而是强制输出结构化数据，这样后面的 Stage 2 和 RefineAgent 可以直接读取和对比。

#### 4. Stage 2 Planning Prompt 的设计思路

Stage 2 的输入不是原始用户查询，而是 Stage 1 的推理摘要。prompt 里我做了两件关键的事：

第一，定义了 5 种查询类型，每种类型有明确的适用场景：

| 类型                | 适用场景          | 举例                                                                      |
| ----------------- | ------------- | ----------------------------------------------------------------------- |
| keyword_precise   | 已有明确术语，追求精准匹配 | "BERT fine-tuning NER"                                                  |
| semantic_broad    | 需要语义泛化，扩大召回范围 | "how to adapt pre-trained language models for named entity recognition" |
| method_specific   | 搜索特定算法或方法名    | "LoRA low-rank adaptation"                                              |
| citation_anchor   | 锚定经典论文，追溯引用网络 | "attention is all you need transformer architecture"                    |
| negation_boundary | 排除常见噪声        | "retrieval augmented generation NOT chatbot NOT dialogue"               |

为什么是这 5 种？因为我在实际测试中观察到，学术检索的问题基本可以分成"精准找""广泛找""找方法""找经典""排噪声"这五类需求，如果生成的查询只覆盖其中一两种，召回率和多样性都不够。

第二，要求每个子查询必须附带 rationale（为什么要生成这条查询）和 target_concepts（它瞄准概念图谱中的哪些概念）。这不是摆设，而是后面 RefineAgent 判断"之前哪些概念覆盖不足"的关键依据。

#### 5. Stage 2 的输出结构

```
QueryPlan:
  sub_queries: [SubQuery]    ← 3-5 条类型化子查询
    - query: str             ← 实际搜索字符串
    - query_type: str        ← 5 种类型之一
    - rationale: str         ← 为什么这么生成
    - target_concepts: [str] ← 瞄准哪些概念
  year_start / year_end: int ← 可选时间过滤
  expected_topics: [str]     ← 预期应覆盖的子主题列表
  coverage_strategy: str     ← 子查询之间如何配合覆盖
```

#### 6. 推理链如何传递给 RefineAgent

这是整个 CoT 设计里我觉得最有意思的一环。Stage 1 的推理输出不是用完就丢，而是用一个 _summarize_reasoning() 方法格式化成可读摘要，保存在 SearchState.cot_reasoning 里。格式大概是这样：

```
Intent: 用户寻求 RAG 在知识密集型任务中的综合理解...
Research Type: specific_technique

Key Concepts:
  - [essential] RAG (core_method) (aka: Retrieval-Augmented Generation)
  - [essential] knowledge-intensive tasks (application_domain)
  - [important] evaluation metrics (evaluation_metric)

Concept Relationships:
  - RAG extends traditional language models with retrieval capabilities
  - Knowledge-intensive tasks benefit from the retrieval component

Search Difficulty: ambiguous terminology, cross-disciplinary, rapidly evolving
Strategy: 先用精确关键词查核心论文，再用语义查询扩展...
Expected Results: 应包含 RAG 奠基论文、最新进展、对比研究...
```

当 JudgeAgent 判定结果不满足时，RefineAgent 拿到的输入包括三样东西：

1. 上面这段推理链（state.cot_reasoning）
2. 之前尝试过的查询列表，包括每条查询的类型和理由（state.query_plan_detail）
3. JudgeAgent 给出的缺失主题和改进建议（evaluation.missing_topics + evaluation.suggestions）

RefineAgent 的 prompt 也是强制分三步推理：

- Step 1: Root Cause Analysis — 为什么之前的查询没覆盖缺失主题？是太宽、太窄、还是用了错误社区的术语？
- Step 2: Concept Gap Analysis — 对照原始推理链的概念图谱，哪些概念在结果里覆盖不足？
- Step 3: Generate Refined Queries — 生成 2-3 条改写查询，每条必须说明 strategy_shift（策略转变了什么）。

这样做的好处是 RefineAgent 不是盲目重写，而是站在第一次推理的基础上做增量修正。举个具体例子：如果第一轮的 keyword_precise 和 semantic_broad 都搜了，但"评估指标"这个子主题在结果里不够，RefineAgent 会从概念图谱里看到 evaluation_metric 被标注为 important 但结果里缺失，然后针对性地生成一条 "evaluation benchmarks for retrieval augmented generation" 这种聚焦评估领域的查询。

#### 7. 为什么输出用 Pydantic + JsonOutputParser 而不是自由文本？

因为 CoT 推理的输出不是给人看的，而是要被程序消费的。如果让模型输出自由文本，后面的 Stage 2 和 RefineAgent 还得做额外的文本解析，很容易出错。用 Pydantic 模型 + JsonOutputParser 有三个好处：

1. 模型输出直接序列化成 Python 对象，不需要正则提取。
2. 如果模型输出不符合 schema，LangChain 的 parser 会直接报错，方便做 fallback。
3. 概念图谱的 category、importance 这些字段是枚举值，可以在后续逻辑里做条件判断。

#### 8. Fallback 降级机制

CoT 推理依赖 LLM 能输出符合 schema 的 JSON。如果 LLM 输出格式出错或者 API 调用超时，我做了两层降级：

Stage 1 失败时：直接把用户原始查询包装成一个最小化的 QueryReasoning 对象，research_type 默认为 survey，concepts 只有一个节点就是原始问题本身。这样即使推理失败，系统也能继续跑。

Stage 2 失败时：直接用用户原始查询作为唯一子查询，类型是 keyword_precise，rationale 标记为 fallback。

这两层降级保证了 CoT 不会成为系统的单点故障。

#### 9. 踩过的坑和迭代过程

**坑 1：推理和规划一步完成导致输出不稳定。** 最早我试过把推理和查询生成写在同一个 prompt 里，让模型一次输出 reasoning + queries。问题是 JSON 结构太复杂（嵌套了概念图谱 + 查询列表），模型经常格式出错。拆成两次调用后，每次 prompt 的输出结构更简单，成功率明显提高。

**坑 2：概念分类不固定导致 RefineAgent 无法对比。** 我一开始没有限制概念的 category 字段，让模型自由写。结果发现同一个概念在不同查询中被分到不同的类别里，RefineAgent 做差距分析时无法可靠对比。后来我改成枚举 6 个固定值（core_method / application_domain / evaluation_metric / baseline / dataset / theory），问题解决。

**坑 3：查询类型没有约束时生成的查询太类似。** 没有定义类型化和强制约束之前，模型倾向于生成 3 条几乎一样的关键词变体。引入 5 种类型之后，加上 prompt 里明确要求"avoid redundant queries that would return the same papers"，生成的查询多样性明显提升。

**坑 4：推理链太长会影响 Stage 2 的输出质量。** Stage 1 输出的 QueryReasoning 对象可能很详细，如果完整序列化一大段塞进 Stage 2 的 prompt 里，模型有时会被长上下文干扰。所以我做了一个 _summarize_reasoning() 方法，把结构化对象转成精简的文本摘要再传给后续阶段。

#### 10. 面试快速总结版

如果面试官问"你的 CoT prompt 怎么写的"，30 秒版本是：

> 我把查询理解拆成两次 LLM 调用。第一次做推理分析，用四步结构化引导：意图分析 → 概念分解（标注类别、重要性、同义词）→ 搜索难度分析 → 策略制定，输出是一个 Pydantic 模型通过 JsonOutputParser 序列化。第二次基于推理摘要生成 3-5 条类型化查询，分为 keyword_precise / semantic_broad / method_specific / citation_anchor / negation_boundary 五种，每条附带 rationale 和 target_concepts。推理链保存在 state 里供 RefineAgent 复用，形成闭环。

如果追问"为什么不一步做完"：

> 因为一步做完时 JSON 输出结构太深，模型格式错误率高；而且推理和规划的认知负载不同，分开后每步 prompt 更简单、输出更稳定。

如果追问"RefineAgent 怎么用这个推理链"：

> RefineAgent 收到的输入包括之前的推理链、已尝试的查询列表和 JudgeAgent 的缺失主题反馈。它的 prompt 也是三步结构化推理：根因分析 → 概念差距分析 → 生成带 strategy_shift 的改写查询。本质上是在第一次推理的基础上做增量修正，而不是从头猜。

---
### Q6：为什么要多源并行检索，而不是只接一个学术搜索源？

答：

因为单一来源的结果分布很不稳定。ArXiv 更偏预印本和前沿方向，Semantic Scholar 的元数据更丰富，OpenAlex 在论文元数据和领域标签上有补充优势。复杂问题下，只依赖一个源会出现两类风险：要么漏召回，要么结果过度偏向某一类论文。

多源并行的价值有两个：

1. 扩大召回范围，减少单一来源带来的盲区。
2. 让后面的融合和重排有更好的候选池。

你可以把这件事理解成“先尽量多拿到候选，再靠融合和重排提高精度”，而不是一开始就押宝某一个搜索源。

---

### Q7：为什么你用 RRF 做融合？

答：

RRF 适合多源学术检索的原因很直接：不同来源的原始分数不可比。比如一个源给的是内部相关度分，另一个源给的是接口返回顺序，还有一个源甚至没有稳定分值。如果强行做线性加权，往往要先做复杂归一化，而且不同数据源的分布会漂。

RRF 只看排名，不强依赖原始分数，公式是：

score(d) = Σ w / (k + rank)

这样做的好处是某篇论文只要在多个来源里都排得靠前，就更容易被顶上来。它对于异构结果融合非常稳，也比较适合工程落地。

面试里一句话总结就是：我选 RRF，不是因为它最花哨，而是因为它对多源异构结果最稳。

---

### Q8：Reranker 为什么不能解决所有问题？

答：

因为 Reranker 只能“排已有候选”，不能“召回缺失文档”。

这是检索系统里一个非常核心的认识。如果第一阶段就没把论文召回进来，后面重排模型再强也没有机会把它排到前面。所以检索系统里真正决定上限的通常是“查询质量 + 候选池质量”，不是单独一个 Reranker。

我的系统里 Cross-Encoder 的角色是把已经拿到的多源候选做更细粒度的语义排序，但它不是整个系统的唯一亮点。真正重要的是前面 QueryAgent 的查询规划和多源并行召回。

---

### Q9：当前重排是怎么做的？

答：

当前代码里实际落地的是 Cross-Encoder 精排。做法是把 query 和 paper 的 title + abstract 组成 pair，送进 Cross-Encoder 输出相关度分数，再按分数排序。

这里我会主动强调一句：我没有在面试里把它说成“很复杂的三阶段工业排序链路”，因为当前版本真实落地的是 RRF 之后接 Cross-Encoder，这样讲是最稳的。如果面试官问后续优化方向，我会说可以继续往两阶段甚至三阶段排序演进，比如先用更轻量的模型筛一遍，再用更重的 Cross-Encoder 做精排。

---

### Q10：JudgeAgent 是怎么判断结果够不够好的？

答：

JudgeAgent 不是只看“有没有几篇看起来相关”，而是从结果集合角度做判断。当前设计里会看三类信息：

1. 平均相关性够不够高。
2. 主题覆盖度够不够完整。
3. 缺失主题是不是已经很少。

如果相关性评分达到阈值、覆盖度达标，而且缺失主题不多，就可以停止；否则就说明当前结果集还不完整，需要 RefineAgent 改写查询继续找。

这个机制的价值在于：终止不再靠拍脑袋设定固定轮数，而是靠“当前结果到底有没有满足用户问题”来决定。

---

### Q11：终止策略具体怎么设计？

答：

我用了三层终止逻辑，再加一个兜底上限：

1. 最大迭代次数，避免系统在极端情况下无限循环。
2. 边际收益，如果新一轮带来的有效新结果很少，就说明继续搜收益不高。
3. 结果收敛，如果当前 Top 结果和上一轮高度重合，说明已经接近稳定。
4. 质量满足，如果 JudgeAgent 给出的相关性和覆盖度已经达到阈值，就直接结束。

这套策略比“固定 3 轮”更合理，因为它把工程成本和用户体验都考虑进来了。

---

### Q12：RefineAgent 和普通 query rewrite 有什么区别？

答：

普通 query rewrite 往往只是换几个词，或者做同义词扩展；RefineAgent 则是结合两类上下文来改：

1. QueryAgent 第一轮保留下来的推理链。
2. JudgeAgent 当前轮给出的缺失主题和优化建议。

所以它不是盲目重写，而是带着明确诊断去改，比如“之前查询太宽了”“用了错误社区的话语体系”“缺了某个子主题的关键词”。

这件事对面试官来说是个很好的亮点，因为它体现的是闭环优化，不是静态模板化搜索。

---

### Q13：MCP 是什么？

答：

MCP，全称是 Model Context Protocol，本质上是一个让模型、工具和上下文以统一协议交互的标准。你可以把它理解成“给模型调用外部能力制定统一接口规范”。

它的核心价值不在于某个特定框架，而在于三件事：

1. 工具描述统一，模型知道有哪些工具、输入参数是什么、输出长什么样。
2. 上下文传递统一，工具结果怎么回给模型是标准化的。
3. 便于替换和扩展，新增工具不需要重写整套调用逻辑。

如果面试官要你一句话定义，可以说：MCP 是为大模型接外部工具和上下文设计的标准协议层。

---

### Q14：那你这个项目里 MCP 是怎么落地的？

答：

这个问题建议你分阶段回答。

第一阶段，也就是 v1 原型阶段，我的思路是按照 MCP/工具协议化的方式去封装学术检索能力。比如 ArXiv 和 Google Scholar 都被包装成统一的 search tool，输入是关键词和筛选条件，输出是标准化 paper schema。这样 Agent 不需要关心底层接口细节，只要关心“调用哪个检索工具”。

第二阶段，也就是 v2 企业级版本，我把重点从“模型如何调工具”转向“如何把整个检索链路做成稳定的服务”。所以主链路最终落在 FastAPI 服务里，由 Orchestrator 统一调度 QueryAgent、RetrievalEngine、JudgeAgent 和 RefineAgent。也就是说，v2 的核心不是把整个系统完全托管给 MCP runtime，而是保留工具化抽象思想，把主流程做成更适合上线的服务化编排。

这个回答的好处是既承认了项目和 MCP 的渊源，又不会和当前代码实现冲突。

---

### Q15：如果面试官追问“你具体怎么包装工具的”，怎么答？

答：

你可以这样回答：

我先定义统一的 paper 数据结构，把 title、authors、year、abstract、source、citation_count 这些字段标准化；然后每个数据源各自负责把原始响应转成统一 schema。这样上层编排逻辑永远只面向标准化 paper，不直接依赖某个外部 API 的字段格式。

以这个项目来说，ArXiv、Semantic Scholar、OpenAlex 三个源虽然返回格式不同，但进入检索主链路之后都会映射到同一个 Paper 模型，后续的去重、融合、重排和评估都围绕这个统一对象进行。

如果再深挖，你可以补一句：这就是典型的 adapter 思路，也是工具协议化落地最重要的一层。

---

### Q16：你是怎么做去重的？

答：

去重的关键是定义 canonical id。当前实现优先用 DOI，其次用 arXiv id，再退化到 URL。这样做比只按标题去重要稳，因为学术论文标题可能存在版本差异、大小写差异或副标题差异。

另外，多源结果合并时我不仅要去重，还要保留“元数据更丰富”的版本。比如同一篇论文如果 Semantic Scholar 返回了 citation_count，而另一个源只有标题和摘要，我会优先保留信息更完整的那份。

---

### Q17：缓存是怎么做的？

答：

当前实现里有一层 QueryCache，会把 query 和过滤条件做哈希，命中后直接返回已有结果。配置层支持 Redis，也保留了内存回退逻辑。现在代码里实际有内存 LRU + TTL 兜底实现，所以即使 Redis 不可用，系统也不会完全失去缓存能力。

如果面试官问为什么这么设计，可以回答：因为检索场景里热点问题很多，缓存既能减轻外部 API 压力，也能明显改善重复查询的响应时间。

---

### Q18：多个 Agent 为什么要配不同模型和 temperature？

答：

因为它们承担的认知任务不一样。

- QueryAgent 做的是分析和规划，要求稳定、确定，所以 temperature 要低。
- JudgeAgent 做的是评审和打分，也强调一致性，所以 temperature 也低。
- RefineAgent 做的是策略性改写，需要一点发散能力，所以温度可以稍高。
- SynthesisAgent 做的是文本组织和综述输出，更看重表达流畅性。

当前实现里我专门做了模型工厂，支持按 Agent 独立指定 provider、model 和 temperature。这样后面不管是换 OpenAI 兼容接口，还是接本地 Ollama，改动都很小。

---

### Q18 补充：整个链路里每个环节到底用什么模型？

答：

这个问题一定要先分清楚，项目里其实有三类完全不同的模型，不能混着讲：

1. LLM：负责推理、评估、改写、综述生成，比如 DeepSeek、Qwen。
2. Embedding 模型：负责把 query 和文档编码成向量做召回，比如 e5、bge embedding。
3. Reranker 模型：负责对候选结果做 pairwise 精排，比如 MiniLM Cross-Encoder、bge-reranker。

如果面试官问“你这里到底是 bge 还是 qwen”，最稳的回答是：它们不在同一层，BGE 更偏检索和重排，Qwen/DeepSeek 更偏生成和推理，不能直接互相替代。

当前代码里可以这样讲：

1. QueryAgent、JudgeAgent、RefineAgent、SynthesisAgent 这一层，默认都走 SiliconFlow 上的 DeepSeek-V2.5，只是 temperature 不一样。
2. Dense Retrieval 这一层，配置的是 intfloat/e5-large-v2，用来做向量化召回。
3. Bi-Encoder 预筛层，配置里预留了 BAAI/bge-large-en-v1.5，但当前主实现还没有把这一层完整跑进主链路，所以面试里更适合说“已预留两阶段排序能力”。
4. Cross-Encoder 精排层，当前实际落地模型是 cross-encoder/ms-marco-MiniLM-L-6-v2，这是现在代码里最真实的线上口径。
5. 本地替代方案层，配置支持 Ollama 的 qwen2.5:7b，适合离线开发或低成本本地部署。

你可以把整个模型链路概括成一句话：生成层用 DeepSeek，召回层用 e5，精排层当前用 MiniLM，配置上为 bge 和 Qwen 的替换预留了接口。

---

### Q18 补充：为什么 Query/Judge/Refine/Synthesis 没有全部换成 Qwen？

答：

当前版本我更看重“稳定推理 + 易于服务化”，所以 LLM 主链路优先选了 DeepSeek-V2.5，而不是直接全换成本地 Qwen。原因主要有四个：

1. QueryAgent 要做两阶段推理和结构化 JSON 输出，对推理稳定性要求最高，DeepSeek-V2.5 在这一类任务上更稳。
2. JudgeAgent 要做评分和缺失主题判断，核心诉求是同输入下输出尽量一致，所以我优先选低 temperature 的强推理模型。
3. RefineAgent 虽然需要一点发散能力，但本质还是基于诊断去改写，不是纯创作，因此仍然适合放在同一条强推理模型链路上。
4. SynthesisAgent 其实是最容易切换到 Qwen 的环节，因为它偏表达和组织。如果后面做校内部署、强调中文表达或本地可控，我会优先把综述生成切到 Qwen2.5 或更强的中文模型。

更具体一点，按环节可以这样讲：

1. QueryAgent：适合强推理模型，temperature=0.0，优先 DeepSeek-V2.5、GPT-4o、Claude 这类。
2. JudgeAgent：适合稳定评分模型，temperature=0.0，强调一致性。
3. RefineAgent：适合“推理能力够强且允许适度发散”的模型，temperature=0.5。
4. SynthesisAgent：适合中文表达好、成本更低的模型，Qwen 在这个位置是很好的候选。

所以不是 Qwen 不好，而是它更适合放在“本地化部署、中文生成、成本优化”这些场景里；当前这版为了把检索主链路先做稳，我优先把最难的规划和评估环节放在强推理模型上。

---

### Q18 补充：e5、BGE、Qwen、MiniLM 分别有什么区别？

答：

这个问题可以直接按职责来答：

1. e5-large-v2：典型的 embedding 模型，擅长 query-document 语义匹配，适合做向量召回。它的优势是检索导向比较明确，英文学术检索场景适配度较高。
2. bge-large-en-v1.5：也是 embedding 家族，但在通用语义匹配上表现很好，适合做召回或轻量级预筛。它比生成模型便宜得多，但不能直接负责复杂推理。
3. bge-reranker-v2-m3：这是重排模型，不是 embedding。它做的是“把 query 和文档一起输入，再输出更精细的相关性分数”，效果通常比 embedding 打分更强，但计算成本也更高。
4. cross-encoder/ms-marco-MiniLM-L-6-v2：也是重排模型，优点是轻量、速度快、CPU 也比较能跑，适合当前这个阶段先把服务稳定落地。
5. Qwen2.5：属于生成式大模型，强项是问答、总结、中文表达和一定程度的推理，不适合直接替代 embedding 或 reranker。
6. DeepSeek-V2.5：同样属于生成式大模型，但我当前更看重它在复杂推理和结构化输出上的稳定性，所以被放在 Query/Judge/Refine 主链路上。

一句话区分：

1. e5 / bge embedding 负责“找”。
2. MiniLM / bge-reranker 负责“排”。
3. Qwen / DeepSeek 负责“想”和“写”。

---

### Q18 补充：那为什么当前精排没有直接上 bge-reranker？

答：

因为这是一个典型的效果和时延 trade-off。

如果我直接追求最强精排效果，bge-reranker-v2-m3 这类模型通常会比 MiniLM 更强；但当前系统要先满足“多源检索 + 多轮评估 + API 服务化”的整体稳定性，所以我实际代码里先选了更轻量的 MiniLM Cross-Encoder，原因有三个：

1. 更轻，启动和推理开销更低，更容易先把 API 时延控制住。
2. 作为服务默认模型更稳，CPU 或普通 GPU 环境都更容易部署。
3. 代码里做了 fallback 逻辑，MiniLM 这类模型在工程上更适合作为保守默认值。

如果面试官问“那你后续怎么升级”，你就答：

我会把 Cross-Encoder 从 MiniLM 升级到 bge-reranker-v2-m3，或者做成两档策略，quick search 用轻模型，full search 用重模型。这样既保留低时延入口，也能给复杂问题更高质量的精排。

---

### Q18 补充：如果让我重新设计一版，更好的 RAG 架构是什么？

答：

当前这版更准确地说是“面向学术检索的 Agentic Retrieval + Synthesis”，它已经比普通 RAG 更接近企业检索系统了；但如果继续升级，我认为有三条明显更强的方向。

1. 分层检索的 Hierarchical RAG。
当前系统主要是 paper-level 检索，也就是先找论文，再生成综述。更强的做法是把论文进一步拆成 section 和 chunk：先在 paper level 做粗召回，再在 section level 找最相关章节，最后在 chunk level 给生成模型提供证据。这样回答具体技术问题时，证据粒度会更细。

2. 引文图谱增强的 GraphRAG。
学术场景天然有 citation graph、author graph、venue graph，这些都是普通 RAG 没有充分利用的信号。更强的架构是先用关键词和向量召回种子论文，再沿着“被引、共引、同 venue、同 topic”做图扩展，再交给 reranker 排序。这会比纯文本相似度更适合找经典论文、方法演化链和综述脉络。


3. Late Interaction 检索替代单向量 Dense Retrieval。
如果追求更高召回质量，我会考虑把 e5 单向量召回升级为 ColBERT 这类 late interaction 架构。因为学术检索里专有名词、缩写、数字、方法名很多，单向量压缩容易损失细粒度信息，而 late interaction 对这类术语更友好。

如果只让我选一个“最适合 ScholarLens 下一步”的方案，我会选：

Graph-aware Hierarchical Hybrid RAG。

也就是：

1. QueryAgent 先判断问题类型。
2. Hybrid Retrieval 先召回候选论文。
3. Citation Graph Expansion 做图扩展补召回。
4. Section/Chunk Retrieval 下钻到论文内部证据。
5. Cross-Encoder 或 bge-reranker 做精排。
6. JudgeAgent 做结果置信度判断。
7. SynthesisAgent 基于证据片段输出综述或问答。

这个架构比当前版本更强的原因是，它不只回答“有哪些论文”，还能回答“论文里哪一段真正支撑这个结论”。对学术检索来说，这比普通的摘要式 RAG 更有说服力。

---

### Q18 补充：如果面试官问“你觉得现在这版最大模型短板是什么”，怎么答？

答：

我会直接说三个短板：

1. 召回层的 dense 模型还是单向量路线，面对长尾术语和细粒度方法名时上限有限。
2. 精排层当前默认是 MiniLM，更偏工程稳妥，效果上还有升级空间。
3. 生成层目前虽然支持按 Agent 差异化配置，但还没有做到“按请求类型动态路由不同模型”，后续可以做 query-aware model routing。

这样回答的好处是，你既承认当前方案不是终态，也能顺势引出你的后续优化思路。

---

### Q19：为什么要做 FastAPI API 层，而不是继续用脚本直接跑？

答：

因为一旦你想让前端接入、让多人使用、让系统真正部署起来，脚本式调用就不够了。API 层的价值至少有四个：

1. 对前端暴露稳定接口。
2. 能统一处理参数校验、异常返回和服务健康检查。
3. 更容易接容器化部署和后续监控。
4. 能把“模型链路”变成“产品能力”。

所以 v2 最大的工程化动作之一，就是把原本偏实验性的检索流程变成标准 API 服务。

---

### Q20：系统是怎么部署的？

答：

当前部署方式是 Docker 容器化。主服务容器里运行 FastAPI 应用，启动命令是 uvicorn；旁边有 Redis 容器用于缓存。这样做的好处是环境一致、依赖固定、方便在服务器上直接拉起。

如果面试官继续追问“上线后的后端架构”，你可以这样展开：

对外是一个统一的检索 API 服务，对内由 Orchestrator 把请求拆解成查询规划、检索、融合、重排、评估和综述生成几个阶段；缓存层放在 Redis；模型和检索源通过配置中心统一管理。整个服务是典型的“API 层 + 编排层 + 检索执行层 + 缓存层”的结构。

---

### Q21：如果真的面向全校规模，你会怎么扩？

答：

这个问题不要死磕“我已经做了”，而是讲你的架构思路。

我会从四个方向扩：

1. API 层多副本，前面加网关做限流和鉴权。
2. 把重模型推理和主 API 解耦，避免高峰期阻塞请求线程。
3. 把热门 query 结果缓存下来，减少外部学术源调用。
4. 为检索、重排、外部 API 超时分别做监控和降级策略。

这类回答最能体现你理解“从能跑到能上线”的差别。

---

### Q22：系统里最有技术含量的一点，你会怎么讲？

答：

我会讲 QueryAgent 和 Judge/Refine 组成的闭环，因为它体现的是“系统如何主动优化检索策略”，而不是“模型帮我调用几个 API”。

具体来说，难点不在接几个学术搜索源，而在于如何让系统知道自己搜得够不够好、差在哪里、下一轮该怎么改。这个项目里我把这件事拆成了可解释的状态机和多 Agent 协作，这比单纯说“我接了 LLM”更有技术深度。

---

### Q23：你如何评估这个系统好不好？

答：

我从两类指标看。

第一类是检索质量，包括 NDCG、MAP、MRR、Recall、Precision，这些指标项目里已经有评测模块和 benchmark runner。第二类是工程指标，比如单次请求总时延、迭代轮数、缓存命中情况以及外部源失败时的降级表现。

这里建议你别报自己没跑过的具体数字，但要明确说出“我用什么维度评估，以及为什么这么评估”。

---

### Q24：如果面试官问“这个项目最大的 trade-off 是什么”，怎么答？

答：

最大的 trade-off 是效果和时延之间的平衡。多 Agent、迭代式检索和 LLM 评估能提高复杂问题下的结果质量，但一定会引入更多调用成本和响应时间。所以我在架构上同时保留了 /search 和 /search/quick 两类接口：

- quick search 适合低延迟场景，单轮返回。
- full search 适合高质量场景，允许多轮评估和改写。

这个设计体现的是产品意识：不是所有请求都值得走最重的链路。

---

### Q25：如果面试官质疑“你这个项目是不是就是调 API”，怎么回？

答：

我会把回答重心放在“系统设计”而不是“接了哪些接口”。

外部学术源只是数据入口，真正的难点是：

1. 怎么把自然语言问题转成高质量检索策略。
2. 怎么处理多源异构结果的统一建模、去重和融合。
3. 怎么根据结果质量判断是否继续迭代。
4. 怎么把整个链路包装成稳定可调用的服务。

换句话说，接 API 是最外层，真正的工程价值在编排、评估、优化和服务化落地。

---

## 五、MCP 问题专门答法

### 1. MCP 一句话定义

MCP 是给大模型调用外部工具和上下文设计的标准协议层。

### 2. MCP 在这个项目里怎么讲最稳

项目早期采用 MCP/工具协议化思路封装学术检索能力；在升级到企业级后端后，我把主链路收敛到服务化编排，但底层依然保持统一的工具抽象和标准化数据模型。

### 3. 为什么不把所有东西都直接做成 MCP Server

因为面向真实上线场景时，我更关注的是接口稳定性、编排效率、缓存、异常处理和前后端接入。MCP 适合解决“模型如何统一调用工具”，而企业级检索系统还要解决“服务如何稳定运行”。所以我的做法是保留 MCP 式工具抽象思想，但主链路用 API 服务来承载。

### 4. 如果面试官问“那你到底有没有真的用 MCP”

你可以回答：

有，项目的一期原型就是按照 MCP/工具协议化的思路组织外部检索能力；二期为了上线交付，把重点放在多智能体服务化编排上。对我来说，MCP 不是噱头，真正重要的是用标准化工具接口把模型能力和外部检索能力解耦。

---

## 六、面试时最值得主动抛出的亮点

1. 我不是只做了一个会调接口的 Agent，而是把检索流程做成了可迭代优化的后端系统。
2. 我对检索问题的理解不只停留在“重排模型越强越好”，而是明确区分了召回、融合、重排和评估的职责。
3. 我把 v1 原型真正推进到了 v2 服务化架构，这里面有明显的工程化升级。
4. 我在设计上区分了快路径和全链路路径，兼顾效果和时延。
5. 我对 MCP 的理解不是概念背诵，而是知道它在工具标准化里解决什么问题、又解决不了什么问题。

---

## 七、最后的表达建议

### 面试里推荐你这样收尾

这个项目我最想表达的不是“我做了一个论文搜索工具”，而是“我把一个面向复杂科研问题的智能检索原型，做成了有编排逻辑、有服务接口、有部署方式的后端系统”。如果需要，我可以继续展开讲查询规划、RRF 融合、Judge 驱动迭代，或者讲它如何从 MCP 风格原型演进到服务化架构。

### 你不要主动说死的内容

- 不要主动报没有真实压测支撑的 QPS。
- 不要主动报没有跑出来的 benchmark 数字。
- 不要主动说“全校都在用”除非你能解释用户规模、入口和部署位置。

### 你可以主动说的内容

- 已完成服务化部署与容器化交付。
- 支持校内科研场景接入。
- 已具备面向更大规模扩展的架构基础。
