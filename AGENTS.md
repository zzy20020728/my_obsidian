# LLM Wiki Schema — 秋招备战 & 论文研究知识库

## 身份
- 中科大研二在读，研究方向：大模型无监督强化学习（URLVR）
- 目标：2026年9月秋招进入大厂（大模型算法岗为主，应用开发岗为辅）

## 目录结构

```
my_obsidian/
├── raw/                    # 原始资料（不可修改层）
│   ├── papers/             # 论文 PDF/markdown
│   ├── articles/           # 文章、博客、笔记
│   └── assets/             # 图片等附件
├── wiki/                   # LLM 维护的知识 wiki（编译层）
│   ├── papers/             # 论文摘要与分析页
│   ├── concepts/           # 概念页（RL, RLHF, PPO, GRPO 等）
│   ├── entities/           # 实体页（模型、团队、公司）
│   ├── interview/          # 面试知识点整理
│   ├── projects/           # 项目经验总结
│   └── synthesis/          # 综合分析页（跨论文对比等）
├── plans/                  # 学习计划与进度追踪
├── templates/              # 页面模板
├── index.md                # Wiki 索引（按类别）
├── log.md                  # 操作日志（时间线）
└── AGENTS.md               # 本文件 — Wiki Schema
```

## 页面约定

### 所有 Wiki 页面必须包含 YAML frontmatter：
```yaml
---
title: 页面标题
type: paper | concept | entity | interview | project | synthesis
tags: [相关标签]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [关联的原始资料路径]
status: draft | active | stale
---
```

### 论文页（wiki/papers/）
- 文件名格式：`作者-年份-关键词.md`（如 `luong-2025-urlvr.md`）
- 必须包含：摘要、核心贡献、方法、关键公式、实验结果、与其他工作的关系
- 引用格式：`(Author, Year)` 并注明具体 section

### 概念页（wiki/concepts/）
- 文件名格式：概念名.md（如 `reinforcement-learning.md`）
- 必须包含：定义、关键性质、相关论文链接、面试常问点

### 面试页（wiki/interview/）
- 按主题组织：算法基础、机器学习、深度学习、NLP/LLM、系统设计、Java/编程
- 每个知识点标注面试频率：🔴高频 🟡中频 🟢低频

### 综合分析页（wiki/synthesis/）
- 跨论文比较、方法论对比、领域综述
- 必须引用具体论文页

## 操作工作流

### Ingest（摄入新资料）
1. 将原始文件放入 `raw/` 对应目录
2. LLM 阅读原始资料
3. 在 `wiki/` 中创建或更新相关页面
4. 更新 `index.md`
5. 在 `log.md` 追加记录

### Query（查询）
1. LLM 先读 `index.md` 定位相关页面
2. 读取相关 wiki 页面
3. 综合回答，引用具体页面
4. 有价值的回答归档到 wiki

### Lint（健康检查）
定期执行：
- 检查孤立页面（无入链）
- 检查过时内容
- 检查缺失的交叉引用
- 发现知识空白并建议填补

## 交叉引用
使用 Obsidian wiki links：`[[页面名]]` 或 `[[路径/页面名|显示文本]]`
