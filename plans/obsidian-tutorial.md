---
title: Obsidian & Wiki 使用教程
type: guide
created: 2026-04-07
updated: 2026-04-07
status: active
---

# Obsidian & Wiki 使用教程

> 本教程从零开始，教你如何使用 Obsidian 和这个 Wiki 知识库。

---

## 第一步：打开 Vault

1. 打开 Obsidian
2. 点击左下角「打开文件夹作为仓库」（Open folder as vault）
3. 选择 `C:\Users\lenovo\Desktop\my_obsidian`
4. 你会在左侧看到整个目录结构

---

## 第二步：理解 Wiki Links（最核心的概念）

Obsidian 的灵魂是 **Wiki Links**——用双方括号 `[[]]` 链接页面。这和普通 markdown 链接完全不同。

### 基本语法

```markdown
# 1. 最简单的链接——直接写文件名（不需要路径）
[[ppo]]
点击这个链接就会跳转到 wiki/concepts/ppo.md

# 2. 带显示文本的链接——用竖线分隔
[[ppo|PPO 算法详解]]
显示为「PPO 算法详解」，点击跳转到 ppo.md

# 3. 带路径的链接——当文件名可能重复时使用
[[wiki/concepts/ppo]]
[[wiki/papers/schulman-2017-ppo]]

# 4. 链接到页面的某个标题
[[ppo#关键公式]]
跳转到 ppo.md 的「关键公式」标题处

# 5. 带路径 + 显示文本 + 标题
[[wiki/concepts/ppo#关键公式|PPO 的 Clip 公式]]
```

### 实际例子

假设你在写一篇论文笔记，里面提到了 PPO 和 RLHF：

```markdown
本文基于 [[ppo|PPO]] 算法，在 [[rlhf|RLHF]] 框架下进行改进。
与 [[wiki/papers/rafailov-2023-dpo|DPO]] 相比，本方法不需要 reward model。
详见 [[wiki/synthesis/rl-alignment-methods|RL 对齐方法对比]]。
```

在 Obsidian 中渲染后，每个 `[[]]` 都变成一个可点击的紫色链接。

### 链接不存在的页面

```markdown
[[一个还不存在的页面]]
```

这个链接会显示为灰色/虚线。**按住 Ctrl 点击它**，Obsidian 会自动创建这个文件。这是快速建页的方式。

---

## 第三步：理解 Frontmatter（页面元数据）

每个 Wiki 页面顶部都有一段 YAML 格式的元数据，用 `---` 包裹：

```yaml
---
title: PPO
type: concept
tags: [RL, policy-gradient, 对齐]
created: 2026-04-07
updated: 2026-04-07
sources: [raw/papers/schulman-2017-ppo.pdf]
status: draft
---
```

**这段不会在阅读模式下显示**，但有两个重要作用：
1. Obsidian 的搜索和 Dataview 插件可以根据它筛选页面
2. LLM（我）读取页面时用它来理解页面类型和状态

### status 字段含义
- `draft` — 刚创建，内容不完整
- `active` — 内容完善，持续更新中
- `stale` — 可能过时，需要核实

---

## 第四步：日常工作流

### 场景 A：你白天读了一篇论文

1. 把 PDF 放入 `raw/papers/` 目录
2. 告诉我：「帮我 ingest 这篇论文 raw/papers/xxx.pdf」
3. 我会：
   - 在 `wiki/papers/` 创建论文分析页
   - 在 `wiki/concepts/` 创建或更新相关概念页
   - 更新 `index.md`
   - 在 `log.md` 追加记录
4. 你在 Obsidian 中刷新就能看到新页面和链接

### 场景 B：你想快速记录一个想法

直接在 Obsidian 里新建文件，写到 `raw/articles/` 里：

```markdown
# 今天的想法：URLVR 中的 reward shaping

看了 xxx 论文后，觉得可以用 [[grpo]] 的 group relative 思路
来做 unsupervised 的 reward signal。

和 [[wiki/papers/luong-2025-urlvr|Luong 的方法]] 不同的是...
```

之后让我帮你整理到 wiki 中。

### 场景 C：你想复习面试知识

1. 打开 `wiki/interview/llm-and-rl.md`
2. 看到不熟悉的概念，点击 `[[]]` 链接跳转到概念页
3. 概念页里有详细解释、公式、相关论文链接
4. 论文页里有原文摘要和实验结果

**这就是 Wiki 的价值：一切互联，点击即达。**

### 场景 D：周末整理论文

1. 告诉我：「帮我做一次 lint」
2. 我会检查：
   - 哪些页面是孤立的（没有其他页面链接到它）
   - 哪些概念被提到但还没有自己的页面
   - 哪些内容可能过时
   - 建议你补充什么

---

## 第五步：Obsidian 关键操作速查

### 快捷键

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 快速打开文件 | `Ctrl + O` | 输入文件名模糊搜索，最常用 |
| 全局搜索 | `Ctrl + Shift + F` | 搜索所有文件内容 |
| 新建文件 | `Ctrl + N` | 在当前目录新建 |
| 切换编辑/阅读模式 | `Ctrl + E` | 编辑模式看源码，阅读模式看渲染效果 |
| 打开图谱视图 | `Ctrl + G` | 看所有页面的关系图（非常直观） |
| 打开命令面板 | `Ctrl + P` | 所有操作都能在这里找到 |
| 跟随链接 | `Ctrl + 点击链接` | 跳转到链接页面 |
| 返回 | `Ctrl + Alt + ←` | 返回上一个页面 |
| 打开反向链接面板 | 点右侧栏图标 | 查看哪些页面链接到当前页面 |

### 图谱视图（Graph View）

按 `Ctrl + G` 打开。你会看到：
- 每个文件是一个圆点
- 有 `[[]]` 链接的文件之间会有连线
- 圆点大小 = 被链接次数（越多越大）
- 你可以拖动、缩放、筛选

**这是检查知识库健康度的最佳方式。** 孤立的点说明缺少交叉引用。

---

## 第六步：推荐安装的插件

打开 Obsidian → 设置（左下角齿轮）→ 第三方插件 → 关闭安全模式 → 浏览

| 插件 | 用途 | 优先级 |
|------|------|--------|
| **Dataview** | 用 YAML frontmatter 自动生成表格和列表 | 高 |
| **Templater** | 更强大的模板系统，新建页面时自动填充日期等 | 高 |
| **Calendar** | 侧栏日历，方便按日期查看笔记 | 中 |
| **Obsidian Git** | 自动定时 git commit + push | 中 |
| **Tag Wrangler** | 批量管理标签 | 低 |

### Dataview 示例

安装 Dataview 后，在任意页面写：

````markdown
```dataview
TABLE title, status, updated
FROM "wiki/papers"
SORT updated DESC
```
````

它会自动读取 `wiki/papers/` 下所有文件的 frontmatter，生成一个表格：

| title | status | updated |
|-------|--------|---------|
| PPO 论文分析 | draft | 2026-04-07 |
| DPO 论文分析 | draft | 2026-04-07 |

页面越多，这个表格越有价值。

### Obsidian Git 插件配置

安装后在设置中配置：
- Auto backup every X minutes: `30`（每 30 分钟自动 commit + push）
- Auto pull on startup: `true`（打开时自动拉取最新）

这样你不需要手动 git push，知识库自动同步到 GitHub。

---

## 第七步：理解我们的目录约定

```
my_obsidian/
├── raw/                    ← 你放原始资料的地方（不让我改）
│   ├── papers/             ← 论文 PDF 放这里
│   ├── articles/           ← 你自己写的笔记、剪藏的文章
│   └── assets/             ← 图片自动保存到这里
│
├── wiki/                   ← 我维护的区域（你只读，我来写）
│   ├── papers/             ← 每篇论文一个分析页
│   ├── concepts/           ← 每个概念一个页面（PPO、RLHF...）
│   ├── entities/           ← 模型、公司、人物
│   ├── interview/          ← 面试知识点
│   ├── projects/           ← 项目经验
│   └── synthesis/          ← 跨论文对比
│
├── plans/                  ← 学习计划（你和我共同维护）
├── templates/              ← 页面模板
├── index.md                ← 总索引（我维护）
└── log.md                  ← 操作日志（我维护）
```

**核心原则：**
- `raw/` = 原始资料，只有你往里放东西，我不修改
- `wiki/` = 我负责写和更新，你负责阅读和审查
- 你要做的就是：放资料 → 让我处理 → 在 Obsidian 中浏览结果

---

## 第八步：快速上手练习

打开 Obsidian 后，试试以下操作：

### 练习 1：浏览现有页面
1. `Ctrl + O`，输入 `master`，打开学习计划
2. 点击里面的 `[[]]` 链接，感受页面跳转
3. `Ctrl + Alt + ←` 返回

### 练习 2：看图谱
1. `Ctrl + G` 打开图谱视图
2. 看看当前有哪些页面、哪些有连线
3. 随着 wiki 内容增多，图谱会越来越丰富

### 练习 3：看反向链接
1. 打开 `wiki/interview/llm-and-rl.md`
2. 在右侧栏找到「反向链接」（Backlinks）面板
3. 它显示哪些页面链接到了这个页面

### 练习 4：创建一个链接
1. 打开任意 md 文件，切到编辑模式（`Ctrl + E`）
2. 输入 `[[`，Obsidian 会弹出自动补全列表
3. 选一个已有页面，回车
4. 切回阅读模式（`Ctrl + E`），点击链接试试

---

## 常见问题

### Q: 链接显示灰色/虚线是什么意思？
A: 目标页面还不存在。`Ctrl + 点击` 可以自动创建它。

### Q: 我能直接编辑 wiki/ 下的文件吗？
A: 可以，但建议让我来维护（保持格式一致性）。你想改什么告诉我就行。

### Q: 图片怎么插入？
A: 直接把图片拖拽到 Obsidian 编辑器中，它会自动复制到 `raw/assets/` 并插入链接。语法是 `![[图片名.png]]`。

### Q: 怎么搜索特定内容？
A: `Ctrl + Shift + F` 全局搜索。支持正则表达式。

### Q: 标签怎么用？
A: 在文本中任意位置写 `#标签名`（如 `#RL` `#面试`），或在 frontmatter 的 tags 字段中添加。

---

## 和我（LLM）协作的指令参考

| 你说 | 我做什么 |
|------|----------|
| 「帮我 ingest 这篇论文」 | 读论文 → 创建 wiki 页面 → 更新索引和日志 |
| 「xxx 是什么？」 | 查 wiki → 综合回答 → 有价值的回答存入 wiki |
| 「帮我做一次 lint」 | 检查孤立页面、缺失引用、过时内容 |
| 「帮我对比 A 和 B」 | 创建 synthesis 页面，对比分析 |
| 「更新面试题 xxx」 | 在 wiki/interview/ 中添加或完善知识点 |
| 「帮我整理本周论文」 | 把你本周读的论文整理到 wiki/papers/ |

---

*现在去打开 Obsidian 试试吧。下面有两个示例页面可以体验链接效果。*
- [[wiki/concepts/ppo|示例：PPO 概念页]]
- [[wiki/papers/schulman-2017-ppo|示例：PPO 论文页]]
