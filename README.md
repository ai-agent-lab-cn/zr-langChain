# zrLangChain

基于 Python、LangChain、Gemini 和 LangSmith 的 AI Agent 学习与实战项目。

## 技术栈

- Python 3.11
- uv
- LangChain
- Google Gemini
- LangSmith
- LangGraph

## 实战路线

| 章节 | 主题 | 状态 |
|---|---|---|
| 01 | [基础 AI Agent 入门实战](./01-basic-agent/) | ✅ 已完成 |
| 02 | [Multi-Tool Agent：多工具调用](./02-multi-tool-agent/) | ✅ 已完成 |
| 03 | [Structured Output：结构化输出](./03-structured-output/) | ✅ 已完成 |
| 04 | [Agent Memory：记忆机制](./04-agent-memory/) | ✅ 已完成 |
| 05 | MCP：工具集成 | 规划中 |
| 06 | RAG Agent：检索增强生成 | 规划中 |
| 07 | LangGraph：工作流编排 | 规划中 |

## Quickstart

安装依赖：

```bash
uv sync
```

在项目根目录配置 `.env`，然后运行第一章：

```bash
uv run python 01-basic-agent/main.py
```

## 项目规范

每个章节包含：

- `main.py`：独立可运行的示例代码
- `README.md`：完整技术文档

所有章节共享根目录的 Python 环境与依赖配置。
