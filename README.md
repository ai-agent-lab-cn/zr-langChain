# zrLangChain：LangChain / LangGraph 快速入门

面向 Python 开发者的 **12 章 AI Agent 学习与实战**。从最小 Agent 出发，依次学习工具、结构化输出、记忆、MCP、RAG、工作流、人工审核、多智能体、上下文工程、可靠性与评测。

**整库最终收尾版｜2026-10-27。** 12 章内容与现有教学实现已齐备；本轮逐章审查源码并修复 8 类问题；172 项项目测试全部通过，第 03—12 章的 10 个离线自检入口全部通过（含真实 MCP stdio）。第 11、12 章的完成范围为**离线教学验收**，在线模型与 LangSmith 服务端验收未执行。项目不宣称生产就绪。

## 一、先跑通离线闭环

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。所有命令均在仓库根目录执行。

```bash
# 首次安装；使用仓库锁文件，不升级依赖。
uv sync --locked

# 不配置 API Key，先验证第 12 章的真实 LangGraph 离线流程。
uv run --locked --offline python -B 12-agent-evaluation/main.py --self-test
```

预期 A、B、C、D 均为 `PASS`，最后输出 `总体验收：PASS；执行场景：A,B,C,D`。这里使用 Fake Tool 与固定路由，不请求模型、不上传 Trace。`--offline` 限制的是 uv 的依赖联网；本章不调用外部服务由自身实现保证。

开始在线第一章前，再配置本地 `.env`：

```bash
# 已有 .env 时保留原文件。
test -e .env || cp .env.example .env

# 填入有效 Gemini Key、模型 ID 后执行；会调用模型，可能计费。
uv run --locked python 01-basic-agent/main.py
```

详细安装、环境变量和排障见 [INSTALL.md](./INSTALL.md)。第一章以实际 `main.py` 为准；早期文章中的 `LChain-Demo/src/agent.py` 是独立入门示例的目录写法。

## 二、12 章学习路线

| 阶段 | 章节 | 学完应能说明 / 验证 | 当前验收范围 |
| --- | --- | --- | --- |
| Agent 基础 | [01 基础 AI Agent](./01-basic-agent/) | 区分模型请求工具与 Python 执行工具 | 历史在线 / Trace 记录见章内文档 |
| Agent 基础 | [02 Multi-Tool Agent](./02-multi-tool-agent/) | 检查工具选择、连续调用及 Call ID | 历史在线 / Trace 记录见章内文档 |
| Agent 基础 | [03 Structured Output](./03-structured-output/) | 区分 Schema 合法与事实正确 | 历史在线 / Trace 记录见章内文档 |
| Agent 基础 | [04 Agent Memory](./04-agent-memory/) | 验证会话记忆、更正与线程隔离 | 历史在线 / Trace 记录见章内文档 |
| 工具与知识 | [05 MCP 工具集成](./05-mcp-agent/) | 理解 MCP Client / Server 与工具适配 | 历史实验记录见章内文档 |
| 工具与知识 | [06 RAG Agent](./06-rag-agent/) | 追踪文档、检索证据与最终回答 | 历史实验记录见章内文档 |
| 流程编排 | [07 LangGraph Workflow](./07-langgraph-workflow/) | 理解 State / Node / Edge / Reducer | 历史实验记录见章内文档 |
| 流程编排 | [08 Human-in-the-loop](./08-langgraph-hitl/) | 验证 interrupt、审核、恢复和拒绝 | 历史实验记录见章内文档 |
| 流程编排 | [09 Multi-Agent](./09-langgraph-multi-agent/) | 区分 Subagents 委派与 Handoff | [小样本在线 / Trace 验收](./reports/chapter09-acceptance.md) |
| 工程基础 | [10 Context Engineering](./10-context-engineering/) | 管理裁剪、摘要、偏好及专家上下文 | [离线 + 小样本在线 / Trace 验收](./10-context-engineering/acceptance.md) |
| 工程基础 | [11 Agent Reliability](./11-agent-reliability/) | 检查有限重试、恢复、副作用与权限 | [离线教学验收通过](./reports/chapter11/acceptance.md) |
| 工程基础 | [12 Agent Evaluation](./12-agent-evaluation/) | 用 Dataset / Evaluator 验证轨迹、安全和回归 | [离线教学验收通过](./reports/chapter12/acceptance.md) |

建议按 01—12 顺序学习。已有基础时，可以按阶段复习。每章 README 是运行入口；`main.py` 是可执行入口；后期章节的 `basic-knowledge.md` 承接概念、源码讲解和流程图。

## 三、技术栈与复现

核心组件：Python、uv、LangChain、LangGraph、Pydantic、Gemini / DeepSeek、LangSmith。模型选择是各章既有实现的一部分；不要求所有章节使用同一提供商。

| 2026-10-27 本机检查项 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| LangChain / LangGraph | 1.4.6 / 1.2.16 |
| langchain-google-genai / langchain-deepseek | 4.4.2 / 1.1.1 |
| LangSmith / Pydantic | 0.14.11 / 2.14.1 |

完整依赖以 [pyproject.toml](./pyproject.toml) 和 [uv.lock](./uv.lock) 为准。这些版本是当前检查快照，不代表最新版，也不保证历史模型 ID 今天对所有账号开放。

```bash
# 项目离线回归；不请求模型或查询 LangSmith。
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  uv run --locked --offline python -B -m unittest discover -s tests -p 'test_*.py' -v
```

本次结果与范围见 [最终检查记录](./reports/final-closeout-20261027.md)。第 12 章的新证据同时记录 Dataset、Evaluator 和源码指纹，保留此前批次。

## 四、结业边界与后续使用

- 能力目标：能运行示例、解释状态与工具消息、定位 Trace、识别失败，并为修改补充必要的评测。
- Fake / Mock 只验证可控教学条件；`usage_metadata=null` 表示缺少真实模型用量，不是实测零 Token。
- `InMemorySaver` 与内存账本不能证明跨进程恢复；`thread_id` 不能替代身份认证或权限控制。
- 现有在线结果属于历史小样本；本次收尾未重复请求模型，也未重新查询 LangSmith。
- 持久化、真实身份与业务系统、长期质量评估、部署与运维属于进阶 / 扩展，按后续项目需求安排。本系列固定为 12 章。

`.env`、`.venv/`、`.runs/` 均为本地文件，不进入 Git。查看历史工程变更可读 [CHANGELOG.md](./CHANGELOG.md)；Trace 分析工具见 [scripts/README.md](./scripts/README.md)。
