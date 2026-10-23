# 第 09 章前置学习：LangGraph Multi-Agent 多智能体协作

**学习主题：Supervisor · Subagents · Handoff**

**项目：zrLangChain**

**阶段：基础知识学习（暂不进入 Demo 开发）**

---

## 01. 为什么需要 Multi-Agent？

### 1.1 从单 Agent 到多 Agent

在前面的章节中，我们学习了：

- 单 Agent 如何调用工具（Tool Calling）
- 如何使用 LangGraph 编排多个节点
- 如何通过 `interrupt()` 实现人工审核与断点恢复

但当一个 Agent 需要同时处理账号、物流、支付、企业商务等不同领域时，会出现以下问题：

1. 工具数量越来越多，模型容易选择错误工具。
2. System Prompt 不断膨胀，上下文管理困难。
3. 不同业务知识相互干扰。
4. 不同模块难以独立开发、测试和维护。

**Multi-Agent 的核心思路：将复杂任务拆分给专业 Agent，通过明确的协作机制完成整体目标。**

但 Multi-Agent 并不总是优于单 Agent。如果任务简单、工具很少，优先使用单 Agent。

### 1.2 多智能体的核心组成

| 组成 | 含义 | 作用 |
|---|---|---|
| Agent | 具备模型、工具及运行循环的智能体 | 执行任务与作出决策 |
| Supervisor | 中央协调者 | 分派任务、协调结果 |
| Subagent | 专业子智能体 | 处理特定领域任务 |
| Tool | 可调用能力 | 暴露 Agent 或业务函数 |
| Handoff | 控制权移交机制 | 切换当前处理主体或阶段 |
| State | 工作流状态 | 记录任务、消息与控制信息 |
| Checkpoint | 状态持久化快照 | 支持多轮对话与执行恢复 |

---

## 02. Supervisor：中央调度模式

### 2.1 什么是 Supervisor？

Supervisor 是一个负责协调其他 Agent 的主 Agent。

它通常负责：

- 理解用户的整体任务
- 决定调用哪个专业 Agent
- 必要时连续调用多个 Agent
- 整合专业 Agent 的输出
- 决定继续执行还是结束

Supervisor 不一定直接执行全部业务操作。

### 2.2 架构示意

```text
                  用户请求
                     │
                     ▼
               Supervisor
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Account Agent  Logistics Agent  Business Agent
        │            │            │
        └────────────┼────────────┘
                     ▼
               Supervisor
                     │
                     ▼
                 最终回复
```

例如用户提出：

“我的账号无法登录，而且 VIP 套餐也无法升级。”

Supervisor 可以先让 Account Agent 排查登录问题，再让 Business Agent 分析升级流程，最后汇总回答。

### 2.3 Supervisor 与 Router 的区别

| 对比项 | Router | Supervisor |
|---|---|---|
| 核心作用 | 分类与分发 | 持续协调与调度 |
| 决策次数 | 通常一次 | 可以多次 |
| 是否需要 Agent | 不一定 | 通常是 Agent |
| 是否整合多个专家结果 | 可由后续工作流负责 | 通常由 Supervisor 负责 |
| 典型用途 | 单次工单分类 | 复杂工单跨领域协作 |

**关键理解：Router 解决“这次交给谁”，Supervisor 还要解决“下一步做什么、是否需要继续委派”。**

### 2.4 Supervisor 的两种实现思路

**方式 A：使用 `langgraph-supervisor` 封装库**

旧教程常见：

- `create_supervisor()`
- `create_handoff_tool()`
- 将专业 Agent 组成分层图

这套 API 有助于理解 Supervisor 的历史实现，但官方已经不再积极维护该封装库。

**方式 B：Supervisor + Subagents as Tools（当前推荐）**

通过：

- `create_agent()` 创建 Supervisor
- `create_agent()` 创建专业 Agent
- `@tool` 将专业 Agent 封装成 Tool
- Supervisor 通过 Tool Calling 委派任务

第 09 章实践建议采用方式 B。

---

## 03. Subagents：专业子智能体模式

### 3.1 什么是 Subagent？

Subagent 是一个具有特定职责的专业 Agent。

例如：

| Subagent | 职责 | 可用能力 |
|---|---|---|
| Account Agent | 登录、密码、账户安全 | 账号知识库、状态查询 |
| Logistics Agent | 运单、发货、物流异常 | 运单查询、物流规则 |
| Business Agent | VIP、报价、企业需求 | 套餐知识库、商务规则 |

Subagent 可以有自己的 Prompt、模型和工具集。

在典型的 **Subagents as Tools** 架构中，Subagent 不直接负责面向用户的最终交流，而是将结果返回给 Supervisor。

### 3.2 核心机制：Agent as Tool

```text
Supervisor
    │
    │ Tool Calling
    ▼
call_account_agent(query)
    │
    ▼
Account Subagent
    │
    │ 内部推理、调用工具
    ▼
专业处理结果
    │
    ▼
Tool Result
    │
    ▼
Supervisor
```

关键不是将一个 Python 函数命名为 Agent，而是函数内部调用了一个具有自身执行循环的专业 Agent。

### 3.3 Subagent 的上下文隔离

这是本章最重要的知识之一。

假设 Supervisor 的历史消息已经有 100 条：

- 不一定需要将这 100 条消息全部发送给 Account Agent。
- 可以仅传入当前账号问题及必要的业务信息。
- Account Agent 完成任务后，只返回总结结果。
- Supervisor 继续维护整体对话。

这种方式可以降低无关上下文的干扰。

**上下文隔离不等于安全权限隔离。** 数据访问权限仍然需要独立验证和控制。

### 3.4 Subagent 是否具有 Memory？

典型的 Subagent Tool 调用采用短生命周期上下文，默认不依赖跨调用的对话记忆。

但是，如果专门为 Subagent 配置持久化与状态管理，也可以设计有状态的子智能体。

需要区分：

- Supervisor 的对话记忆
- Subagent 单次任务的临时上下文
- 跨调用的 Subagent 持久化记忆

不能简单理解为“所有 Subagent 永远无状态”。

### 3.5 两种工具封装方式

**方式一：每个 Agent 对应一个 Tool**

例如：

- `call_account_agent`
- `call_logistics_agent`
- `call_business_agent`

优点是职责和输入输出清晰，适合初学者。

**方式二：统一分发 Tool**

例如：

`task(agent_name, description)`

通过 Agent Registry 根据名称选择专业 Agent。

适合 Agent 数量较多或动态扩展的场景。

第 09 章建议先采用第一种方式。

---

## 04. Handoff：Agent 控制权移交

### 4.1 什么是 Handoff？

Handoff 指的是：**根据工作流状态，把当前处理权切换给另一个 Agent 或新的处理阶段。**

例如：

用户最初咨询物流问题：

“我的快递三天没更新，我想申请退款。”

流程可能是：

```text
用户
 │
 ▼
Logistics Agent
 │
 │ 判断问题需要退款处理
 ▼
Handoff
 │
 ▼
Refund / Business Agent
 │
 ▼
继续与用户交互
```

与 Subagent Tool Calling 的关键区别：

**Subagent：完成委派任务后，通常把结果返回 Supervisor。**

**Handoff：切换当前处理主体，后续交互可以由新 Agent 或新阶段继续接管。**

### 4.2 Handoff 不一定需要多个 Agent

官方文档给出了两种实现路径。

**模式 A：单 Agent + 动态状态**

只有一个 Agent，但根据 `current_step` 动态改变 Prompt 和可用工具。

```text
Agent
  │
  ├── triage：收集问题
  │
  ├── verification：核实资料
  │
  └── specialist：专业处理
```

适合多阶段客服对话和流程约束。

**模式 B：多个 Agent + Graph 控制权转移**

每个专业 Agent 作为独立 Graph 节点，使用控制命令切换执行位置。

```text
Support Agent
      │
      ▼
 Command(goto="sales_agent")
      │
      ▼
 Sales Agent
```

第 09 章建议优先实践模式 B，更容易观察真实的多 Agent 状态和节点转移。

### 4.3 Handoff 依赖状态

常见状态字段包括：

| 字段 | 含义 |
|---|---|
| `messages` | 对话消息 |
| `active_agent` | 当前负责交互的 Agent |
| `current_step` | 当前处理阶段 |
| `handoff_reason` | 为什么移交 |
| `handoff_count` | 已发生多少次转移 |
| `case_id` | 当前工单标识 |

通常由工具更新状态，再由 Graph 或动态配置逻辑决定后续执行。

### 4.4 Handoff 与第 08 章 HITL 的区别

| 对比 | HITL | Handoff |
|---|---|---|
| 核心目的 | 等待人工介入 | 转移 Agent 或工作阶段 |
| 典型 API | `interrupt()`、`Command(resume=...)` | `Command(update=...)`、`Command(goto=...)` |
| 是否必须暂停 | 通常暂停等待外部输入 | 不一定 |
| 是否需要人工 | 常见，但也可等待其他外部输入 | 不一定 |
| 典型场景 | 高风险操作审核 | 跨部门客服转接 |

两种机制可以组合：移交给商务 Agent 后，若涉及实际扣费，可以再通过 HITL 请求人工审批。

---

## 05. 三个概念的关系

这是本章最需要彻底理解的部分。

| 概念 | 所处层次 | 回答的问题 |
|---|---|---|
| Supervisor | 协调角色 / 架构 | 谁负责组织和决策？ |
| Subagents | 专业任务执行单元 | 谁负责具体工作？ |
| Handoff | 控制流程机制 | 当前执行权交给谁？ |

**Supervisor 和 Subagents 经常构成同一种架构；Handoff 可以独立存在，也可以参与复杂的多 Agent 系统。**

不能将三者理解成相互排斥、必须三选一的框架。

### 5.1 决策参考

**需求一：一个工单需要账号专家和商务专家分别分析，最后统一答复。**

优先考虑 Supervisor + Subagents。

**需求二：客服已经完成初步核实，需要把用户对话移交给退款专员。**

优先考虑 Handoff。

**需求三：只需要根据工单类别选择一个固定处理模块。**

优先考虑 Router 或普通 LangGraph 条件分支，不必引入完整 Multi-Agent。

---

## 06. 需要认识的核心 API

本节只学习 API 的职责，不急于编写完整代码。

| API | 所属模块 | 用途 |
|---|---|---|
| `create_agent()` | `langchain.agents` | 创建主 Agent 或专业 Agent |
| `@tool` | `langchain.tools` | 将能力封装为模型可调用工具 |
| `agent.invoke()` | Agent Runnable | 执行同步 Agent 调用 |
| `agent.ainvoke()` | Agent Runnable | 执行异步 Agent 调用 |
| `StateGraph` | `langgraph.graph` | 构建多节点工作流 |
| `add_node()` | StateGraph | 注册 Agent 或业务节点 |
| `add_conditional_edges()` | StateGraph | 根据状态决定下一节点 |
| `Command(update=...)` | `langgraph.types` | 更新图状态 |
| `Command(goto=...)` | `langgraph.types` | 指定后续节点 |
| `Command.PARENT` | Command | 指定路由到父图 |
| `InMemorySaver` | `langgraph.checkpoint.memory` | 保存进程内图执行状态 |

### 6.1 `Command(goto=...)` 的含义

表示当前节点希望 Graph 下一步执行指定节点。

### 6.2 `Command.PARENT` 的含义

在子图中需要把控制权返回父图时，可以指定 `graph=Command.PARENT`。

它不是所有 Handoff 都必须使用的参数。只有涉及父子图边界转移时，才需要根据实际 Graph 结构使用。

### 6.3 工具消息的完整性

当模型发起 Tool Call 后，需要保证相应的工具响应具有匹配的 `tool_call_id`。

在 Handoff 工具更新 `messages` 时，通常需要构造正确的 `ToolMessage`，避免形成不合法的模型消息历史。

### 6.4 图状态与上下文的区别

- **State**：运行中的状态数据，例如当前 Agent、工单类别、处理进度。
- **Context**：某个 Agent 实际接收到的任务信息、Prompt、消息和工具结果。

Graph 具有共享 State，不代表每个 Agent 都必须读取全部上下文。

---

## 07. 第 09 章的建议实验模型

先保持三个相互独立的学习实验，再进行组合。

### 实验 A：Supervisor + Subagents

输入：

“我的账号登录失败，而且 VIP 套餐需要升级。”

学习目标：

1. Supervisor 理解跨领域请求。
2. 调用 Account Subagent。
3. 调用 Business Subagent。
4. 整合两个结果。
5. 输出统一答复。

需要观察：Tool Calling、调用顺序、上下文输入输出和最终整合。

### 实验 B：Handoff

输入：

“快递一直没有更新，我想直接申请退款。”

学习目标：

1. Logistics Agent 识别退款需求。
2. 更新 `active_agent`。
3. 将执行权交给退款或商务 Agent。
4. 验证新的 Agent 成为当前处理主体。
5. 验证不会无限循环转接。

需要观察：Handoff 前后状态变化、Graph 节点执行顺序。

### 实验 C：多 Agent + HITL

输入：

“我要升级企业套餐，并立即完成收费。”

学习目标：

1. Supervisor 委派商务分析。
2. 商务 Agent 准备操作建议。
3. 需要实际收费时进入人工审核。
4. 审核完成后继续执行或拒绝。

这一实验将第 07、08、09 章的核心知识连接起来。

**以上为后续实验设计，不表示已经实现这些功能。**

---

## 08. Multi-Agent 工程设计注意事项

### 8.1 Context Engineering

明确每个 Agent 的输入与输出契约。

不要简单复制全部历史消息给所有 Agent。

### 8.2 防止无限委派

Supervisor 反复调用 Subagent，或者两个 Agent 互相 Handoff，都可能造成循环。

需要限制调用轮次、Handoff 次数或 Graph 执行步数，并提供明确的终止条件。

### 8.3 错误隔离

区分：

- 模型 API 失败
- Subagent 内部失败
- 工具执行失败
- Handoff 目标无效
- Graph 状态不一致

对失败的 Subagent 不应默认重复执行整个多 Agent 流程。

### 8.4 权限边界

Account Agent 不应因为同属一张 Graph，就自动获得退款或支付权限。

高风险操作需要业务层权限验证，必要时增加人工审批。

### 8.5 可观测性

后续结合 LangSmith 验收：

- Supervisor 的实际决策
- Subagent 调用次数与顺序
- Handoff 来源与目标
- 是否存在循环或不必要的重复调用
- LLM 调用量及 Token 成本
- 最终业务结果是否符合要求

---

## 09. 推荐学习顺序

| 阶段 | 重点 | 完成标准 |
|---|---|---|
| S1 | Multi-Agent 基础概念 | 能说明何时需要多 Agent |
| S2 | Supervisor | 能解释中央调度与 Router 的区别 |
| S3 | Subagents | 能解释 Agent as Tool 与上下文隔离 |
| S4 | Handoff | 能解释状态切换与控制权移交 |
| S5 | LangGraph API | 理解 State、Command、ToolMessage、Checkpoint 的关系 |
| S6 | 综合设计 | 能为一个工单画出合理的协作流程 |

---

## 10. 学习完成自检

进入第 09 章代码阶段之前，应能够独立回答：

1. Supervisor 和 Router 的核心差别是什么？
2. 为什么把 Subagent 包装成 Tool 可以减少上下文干扰？
3. Subagent 返回结果，与 Handoff 移交控制权，有什么本质区别？
4. `Command(goto=...)` 与 `Command(resume=...)` 分别解决什么问题？
5. 什么情况下需要 `Command.PARENT`？
6. 为什么 Handoff 中要关注 `ToolMessage` 与 `tool_call_id`？
7. 如何避免两个 Agent 反复互相转接？
8. 应该如何在 LangSmith 中证明一次真正的多 Agent 协作已经发生？

**这八个问题理解清楚，再开始编写第 09 章代码。**

---

## 11. 官方学习资料

按以下顺序阅读：

1. [Multi-Agent Overview](https://docs.langchain.com/oss/python/langchain/multi-agent) — 总体概念与模式选择
2. [Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents) — 专业 Agent 与 Tool 封装
3. [Build a Personal Assistant with Subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant) — Supervisor 官方实践
4. [Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs) — 状态驱动与控制权转移
5. [Build Customer Support with Handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support) — 客服场景实例
6. [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) — StateGraph、Command、State
7. [Migrate from langgraph-supervisor](https://docs.langchain.com/oss/python/migrate/langgraph-supervisor) — 理解旧封装与当前推荐方案

---

## 12. 下一阶段：进入第 09 章代码实践

完成理论学习后，按照以下顺序实施：

**Supervisor + Subagents → Handoff → 综合协作 → 离线测试 → DeepSeek 在线实验 → LangSmith Trace 验收。**

项目将延续前八章的教学风格：中文注释、核心 API 关系图、真实 LangGraph 运行验证、实验日志与结构化验收结果。

最终目标不仅是运行一个多 Agent Demo，而是能够准确解释：

**任务如何分配、上下文如何传递、控制权如何转移、状态如何维护，以及如何证明整个协作过程正确。**