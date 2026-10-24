# 第 10 章：Context Engineering 上下文工程与记忆管理

**当前阶段：离线、DeepSeek A—D 小样本与 LangSmith Trace 阶段验收通过（2026-10-24）。**

本章在第 04 章会话记忆基础上，练习历史裁剪、摘要、跨线程偏好和专家上下文隔离。所有用户、订单与物流数据均为虚构教学数据，不连接真实订单系统。

| 你要做什么 | 从这里开始 |
| --- | --- |
| 学习概念与订单 A100 的完整流程 | [basic-knowledge.md](./basic-knowledge.md#order-example) |
| 运行实验 | [离线运行](#offline) → [A—D 观察点](#experiments) |
| 对照源码理解实现 | [代码阅读路线](#code-map) |
| 阅读实际执行结果 | [acceptance.md](./acceptance.md)：80 项项目离线回归、在线 A—D 与 7 条 Trace 验收通过 |
| 了解在线结果或复现实验 | [DeepSeek 与 LangSmith](#online) |

结果来自已有验收记录：带 Trace 批次的 14 次模型调用共 7,229 Tokens；加上先导 A 的 1,054 Tokens，本轮在线合计 8,283 Tokens。结论限虚构教学小样本，不代表长期质量或生产能力。

<a id="offline"></a>

## 一、离线运行：先完成最小闭环

在仓库根目录执行，使用统一的 uv 环境：

```bash
# 依赖环境未同步时先执行；--locked 防止改变已有锁文件。
uv sync --locked

# 最先运行这一条：强制离线执行 A—D 全部场景与断言。
uv run python 10-context-engineering/main.py --self-test

# 再选一组实验阅读对应代码；A / B / C / D 均可选择。
uv run python 10-context-engineering/main.py --offline --case B

# 运行完整行为与边界测试。
uv run python -m unittest discover -s tests -p 'test_context_engineering.py' -v
```

脚本默认离线；`--offline` 可以省略。`--self-test` 始终运行全部 A—D，不接受同时指定 `--online`。离线脚本不读取 `.env`，不创建 DeepSeek 客户端，不上传 LangSmith；环境依赖的首次下载与同步由 uv 单独处理。

如需保留证据，使用一个尚不存在的文件名：

```bash
uv run python 10-context-engineering/main.py --self-test \
  --evidence .runs/chapter10/offline-first.json
```

证据文件已存在时会拒绝覆盖。若要再跑一次，请更换文件名；`.runs/` 是本地忽略目录。

<a id="experiments"></a>

## 二、A—D：运行后观察什么？

离线只替换模型响应与摘要服务，Agent、Middleware、工具、Checkpoint 和 Store 真实执行。四组实验分别回答一个问题：

| 实验 | 要回答的问题 | 实际检查 |
| --- | --- | --- |
| A：线程记忆与裁剪 | 保存的历史是否必须全部发给模型？ | State 保留旧套餐历史；本次模型输入排除它；A100 仍可回忆，另一线程不知道 A100 |
| B：摘要与预算 | 历史压缩后，用户更正是否仍有效？ | 摘要写回 State；旧消息被替换；A100 在作废列表、B200 为有效订单；裁剪保留前导摘要 |
| C：跨线程长期记忆 | 换对话后如何使用偏好，并守住边界？ | 同用户共享偏好，其他作用域不共享；检查实际偏好工具调用、版本冲突、到期、删除与精确 namespace |
| D：专家隔离与筛选 | 每位专家是否只收到必要信息？ | 账号专家不接收订单 / 支付信息，物流专家不接收登录问题；过滤电话和调试字段；两组工具消息配对 |

先读 `main.py` 的 `run_case()` 对照断言，再到实际构建函数中追踪实现。下面从现有自检字段中摘取两个检查项，仅列部分字段：

```json
[
  {"case": "B", "checks": {"correction_relation_preserved": true}},
  {"case": "C", "checks": {"preference_tool_reads_verified": true}}
]
```

B 不仅要求最终答案是 B200，还解析真实摘要的 JSON 事实。C 不仅要求回答“简洁”，还检查 `read_preference` 对应的真实 ToolMessage 与读取值，避免把正确回答误当成工具执行证据。

<a id="code-map"></a>

## 三、代码阅读路线

| 顺序 | 文件与入口 | 阅读重点 |
| --- | --- | --- |
| 1 | [context_policy.py](./context_policy.py)：`validate_protocol()`、`select_history()` | 工具调用如何配对；按完整轮次裁剪；保留摘要；装不下时为何报 `BudgetExceeded` |
| 2 | [context_policy.py](./context_policy.py)：`MemoryRepository` | namespace、白名单、版本检查、有效期与删除墓碑 |
| 3 | [main.py](./main.py)：`build_demo()`、`Demo.invoke()` | 动态 Prompt、运行身份、工具、摘要、Checkpointer 和 Store 如何组装 |
| 4 | [main.py](./main.py)：`run_case()`、`Capture`、`main()` | 夹具、实际模型输入捕获、断言、CLI 与证据输出 |
| 5 | [offline_model.py](./offline_model.py) | 模型替身如何产生可重复决策；它与真实模型能力的区别 |
| 6 | [测试文件](../tests/test_context_engineering.py) | 正常行为、故障注入、协议错误、隔离与证据防覆盖 |
| 7 | [只读 Trace 验收器](../scripts/verify_chapter10_trace.py)与[边界测试](../tests/test_chapter10_trace.py) | 精确 ID 读取、实际摘要送达、唯一叶子模型计量与服务端对账 |

B 实验用 `trigger=("messages", 10)`、`keep=("messages", 2)` 便于观察摘要触发。摘要服务返回 `superseded_order_ids`、`active_order_id` 和 `constraints`；前两项用于核对订单更正，JSON 格式本身不保证事实正确。

C 将 `AppContext` 中的租户、用户组成 `(tenant_id, user_id, "preferences")`。Store 的 `search` 按前缀查询，仓库额外比较结果 namespace 精确相等，排除子空间同名条目。

D 是 Subagents as Tools：包装工具为两个独立 Agent 构造必要输入，并返回建议、来源与截断标记。本章没有新增 Handoff 流程，相关机制见[第 09 章](../09-langgraph-multi-agent/README.md)。

<a id="evidence"></a>

## 四、如何读结果与失败？

| 字段 / 结果 | 含义 |
| --- | --- |
| `status="PASS"`、退出码 0 | 对应场景的实际断言通过 |
| `status="FAILED"`、非零退出码 | 场景运行失败；仅保留异常类型，后续停止 |
| `status="NOT_RUN"` | 因前一场景失败而尚未运行，不能算通过 |
| `checks` | 对实际状态、输入或调用结果执行的检查 |
| `model_calls` | 当前实验捕获的离线 / 在线模型调用输入批次数 |
| `usage_metadata=null` | 未获得提供商用量，不能填成真实 Token 为零 |
| `observed_model_run_ids` | 本地回调观察到的调用标识，不证明 LangSmith 已收到 Trace |
| `trace_requested=true` | 已请求启用追踪，服务端接收与内容完整性仍需另验 |

导出 JSON 不保存模型回答或对话原文；`Capture` 在进程内保留实际输入供断言使用。启用 LangSmith 是另一条数据发送路径，见下一节。

遇到失败时，先确定失败场景，再对照其断言、预算和工具消息检查。`--evidence` 保存的是脱敏结果，不包含完整调试信息；配置阶段失败可能只输出异常类型，不能据此认定某个场景已运行。原始验收证据和修复历史见 [acceptance.md](./acceptance.md)。

<a id="online"></a>

## 五、DeepSeek 与 LangSmith：结果与复现

已完成先导 A 和一轮 A—D 在线实验；随后只读核对了 7 条 Trace、70 个 span，叶子模型 Token 与本地提供商返回值一致。详细分组统计、调用标识与边界见 [acceptance.md](./acceptance.md)。以下命令用于复现，会产生新的模型请求。

根目录 `.env` 沿用 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`；开启 Trace 时另需 `LANGSMITH_API_KEY`，可沿用 `LANGSMITH_PROJECT`。此处不固定未经核验的模型名称。

```bash
# ① 先单组验证；真实模型请求会产生提供商费用。
uv run python 10-context-engineering/main.py --online --case A \
  --evidence .runs/chapter10/deepseek-A-first.json

# ② 小样本验证后再检查 A—D 与 Trace；请使用新的证据文件名。
uv run python 10-context-engineering/main.py --online --case all --trace \
  --evidence .runs/chapter10/deepseek-trace-first.json
```

`--online` 不自动开启 Trace，即使 `.env` 中存在 `LANGSMITH_TRACING=true`；必须显式加 `--trace`。开启后会将虚构实验的调用输入输出发送至 LangSmith。

在线 A—D 已全部通过且原始证据已保存后，使用仓库内的只读工具验收服务端记录：

```bash
uv run python scripts/verify_chapter10_trace.py \
  .runs/chapter10/deepseek-traced-20261024-first.json \
  .runs/chapter10/langsmith-audit-repeat.json

# 单独检查验收器的离线边界，不请求模型或 LangSmith。
uv run python -m unittest discover -s tests -p 'test_chapter10_trace.py' -v
```

该工具读取完整 A—D 已追踪在线批次，按精确模型 ID 查询，复用 `langsmith_reader.py` 的完整调用树校验；拒绝离线、失败、部分批次及不完整用量。输出路径必须不存在。只读查询不产生新的 DeepSeek 模型请求，原始输入输出只用于内存验收。

复现或升级后，按以下顺序重新验收：

1. 核实当前模型名称、窗口限制和工具能力，再检查 A—D 的实际回答、工具选择与摘要事实。
2. 在 LangSmith 对照模型、摘要、专家及工具调用链，检查工具调用与 `tool_call_id` 的关联。
3. 逐调用核对提供商用量；摘要与专家模型调用一并计入，避免父子 run 重复累计。缺失值记为不可得，当前代码不计算费用。本次已依此完成服务端核对；原始在线 JSON 的 `trace_acceptance="pending"` 不回写，独立审计 JSON 的 `status="PASS"` 记录后续验收结论。

当前在线配置为 `temperature=0`、`timeout=45`、`max_retries=0`、`max_tokens=600`。摘要通过显式服务调用，避免内置模型摘要包装器的默认重试；工具循环使用 `recursion_limit=12`。这些设置不构成整次实验的总费用硬上限。

<a id="limits"></a>

## 六、当前实现边界

| 范围 | 当前做法与限制 |
| --- | --- |
| 身份与权限 | `AppContext` 与进程内线程归属检查用于教学；没有真实登录、生产级 ACL 或订单接口 |
| 存储与并发 | 内存后端，退出进程后丢失；版本锁与墓碑限当前仓库实例，没有跨进程事务 |
| 过期与删除 | 读取时隐藏过期条目；删除只覆盖 Store 入口，不清除旧 State、摘要、Checkpoint、Trace 或备份 |
| 历史预算 | 使用消息估算，未涵盖全部系统提示、工具 schema 和协议开销，不是整个模型窗口的精确硬限制 |
| 摘要兼容性 | 识别 `lc_source="summarization"` 及当前前缀格式，已核对安装版本源码；升级后需复验 |
| 未实施扩展 | 数据库后端、自动 TTL 清理、向量检索、完整删除链与跨进程一致性 |

摘要仍可能有损；保留摘要之外的长期关键约束，也需要另行结构化管理。学习原理见[基础知识文档](./basic-knowledge.md)，当前结果与边界以[验收记录](./acceptance.md)为准。

[返回全系列目录](../README.md)
