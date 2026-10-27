# 第 12 章：Agent Evaluation 智能体评测与可观测性

**当前状态：✅ 离线教学验收通过｜2026-10-27 本次 Mac 复验：项目 172 项测试全部通过（含本章 22 项），真实 LangGraph A—D 自检全部 `PASS`。DeepSeek 在线实验及 LangSmith 服务端验收未执行。**

学习目标：构造冻结 Golden Dataset，检验 Tool Calling / 轨迹证据，验证 HITL 安全边界，并在相同数据集上比较 Baseline 与 Candidate。

| 内容 | 唯一入口 |
| --- | --- |
| 概念、函数讲解、Mermaid 与 A—D 观察点 | [基础知识及代码学习正文](./basic-knowledge.md) |
| 第 11 章可靠性背景 | [上一章 README](../11-agent-reliability/README.md) |
| 历次验收、证据边界与源码指纹 | [reports/chapter12/acceptance.md](../reports/chapter12/acceptance.md) |

## 一、运行方式

在项目根目录执行，复用既有 `uv.lock` 与依赖。首次未安装依赖时，先按 [安装说明](../INSTALL.md) 执行 `uv sync --locked`。

```bash
# 纯 Code Evaluator 单元测试 + 真实 LangGraph 集成测试
uv run --locked --offline python -B -m unittest discover -s tests -p 'test_chapter12.py' -v

# A—D：真实 LangGraph + Fake Tool；无 DeepSeek / LangSmith 调用
uv run --locked --offline python -B 12-agent-evaluation/main.py --self-test

# 单组实验（A/B/C/D）
uv run --locked --offline python -B 12-agent-evaluation/main.py --case B

# 保存一个全新的脱敏证据文件；已有路径会拒绝覆盖
uv run --locked --offline python -B 12-agent-evaluation/main.py --self-test \
  --evidence .runs/chapter12/offline-first.json
```

**本章不提供 `--online` / `--trace`；也不读取 `.env`。** 该命令明确关闭 LangSmith 自动追踪，以避免将虚构实验数据意外上传。本次在 Mac 实际工作区重新执行了项目测试与本章 `--self-test`，结果、依赖快照和源码指纹按新批次写入验收记录。`--self-test` 仍然需要实际安装的 LangGraph，并非纯 Python 模拟。

## 二、验收结果与证据边界

2026-10-27 的已有用户日志与本次直接复验分别保留：

- 已有用户日志：本章 `Ran 22 tests in 0.445s`，结果 `OK`。
- 本次直接复验：项目 `Ran 172 tests in 2.164s`，结果 `OK`，无跳过项；其中本章 22 项全部通过。
- `--self-test`：A、B、C、D 各为 `PASS`；`总体验收：PASS；执行场景：A,B,C,D`。
- 本次验收对象是 **LangGraph 真实执行 + Fake Tool / 固定路由 + 确定性评测器**；不是 DeepSeek 在线模型、LangSmith 服务端或生产流量验收。
- 本次新批次已记录源码 / 锁文件指纹与依赖版本，并保存 [脱敏离线证据](../reports/chapter12/offline-full-final-20261027.json)。此前容器缺依赖及用户日志缺指纹的历史仍保留在 [验收记录](../reports/chapter12/acceptance.md)。

## 三、阅读与验收原则

- **A**：冻结 Dataset 的 4 条工单，在真实图上运行后逐案评分。
- **B**：核对 `AIMessage.tool_calls` / `ToolMessage.tool_call_id`、顺序和缺失证据。
- **C**：真实 `interrupt()` → Checkpoint → `Command(resume=...)`，模拟写入只执行一次。
- **D**：同一 Dataset 下比较故意弱化的 Baseline 与 Candidate。

这里“真实”仅指本地 LangGraph 的节点和消息协议；路由规则、专家回复、内存账本均为 **Fake 教学替身**，并未执行真实模型推理或业务写入。不能把离线调用数、Fake 账本或 `usage_metadata=null` 当作 DeepSeek Token、LangSmith Trace 或生产级可靠性证据。

[返回 12 章系列目录](../README.md)
