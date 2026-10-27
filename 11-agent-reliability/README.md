# 第 11 章：Agent Reliability 可靠性、安全与异常治理

**当前阶段：✅ 离线教学验收通过。A—D 最小离线实验及历史证据见下方记录；2026-10-27 项目 172 项回归全部通过。DeepSeek 与 LangSmith 服务端验收未执行，进阶能力另列扩展。**

学习目标：区分可重试故障与必须停止的条件；掌握图恢复、幂等副作用、工具权限、人工审批与多智能体故障收敛。

| 你要做什么 | 唯一入口 |
| --- | --- |
| 学习原理、看代码和全部流程图 | [学习正文](./basic-knowledge.md) |
| 按图定位函数 | [入口与批次](./basic-knowledge.md#entry) · [重试](./basic-knowledge.md#retry) · [恢复](./basic-knowledge.md#parallel) · [审批](./basic-knowledge.md#approval) · [转交](./basic-knowledge.md#handoff) |
| 查看实验观察点、范围与自测 | [A—D 设计与覆盖](./basic-knowledge.md#experiments) · [知识自测](./basic-knowledge.md#mistakes) |
| 查看实际测试结果和历史证据 | [reports/chapter11/acceptance.md](../reports/chapter11/acceptance.md) |

<a id="offline"></a>

## 一、运行

在项目根目录执行，复用现有环境，无需新增依赖：

```bash
# 全部离线场景与关键断言。
uv run --locked --offline python 11-agent-reliability/main.py --self-test

# 单组实验；可选 A、B、C、D，默认始终离线。
uv run --locked --offline python 11-agent-reliability/main.py --case B

# 本章边界与集成测试。
uv run --locked --offline python -m unittest discover -s tests -p 'test_agent_reliability*.py' -v
```

需要保存本次证据时，选择尚不存在的文件名：

```bash
uv run --locked --offline python 11-agent-reliability/main.py --self-test \
  --evidence .runs/chapter11/offline-first.json
```

已有证据文件会在实验执行前被拒绝。`--self-test` 总是运行四组；本章不提供 `--online` / `--trace`，不读取 `.env`。`--locked --offline` 不改写锁文件、不联网解析依赖；环境缺包时会报错。

## 二、阅读与维护

学习正文按“概念 → 对应函数 → 流程图 → 关键解释 → 观察点”组织。场景 PASS 表示故障按预期处理，业务中的 FAILED 或 REJECTED 可能正是正确结果。

运行命令与当前状态只在本页维护；原理、图和实验设计只在学习正文维护；测试与修复历史只追加到验收记录。

[返回全系列目录](../README.md)
