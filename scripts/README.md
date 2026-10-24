# zrLangChain｜LangSmith Trace Analyzer

项目级**只读 Trace 分析工具**，服务于 `07` 及后续章节，不需要逐条导出 LangSmith JSON。使用官方 Python SDK 新异步 `client.runs.query()` 查询根 Run，`client.traces.list_runs()` 按精确 Trace ID 读取完整调用树。`langsmith_reader.py` 显式选择验收字段、绑定 project_id，并将新版大写 run_type、parent_run_ids、token_details.raw 归一化。只统计 LLM Run 的 Token；缺失计量保留未知，不以0冒充。

## 1. 安装与配置

把 `scripts/langsmith_report.py`、`scripts/README.md` 和 `tests/test_langsmith_report.py` 放到现有 `zr-langChain` 根目录对应位置：

```text
zr-langChain/
├── .env
├── pyproject.toml
├── scripts/
│   ├── langsmith_reader.py
│   ├── verify_chapter09_trace.py
│   ├── langsmith_report.py
│   └── README.md
└── tests/
    └── test_langsmith_report.py
```

项目已声明 LangSmith 依赖，在 VS Code 本地终端同步现有环境：

```bash
uv sync
```

沿用根目录 `.env`：

```dotenv
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=zr-langChain
LANGSMITH_TRACING=true
```

`LANGSMITH_TRACING` 是**实验时**向平台上报 Trace 的配置；分析工具只是**读取**已存在的 Trace，无需再次调用 DeepSeek。

## 2. 自动分析

先完成第八章在线实验，然后尽快运行：

```bash
uv run python 08-langgraph-hitl/main.py --case all
uv run python scripts/langsmith_report.py --chapter 08 --since-hours 1 --expect-online --expect-all
```

第七章也可以分析：

```bash
uv run python scripts/langsmith_report.py --chapter 07 --since-hours 24
```

输出 `reports/chapter08-trace-report.md` 和 `reports/chapter08-trace-summary.json`；两份文件默认不包含工单、Prompt、LLM 回答或凭据。可用于 README 实测记录与代码评审，但提交公开仓库前仍应人工检查一次。

## 3. 第八章检查规则

- 识别 `chapter08-approve/reject/edit/isolation-A/isolation-B-pause|resume` 根 Run。
- **Pause**：检查根输出 `__interrupt__`，或验证直属 `human_review` 的 `GraphInterrupt`。error 形式仅在暂停 State 正确、后续同线程同批次成功恢复、全部已查询 Run 已结束且没有重复生成时豁免；保留原始 error、预期中断和实际错误三类计数。如 `--expect-online`，每个 Pause 应有 1 条 LLM Run。
- **Resume**：检查 `review_action/status/steps`、`human_review` 子 Run，不得重复执行 `prepare_draft` 或再次调用 LLM。
- **配对证据**：两种中断路径均检查双方明确的线程标识（`thread_id` 或脱敏指纹）、非空 `batch_id` 和起止时间。线程/批次冲突、resume 早于 pause 或区间重叠判 FAIL；证据缺失保留 REVIEW。有 error 且证据不足的中断不予豁免，仍保留错误计数。
- **线程隔离**：`isolation-A` 与 `isolation-B` 必须使用不同线程；复用线程判 FAIL，缺少另一场景证据时保留 REVIEW。
- **完整性**：根 Run 和全部已查询到的子 Run 都必须有有效结束时间；未结束的 Trace 禁止 PASS，也不能用于中断豁免。该检查不能证明尚未被服务端返回的记录均已写入。
- **数字边界**：`--case all` 在线运行通常对应 5 个工单线程、10 次根 Graph Run 和 5 次 DeepSeek 调用（无失败或重试的情况下）。报告按实际观测计数，不硬编码通过。
- 如果当前窗口混有之前的多轮实验，`--since-hours` 应缩小。对同名运行仅选最新一条参与配对，但所有匹配 Run 都会计入总体 Token；要做单轮精确成本统计应避免多批次混在同一窗口。

> LangSmith SDK 返回的某些 Graph 根 Run 不一定携带 `thread_id` 或 `__interrupt__` 负载。缺少根中断负载时可用受限的节点中断及后续恢复证据验证；缺少双方线程/批次/时间证据则保留限制，不凭名称推断通过。对于线程隔离，**最终验收还要结合已通过的第八章 `--self-test`**。

## 4. 测试与排障

```bash
uv run python -m unittest discover -s tests -p 'test_langsmith_report.py' -v
```

- `缺少 LANGSMITH_API_KEY`：检查根目录 `.env`，不要将 Key 写进脚本。
- `FAIL`：实际检测到缺失场景、错误 Run 或与预期不符的状态路径，需要查 LangSmith 原始 Trace。
- `REVIEW`：数据不足、Token 缺失、无法核实线程配对或存在多批次 Run；不等价于测试通过。
- `PASS`：当前可观测的规则通过，仍不代表生产级安全与人工流程已经完成验收。
- `--strict`：存在失败项时返回退出码 1；适合 CI，但报告会先落盘。
- `--max-roots`：根查询自动分页，多读1条确认超限，超限报告 REVIEW。
- `--max-spans`：精确 Trace 接口返回完整 items 后检查本地分析上限；恰好等于上限可验收，超限拒绝分析。它不限制服务端单次响应体大小。缺根、缺父、跨 Trace 或祖先链不一致均拒绝分析。

## 5. 隐私与版本管理

工具不会持久化原始 Trace。报告可能包含 Run 名称/ID、状态、节点路径和用量，不包含运行文本。使用本地 Git 提交时先复核生成报告，再决定是否公开。不要将 `.env`、API Key 或原始 Trace 上传到公开仓库。未经本地验证前，不能把这里的开发完成理解为已在你的 LangSmith 账号上完成在线验收。

官方文档：[新 Run/Trace 读取](https://docs.langchain.com/langsmith/smithdb-sdk-migration-runs) · [新根 Run 查询](https://docs.langchain.com/langsmith/smithdb-sdk-migration-query-runs)

## 6. Experiment Reliability V1：错误分类与用例级断点验收

这是第八章线上限额失败后的工程化扩展，**不属于 LangGraph 节点和 Checkpoint 的一部分**。

新增两个仅依赖 Python 标准库的模块：

- `scripts/experiment_errors.py`：识别 `QUOTA_EXHAUSTED / INSUFFICIENT_BALANCE / RATE_LIMIT_TRANSIENT / MODEL_ERROR / WORKFLOW_ERROR / ASSERTION_FAILED`；识别明确的每日额度证据或 DeepSeek 余额不足 402 时终止整批，不自动等待或重试。
- `scripts/experiment_runner.py`：把在线场景的批次记录存为 `.runs/chapter08/latest.json`。仅存储场景状态、匿名批次 ID、错误类别及重试建议；采用临时文件原子替换。

```bash
# 新批次（默认），可先运行单个场景
uv run python 08-langgraph-hitl/main.py --case approve

# 继续这一批：已 PASS 的 approve 不会再次请求 DeepSeek
uv run python 08-langgraph-hitl/main.py --case all --resume-batch

# 完成后，只读取同一批次的 Trace
uv run python scripts/langsmith_report.py --chapter 08 --batch-latest \
  --since-hours 24 --expect-online --expect-all

# 不用调用 API 的模拟验收
uv run python -m unittest discover -s tests -p 'test_*.py' -v
```

`--resume-batch` 是**重新运行未通过的场景**，不是跨进程恢复已暂停的 HITL 节点。版本、模型不匹配时拒绝跳过旧 PASS。隔离场景被视为一个完整场景，任何中断都需要整体重试。

请在仓库根目录的 `.gitignore` 加入 `.runs/`。`reports/` 下的脱敏验收报告如需公开提交，仍需人工复核；**不要提交 `.env` 或原始 Trace**。


## 7. SmithDB 新接口迁移（2026-10-24）

本地 `langsmith==0.14.11` 已支持本次新接口，无新增依赖，也未升级锁文件。
CLI 参数保持兼容；报告工具同步入口内部运行异步读取，异步调用方可直接使用 `load_remote_async`。

- 根查询显式传 `project_ids`、`min_start_time/max_start_time` 和 `selects`，避免默认24小时窗口意外漏查旧记录。
- 精确 Trace 直接使用 `traces.list_runs(trace_id, project_id=...)`，不重复单独读取根 Run，也不调用旧 `read_run/list_runs`。
- 原本的章节、批次、HITL、消息及 Token 判定保持不变；父子关系使用 `parent_run_ids` 最近的最后一项。
- 新版 Graph Token 不是子调用合计；所有统计继续只累加 LLM Run。不上传原始对话，不生成公开 Trace URL。

第09章复核既有 Trace：

```bash
# 真实UUID证据只保存在Git忽略目录；reports/chapter09/为公开匿名版本，不能直接重查。
uv run python scripts/verify_chapter09_trace.py \
  .runs/chapter09-private-evidence/online-evidence.json .runs/ch09-smithdb-trace.json
```

两项精确验收 PASS，与旧结果的完整调用树、消息配对、5次LLM与1,988 tokens一致。
通用报告的七天窗口真实读取成功，但返回 REVIEW：根扫描达到100条上限，5个LLM缺少reasoning计量。
不能将 REVIEW 写为完整 PASS；需要完整根扫描时调整窗口或上限，不能将缺失 reasoning 当成已核实0。
详细记录见 [迁移验收](../reports/langsmith-smithdb-migration.md)。
