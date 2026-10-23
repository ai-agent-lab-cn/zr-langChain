# LangSmith Trace 自动验收报告｜第 08 章

- 项目：`zr-langChain`
- 根 Graph Run：**10**
- 实验批次：`chapter08-a558b6d4962047f5`
- 实际 LLM Run：**5**（只统计 `run_type=llm`）
- 验收状态：**REVIEW**（PASS / REVIEW / FAIL）

## Token 汇总

| 输入 | 输出 | 其中 Reasoning | 总 Token |
|---:|---:|---:|---:|
| 2113 | 565 | 0 | 2678 |

> **数据不完整：** 至少一条 LLM Run 缺少 Token 计量；各列仅为已知值合计。Reasoning 缺失不等于 0，也不表示 input/output/total 必然缺失；具体缺失字段见待复核项。

## 根 Run 明细

| Run | 阶段 | LLM Calls | Tokens（已知） | 原始 error | 预期中断 | 实际错误 |
|---|---|---:|---:|---:|---:|---:|
| `chapter08-approve-pause` | pause | 1 | 542 | 1 | 1 | 0 |
| `chapter08-approve-resume` | resume | 0 | 0 | 0 | 0 | 0 |
| `chapter08-reject-pause` | pause | 1 | 550 | 1 | 1 | 0 |
| `chapter08-reject-resume` | resume | 0 | 0 | 0 | 0 | 0 |
| `chapter08-edit-pause` | pause | 1 | 525 | 1 | 1 | 0 |
| `chapter08-edit-resume` | resume | 0 | 0 | 0 | 0 | 0 |
| `chapter08-isolation-A-pause` | pause | 1 | 521 | 1 | 1 | 0 |
| `chapter08-isolation-B-pause` | pause | 1 | 540 | 1 | 1 | 0 |
| `chapter08-isolation-B-resume` | resume | 0 | 0 | 0 | 0 | 0 |
| `chapter08-isolation-A-resume` | resume | 0 | 0 | 0 | 0 | 0 |

原始 error 保留服务端计数；预期中断只在节点、State、同线程同批次及后续成功恢复全部核实后豁免。

## HITL Pause / Resume 核对

| 案例 | 成对 Trace | 可见的验证项 | thread_id 配对 |
|---|---|---|---|
| `approve` | 是 | same_thread_id_verified, same_batch_verified, pause_before_resume_verified, expected_interrupt_verified, interrupt_observed, resume_review_node_observed, state_and_route_validated | verified |
| `edit` | 是 | same_thread_id_verified, same_batch_verified, pause_before_resume_verified, expected_interrupt_verified, interrupt_observed, resume_review_node_observed, state_and_route_validated | verified |
| `isolation-A` | 是 | same_thread_id_verified, same_batch_verified, pause_before_resume_verified, isolation_threads_distinct_verified, expected_interrupt_verified, interrupt_observed, resume_review_node_observed, state_and_route_validated | verified |
| `isolation-B` | 是 | same_thread_id_verified, same_batch_verified, pause_before_resume_verified, isolation_threads_distinct_verified, expected_interrupt_verified, interrupt_observed, resume_review_node_observed, state_and_route_validated | verified |
| `reject` | 是 | same_thread_id_verified, same_batch_verified, pause_before_resume_verified, expected_interrupt_verified, interrupt_observed, resume_review_node_observed, state_and_route_validated | verified |

## 待复核 / 数据限制

- chapter08-isolation-B-pause 有 1 个 LLM Run 缺少 reasoning Token 计量
- chapter08-isolation-A-pause 有 1 个 LLM Run 缺少 reasoning Token 计量
- chapter08-edit-pause 有 1 个 LLM Run 缺少 reasoning Token 计量
- chapter08-reject-pause 有 1 个 LLM Run 缺少 reasoning Token 计量
- chapter08-approve-pause 有 1 个 LLM Run 缺少 reasoning Token 计量

## 报告范围与隐私

- 只生成脱敏摘要：不包含 Prompt、工单正文、草稿、模型回答或 API Key。
- `thread_id` 只用短哈希匹配；未在 Trace 中出现时，明确标记证据不足。
- 本工具是只读 Trace 分析，不负责证明真实的跨进程持久化或企业权限控制。
- 同一时间窗口若含多轮实验，可能出现跨批次结果；建议运行实验后用较短的 `--since-hours`。
