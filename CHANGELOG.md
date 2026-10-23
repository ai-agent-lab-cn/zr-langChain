# Experiment Reliability V1 — 2026-10-23

新增：
- 在线实验：QUOTA_EXHAUSTED 立即终止后续用例；其他异常结构化分类。
- `.runs/chapter08/latest.json`：原子保存匿名批次状态，脚本/模型指纹不一致拒绝复用。
- `--resume-batch`：跳过 PASS；重跑 FAILED/BLOCKED/PENDING 或上轮异常中断的 RUNNING。
- LangSmith Analyzer：`--batch-latest` / `--batch-id`，使用 batch_id 缩小报告范围，支持脱敏 thread_fingerprint。
- 复查文档：新增第八章用例级断点验收讲解。

保持不变：
- `build_graph`、`offline_proposal` 与核心 Graph 的可执行逻辑。
- Mermaid 两张图与第八章既有章节结构。
- 不修改本地原始 Git 仓库，未自动推送 GitHub。

验证：
- 20 组标准库离线 unittest 通过，包括模拟 429/配额、自动停止、批次恢复、模型版本隔离和 Trace 批次过滤。
- 源码 Python 语法编译通过；核心 Graph 定义 AST 与前版一致。
- 尚未在本环境运行真实 LangGraph 或 Gemini/LangSmith 在线集成；需要本地 VS Code 运行。

## Chapter 08 · DeepSeek 适配

- 第八章模型替换为 `ChatDeepSeek` / `DEEPSEEK_MODEL=deepseek-flash`。
- Pydantic 结构化输出采用 LangChain 的 `function_calling`。
- 增加 DeepSeek 402 余额不足识别与测试；429 仍按瞬时限流处理。
- `max_retries=0`：避免 SDK 自动重试耗尽额度的调用。
- 不影响 01–07 章节的 Gemini 代码和历史 Trace；保留 HITL Graph 核心逻辑。
